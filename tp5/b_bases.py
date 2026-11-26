"""Partie B1 — État typé, réducteur, arêtes, boucle et checkpoints."""
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    tentatives: int
    limite: int
    journal: Annotated[list[str], add]


def verifier_etat(state: State):
    count = state["tentatives"] + 1
    return {"tentatives": count, "journal": [f"Diagnostic {count}"]}


def partir_ou_attendre(state: State):
    return END if state["tentatives"] >= state["limite"] else "attendre"


def attendre(state: State):
    return {"journal": ["Nouvelle tentative demandée"]}


def constituer_graphe(checkpointer=None):
    flow = StateGraph(State)
    flow.add_node("verifier_etat", verifier_etat)
    flow.add_node("attendre", attendre)
    flow.add_edge(START, "verifier_etat")
    flow.add_conditional_edges("verifier_etat", partir_ou_attendre, ["attendre", END])
    flow.add_edge("attendre", "verifier_etat")
    return flow.compile(checkpointer=checkpointer)


graph = constituer_graphe()

if __name__ == "__main__":
    saved = constituer_graphe(InMemorySaver())
    config = {"configurable": {"thread_id": "diagnostic-01"}}
    print(graph.get_graph().draw_mermaid())
    for update in saved.stream({"tentatives": 0, "limite": 3, "journal": []}, config):
        print(update)
    print("Checkpoint :", saved.get_state(config).values)

