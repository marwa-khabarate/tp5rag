"""Partie B2 — Chatbot Agentic RAG, configurable dans LangGraph Studio."""
import json
from typing import NotRequired
from uuid import uuid4

from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tp5.corpus import detect_product, keywords, normal, render_answer, retrieve


@tool
def rechercher(query: str) -> str:
    """Recherche des passages et figures associés dans les manuels SAV."""
    hits = retrieve(query)
    return json.dumps({"relevant": bool(hits), "answer": render_answer(hits)}, ensure_ascii=False)


class State(MessagesState):
    query: NotRequired[str]
    attempts: NotRequired[int]


def choose(state: State):
    history = [m.text for m in state["messages"] if isinstance(m, HumanMessage)]
    question = history[-1].strip() if history else ""
    if normal(question).strip(" !?.") in {"bonjour", "salut", "bonsoir", "merci", "hello"}:
        return {"messages": [AIMessage("Bonjour ! Indiquez le modèle et le symptôme de votre appareil.")], "attempts": 0, "query": ""}
    previous_model = next((model for message in reversed(history[:-1]) if (model := detect_product(message))), "")
    query = question if detect_product(question) else f"{previous_model} {question}".strip()
    return {"query": query, "attempts": 0}


def search(state: State):
    attempt = state["attempts"] + 1
    call = {"name": "rechercher", "args": {"query": state["query"]}, "id": str(uuid4())}
    return {"messages": [AIMessage("", tool_calls=[call])], "attempts": attempt}


def grade(state: State):
    result = json.loads(state["messages"][-1].text)
    return "repondre" if result["relevant"] or state["attempts"] >= 2 else "reformuler"


def rewrite(state: State):
    words = normal(state["query"])
    for pattern, rewrite_as in [
        ("papier", "bourrage P02"), ("cartouch", "toner T12"),
        ("internet", "wifi W14"), ("sans fil", "wifi W14"),
        ("chaud", "temperature H01"),
    ]:
        if pattern in words:
            return {"query": f"{detect_product(state['query']) or ''} {rewrite_as}".strip()}
    # Retire les mots de contexte et conserve uniquement les termes techniques.
    return {"query": " ".join([detect_product(state["query"]) or "", *sorted(keywords(words))]).strip()}


def answer(state: State):
    result = json.loads(state["messages"][-1].text)
    return {"messages": [AIMessage(result["answer"])]}


def build(checkpointer=None):
    flow = StateGraph(State)
    flow.add_node("decider", choose)
    flow.add_node("demander_recherche", search)
    flow.add_node("outils", ToolNode([rechercher]))
    flow.add_node("reformuler", rewrite)
    flow.add_node("repondre", answer)
    flow.add_edge(START, "decider")
    flow.add_conditional_edges("decider", lambda state: "demander_recherche" if state["query"] else END,
                               ["demander_recherche", END])
    flow.add_edge("demander_recherche", "outils")
    flow.add_conditional_edges("outils", grade, ["repondre", "reformuler"])
    flow.add_edge("reformuler", "demander_recherche")
    flow.add_edge("repondre", END)
    return flow.compile(checkpointer=checkpointer)


graph = build()
chatbot = build(InMemorySaver())

