"""Orchestration SAV LangGraph : une demande, un agent spécialisé."""
import argparse
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from tp5.corpus import normal
from tp5.d_garantie import SERIAL, consulter_garantie
from tp5.d_rag_multimodal import MAX_PHOTO


class Request(TypedDict):
    question: str
    photo_ref: NotRequired[str]
    photo_base64: NotRequired[str]


class State(Request, total=False):
    categorie: Literal["documentation", "garantie"]
    agent: str
    reponse_finale: str
    sources: list[dict]


class SpecialistResult(TypedDict):
    reponse_finale: str
    sources: list[dict]


@tool
async def appeler_rag_mcp(question: str, photo_ref: str | None = None, photo_base64: str | None = None) -> dict:
    """Appelle rag_multimodal sur le serveur MCP distant."""
    url = os.getenv("MCP_RAG_URL", "http://127.0.0.1:8002/mcp")
    try:
        async with asyncio.timeout(60):
            async with streamable_http_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("rag_multimodal", {
                        "question": question, "photo_ref": photo_ref, "photo_base64": photo_base64,
                    })
        if result.isError:
            raise ValueError("Échec de l’outil documentaire.")
        payload = result.structuredContent
        if payload is None and len(result.content) == 1 and isinstance(result.content[0], types.TextContent):
            payload = json.loads(result.content[0].text)
        if not isinstance(payload, dict) or not isinstance(payload.get("reponse"), str):
            raise ValueError("Réponse documentaire invalide.")
        return payload
    except Exception:
        return {"reponse": "Le service documentaire MCP est momentanément indisponible.", "sources": []}


def orchestrateur(state: State):
    if state.get("reponse_finale"):
        return {}
    question = normal(state["question"])
    garantie = bool(re.search(r"\b(garanti(?:e|es)?|couverture|couvert(?:e|s|es)?|date d['’]achat)\b", question))
    panne = bool(re.search(r"\b(voyant|pictogramme|panne|code|papier|clignotant|affiche|bloque|depannage)\b", question))
    if not garantie and not panne:
        garantie = bool(SERIAL.search(state["question"]) or re.search(r"\bnumero de serie\b", question))
    category = "garantie" if garantie else "documentation"
    return {"categorie": category, "agent": f"agent_{'garantie' if garantie else 'documentaire'}"}


def route(state: State):
    return END if state.get("reponse_finale") else state["agent"]


def specialiste(replier, outil):
    """Construit un sous-graphe avec décision d’appel, ToolNode et restitution."""
    class AgentState(MessagesState, total=False):
        question: str
        photo_ref: str
        photo_base64: str
        reponse_finale: str
        sources: list[dict]

    def decider_appel(state):
        return {"messages": [AIMessage("", tool_calls=[
            {"name": outil.name, "id": "consultation-sav", "args": replier(state)}
        ])]}

    def restituer(state):
        payload = json.loads(state["messages"][-1].text)
        return {"reponse_finale": payload["reponse"], "sources": payload.get("sources", [])}

    workflow = StateGraph(AgentState, input_schema=Request, output_schema=SpecialistResult)
    workflow.add_node("decider_appel", decider_appel)
    workflow.add_node("outil", ToolNode([outil]))
    workflow.add_node("restituer", restituer)
    workflow.add_edge(START, "decider_appel")
    workflow.add_edge("decider_appel", "outil")
    workflow.add_edge("outil", "restituer")
    workflow.add_edge("restituer", END)
    return workflow.compile()


def questions_documentaires(state):
    return {"question": state["question"], "photo_ref": state.get("photo_ref"), "photo_base64": state.get("photo_base64")}


def numero_de_serie(state):
    serial = SERIAL.search(state["question"])
    return {"numero_serie": serial[0] if serial else ""}


builder = StateGraph(State, input_schema=Request)
builder.add_node("orchestrateur", orchestrateur)
builder.add_node("agent_documentaire", specialiste(questions_documentaires, appeler_rag_mcp))
builder.add_node("agent_garantie", specialiste(numero_de_serie, consulter_garantie))
builder.add_edge(START, "orchestrateur")
builder.add_conditional_edges("orchestrateur", route, ["agent_documentaire", "agent_garantie", END])
builder.add_edge("agent_documentaire", "orchestrateur")
builder.add_edge("agent_garantie", "orchestrateur")
graph = builder.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--photo-ref")
    group.add_argument("--photo", type=Path, help="Fichier PNG ou JPEG, maximum 2 Mo.")
    args = parser.parse_args()
    inputs: Request = {"question": args.question}
    if args.photo_ref:
        inputs["photo_ref"] = args.photo_ref
    if args.photo:
        if args.photo.stat().st_size > MAX_PHOTO:
            parser.error("La photo dépasse 2 Mo.")
        inputs["photo_base64"] = base64.b64encode(args.photo.read_bytes()).decode("ascii")

    async def run():
        async for step in graph.astream(inputs, stream_mode="updates"):
            print(json.dumps(step, ensure_ascii=False))

    asyncio.run(run())

