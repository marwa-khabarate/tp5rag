"""Partie C — Agent LangChain et découverte dynamique des outils MCP."""
import argparse
import asyncio
import os
import re
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, after_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.runtime import Runtime

from tp5.corpus import normal
from tp5.settings import chat


@after_model
async def recover_known_tool(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    reply = messages[-1]
    if reply.tool_calls or any(isinstance(message, ToolMessage) for message in messages):
        return None
    question = normal(next(message.text for message in reversed(messages) if isinstance(message, HumanMessage)))
    if re.search(r"horaire|ouvert|fermet", question):
        tool_name, arguments = "horaires_sav", {}
    elif re.search(r"devis|estimation", question):
        amounts = re.findall(r"[+-]?\d+(?:[.,]\d+)?", question)
        if len(amounts) != 2:
            return None
        first, second = (float(value.replace(",", ".")) for value in amounts)
        tool_name, arguments = "estimer_devis", {"pieces": first, "main_oeuvre": second}
    else:
        return None
    return {"messages": [AIMessage("", id=reply.id, tool_calls=[{
        "id": str(uuid4()), "name": tool_name, "args": arguments,
    }])]}


async def ask(question: str):
    client = MultiServerMCPClient({
        "services_sav": {
            "transport": "streamable_http",
            "url": os.getenv("MCP_DEMO_URL", "http://127.0.0.1:8001/mcp"),
        }
    })
    tools = await client.get_tools()
    agent = create_agent(
        chat("advanced"), tools=tools,
        system_prompt="Tu es l’assistant SAV. Appelle l’outil MCP pertinent avant de répondre. Réponds en français.",
        middleware=[recover_known_tool, ToolCallLimitMiddleware(run_limit=4, exit_behavior="end")],
    )
    return await agent.ainvoke({"messages": [{"role": "user", "content": question}]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    result = asyncio.run(ask(parser.parse_args().question))
    for message in result["messages"][1:]:
        message.pretty_print()

