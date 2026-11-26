"""Partie A — Agents, mémoire, outils et middlewares LangChain."""
import os
import re
from dataclasses import dataclass
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ToolCallLimitMiddleware, after_model, before_agent,
    dynamic_prompt, wrap_model_call, wrap_tool_call,
)
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from tp5.corpus import normal, render_answer, retrieve
from tp5.settings import chat


@dataclass
class Preferences:
    style: str = "clair et concis"
    model_tier: str = "auto"


@tool
def estimer_reparation(pieces: float, main_oeuvre: float) -> str:
    """Calcule une estimation SAV en dirhams à partir des pièces et de la main-d'œuvre."""
    if not (0 <= pieces <= 100_000 and 0 <= main_oeuvre <= 100_000):
        raise ValueError("Montant hors limites.")
    return f"Estimation indicative : {pieces + main_oeuvre:.2f} MAD."


@tool
def chercher_manuel(question: str) -> str:
    """Consulte les manuels des modèles PX-200 et LX-40. Retourne un passage et sa référence."""
    return render_answer(retrieve(question))


@dynamic_prompt
def dynamic_profile(request):
    preferences = request.runtime.context or Preferences()
    return (
        f"Tu es l'assistant SAV d'un distributeur. Réponds en français, avec un style {preferences.style}. "
        "Pour calculer une estimation, appelle estimer_reparation. "
        "Utilise les outils selon leur description. N’invente pas un résultat d’outil."
    )


@wrap_model_call
def dynamic_model(request, handler):
    preferences = request.runtime.context or Preferences()
    tier = preferences.model_tier
    if tier == "auto":
        size = sum(len(message.text) for message in request.messages if isinstance(message, HumanMessage))
        tier = "advanced" if size > 280 else "standard"
    return handler(request.override(model=chat(tier)))


@wrap_tool_call
def tool_error_handling(request, handler):
    try:
        result = handler(request)
    except Exception:
        return ToolMessage(
            content="L’outil a échoué. Vérifie ses arguments ou signale son indisponibilité.",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    if not isinstance(result, ToolMessage) or result.status != "error":
        return result
    return result.model_copy(update={"content": "Arguments invalides ou outil indisponible."})


@before_agent(can_jump_to=["end"])
def guardrails(state: AgentState, runtime: Runtime):
    last = state["messages"][-1]
    sensitive = r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}\b"
    if isinstance(last, HumanMessage) and re.search(sensitive, last.text):
        return {"messages": [AIMessage("Retirez les identifiants secrets avant d’envoyer la demande.")], "jump_to": "end"}
    return None


def predefined_tools(include_tavily: bool = False):
    tools = [DuckDuckGoSearchRun()]
    if include_tavily:
        if not os.getenv("TAVILY_API_KEY"):
            raise RuntimeError("Renseignez TAVILY_API_KEY dans .env pour activer Tavily.")
        tools.append(TavilySearch(max_results=3))

    repl = PythonREPLTool()

    @tool("python_repl")
    def execute_calculation(query: str) -> str:
        """Exécute print suivi d'un calcul arithmétique simple, par exemple print(12 * 7)."""
        match = re.fullmatch(r"print\(\s*([0-9]{1,8}(?:\.[0-9]{1,5})?\s*[-+*/]\s*[0-9]{1,8}(?:\.[0-9]{1,5})?)\s*\)", query)
        if not match:
            raise ValueError("Utiliser print(a + b), print(a - b), print(a * b) ou print(a / b).")
        return repl.invoke(f"print({match[1]})")

    return [*tools, execute_calculation]


def simple_agent(*, memory=False):
    return create_agent(chat(), system_prompt="Tu es un assistant SAV. Réponds en français.",
                        checkpointer=InMemorySaver() if memory else None)


def enhanced_agent(*, web=False, tavily=False):
    tools = [estimer_reparation]
    if web or tavily:
        tools.extend(predefined_tools(tavily))
    approval = {"estimer_reparation": {"allowed_decisions": ["approve", "reject"]}}
    if web or tavily:
        approval["python_repl"] = {"allowed_decisions": ["approve", "reject"]}
    return create_agent(
        chat(),
        tools=tools,
        middleware=[guardrails, dynamic_profile, dynamic_model, tool_error_handling,
                    ToolCallLimitMiddleware(run_limit=5, exit_behavior="end"),
                    HumanInTheLoopMiddleware(interrupt_on=approval)],
        context_schema=Preferences,
        checkpointer=InMemorySaver(),
    )


@after_model
def ground_rag_answer(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    reply = messages[-1]
    if reply.tool_calls:
        return None
    turn = next(i for i in reversed(range(len(messages))) if isinstance(messages[i], HumanMessage))
    request = messages[turn].text
    evidence = next((message for message in reversed(messages[turn + 1:])
                     if isinstance(message, ToolMessage) and message.name == "chercher_manuel"), None)
    if evidence is not None:
        return {"messages": [AIMessage(evidence.text, id=reply.id)]}
    if normal(request).strip(" !?.") in {"bonjour", "salut", "bonsoir", "merci"}:
        return None
    # Si le petit modèle propose du texte avant la recherche, on impose la consultation.
    return {"messages": [AIMessage("", id=reply.id, tool_calls=[{
        "id": str(uuid4()), "name": "chercher_manuel", "args": {"question": request},
    }])]}


def rag_agent():
    return create_agent(
        chat(), tools=[chercher_manuel], checkpointer=InMemorySaver(),
        system_prompt=(
            "Tu es le chatbot documentaire SAV. Pour toute question technique, appelle chercher_manuel "
            "avec le modèle et le symptôme. Reprends le passage et sa référence dans ta réponse. "
            "Si l’outil ne fournit aucun résultat, demande une précision. Réponds en français."
        ),
        middleware=[ground_rag_answer, ToolCallLimitMiddleware(run_limit=3, exit_behavior="end")],
    )

