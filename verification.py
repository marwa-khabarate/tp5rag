"""Vérifications reproductibles avec les deux serveurs MCP démarrés."""
import asyncio
import base64
from uuid import uuid4

from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from streamlit.testing.v1 import AppTest

from tp5.b_bases import graph as bases
from tp5.b_rag import chatbot
from tp5.d_orchestrateur import graph as sav
from tp5.settings import DATA, ROOT


async def verify_protocol():
    client = MultiServerMCPClient({"sav": {
        "url": "http://127.0.0.1:8001/mcp", "transport": "streamable_http",
    }})
    tools = {tool.name: tool for tool in await client.get_tools()}
    assert set(tools) == {"estimer_devis", "horaires_sav"}
    for name, args, expected in [
        ("estimer_devis", {"pieces": 120, "main_oeuvre": 80}, "200"),
        ("horaires_sav", {}, "09:00"),
    ]:
        tool_result = await tools[name].ainvoke({"type": "tool_call", "name": name, "args": args, "id": name})
        assert expected in tool_result.text, tool_result


async def verify_sav():
    question = "Que signifie le pictogramme sur cette photo du PX-200 ?"
    photo = (DATA / "photos" / "img_px200_01.png").read_bytes()
    for payload in [{"photo_ref": "img_px200_01"}, {"photo_base64": base64.b64encode(photo).decode()}]:
        inputs = {"question": question, **payload}
        trace = [step async for step in sav.astream(inputs, stream_mode="updates")]
        assert [next(iter(step)) for step in trace] == ["orchestrateur", "agent_documentaire", "orchestrateur"], trace
        answer = trace[1]["agent_documentaire"]
        assert "bourrage" in answer["reponse_finale"] and answer["sources"][0]["page"] == 1, answer

    for serial, expected in [("PX200-0042", "11/02/2028"), ("PX200-0011", "31/03/2024"),
                             ("PX200-9999", "absent"), ("", "Indiquez")]:
        inputs = {"question": f"Quelle est ma garantie ? {serial}"}
        trace = [step async for step in sav.astream(inputs, stream_mode="updates")]
        assert [next(iter(step)) for step in trace] == ["orchestrateur", "agent_garantie", "orchestrateur"], trace
        assert expected in trace[1]["agent_garantie"]["reponse_finale"], trace


def verify_graphs():
    result = bases.invoke({"tentatives": 0, "limite": 3, "journal": []})
    assert result["tentatives"] == 3 and len(result["journal"]) == 5
    config = {"configurable": {"thread_id": str(uuid4())}}
    def query(text):
        return {"messages": [HumanMessage(text)]}
    assert "Bonjour" in chatbot.invoke(query("Bonjour"), config)["messages"][-1].text
    assert "Figure 1" in chatbot.invoke(query("Code P02 sur le PX-200"), config)["messages"][-1].text
    updates = list(chatbot.stream(query("Et si internet est coupé ?"), config, stream_mode="updates"))
    assert any("reformuler" in step for step in updates)
    assert "W14" in updates[-1]["repondre"]["messages"][-1].text


def verify_streamlit():
    app = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=90).run()
    app.chat_input[0].set_value("PX-200 : que signifie P02 ?").run()
    assert not app.exception and any("bourrage" in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Dépannage photo et garantie").run()
    app.chat_input[0].set_value("Garantie PX200-0042 ?").run()
    assert not app.exception and any("agent_garantie" in item.value for item in app.markdown)

    app.sidebar.checkbox[0].set_value(True).run()
    app.chat_input[0].set_value("Que signifie ce pictogramme sur le PX-200 ?").run()
    assert not app.exception and any("Figure 1" in item.value for item in app.markdown)


if __name__ == "__main__":
    verify_graphs()
    print("LangGraph B1/B2 : OK")
    asyncio.run(verify_protocol())
    print("Découverte et appels MCP en Streamable HTTP : OK")
    asyncio.run(verify_sav())
    print("Orchestrateur, photo, registre et routage exclusif : OK")
    verify_streamlit()
    print("Streamlit, deux parcours : OK")

