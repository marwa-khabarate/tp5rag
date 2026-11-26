"""Interface Streamlit des parties B et D."""
import asyncio
import base64
from uuid import uuid4

import streamlit as st
from langchain.messages import HumanMessage

from tp5.b_rag import chatbot
from tp5.d_orchestrateur import graph as sav
from tp5.d_rag_multimodal import MAX_PHOTO

st.set_page_config(page_title="Atelier SAV", page_icon="🛠️")
st.title("Atelier SAV")
mode = st.sidebar.radio("Parcours", ["Assistant documentaire", "Dépannage photo et garantie"])
is_sav = mode == "Dépannage photo et garantie"

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
session = st.session_state.sessions.setdefault(mode, {"thread": str(uuid4()), "history": []})
if st.sidebar.button("Nouvelle conversation", width="stretch"):
    st.session_state.sessions.pop(mode)
    st.rerun()
if is_sav:
    with st.sidebar:
        photo = st.file_uploader("Photo de l’écran", type=["png", "jpg", "jpeg"], key=f"photo-{session['thread']}")
        example = st.checkbox("Utiliser la photo de démonstration", value=False, disabled=photo is not None)
    st.caption("Pour consulter la garantie, joignez le numéro de série à votre question.")
else:
    st.caption("Posez une question sur les manuels PX-200 et LX-40.")

for role, content in session["history"]:
    st.chat_message(role).markdown(content)
if question := st.chat_input("Ex. Quel est le code P02 sur le PX-200 ?"):
    if is_sav and photo and photo.size > MAX_PHOTO:
        st.error("La photo dépasse 2 Mo.")
        st.stop()
    session["history"].append(("user", question))
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"), st.spinner("Consultation en cours…"):
        if is_sav:
            inputs = {"question": question}
            if photo is not None:
                inputs["photo_base64"] = base64.b64encode(photo.getvalue()).decode("ascii")
            elif example:
                inputs["photo_ref"] = "img_px200_01"
            result = asyncio.run(sav.ainvoke(inputs))
            response = result["reponse_finale"] + f"\n\n*{result['agent']}*"
        else:
            result = chatbot.invoke({"messages": [HumanMessage(question)]},
                                    {"configurable": {"thread_id": session["thread"]}})
            response = result["messages"][-1].text
        st.markdown(response)
    session["history"].append(("assistant", response))

