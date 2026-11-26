"""Partie D — Agent RAG multimodal derrière l'outil MCP."""
import base64
import binascii
import os
from typing import NotRequired, TypedDict

from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from PIL import Image

from tp5.corpus import Hit, detect_product, match_image, render_answer, retrieve
from tp5.settings import DATA, chat

MAX_PHOTO = 2_000_000
PHOTO_REFERENCES = {"img_px200_01": DATA / "photos" / "img_px200_01.png"}


class State(TypedDict):
    question: str
    photo_ref: NotRequired[str]
    photo_base64: NotRequired[str]
    visual_hit: NotRequired[Hit | None]
    visual_note: NotRequired[str]
    visual_query: NotRequired[str]
    hits: NotRequired[list[Hit]]
    response: NotRequired[dict]


def analyser_photo(state: State):
    reference = state.get("photo_ref")
    encoded = state.get("photo_base64")
    if reference and encoded:
        return {"visual_note": "Fournissez une photo ou une référence, au choix."}
    if reference:
        path = PHOTO_REFERENCES.get(reference)
        if path is None:
            return {"visual_note": f"Référence photo inconnue ({reference})."}
        raw = path.read_bytes()
    else:
        if not encoded or len(encoded) > 4 * ((MAX_PHOTO + 2) // 3):
            return {"visual_note": "Photo absente ou trop volumineuse."}
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return {"visual_note": "Photo illisible."}
        if len(raw) > MAX_PHOTO:
            return {"visual_note": "Photo trop volumineuse."}
    try:
        match = match_image(raw, detect_product(state["question"]))
    except (ValueError, OSError, Image.DecompressionBombError):
        return {"visual_note": "Photo illisible ou pictogramme indétectable."}
    if match:
        return {"visual_hit": match, "visual_note": f"Pictogramme rapproché de la figure du manuel ({match.title})."}
    if os.getenv("ENABLE_OLLAMA_VISION", "").lower() == "true":
        try:
            mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
            vision = chat("vision").invoke([HumanMessage(content=[
                {"type": "text", "text": "Décris uniquement la forme du pictogramme en français et son code s’il est visible. Ne déduis aucune panne."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}},
            ])])
            return {"visual_query": vision.text, "visual_note": "Recherche affinée à partir d’une description visuelle du pictogramme."}
        except Exception:
            return {"visual_note": "Analyse visuelle indisponible. Décrivez le pictogramme ou indiquez son code."}
    return {"visual_note": "Pictogramme non reconnu. Décrivez-le ou indiquez son code."}


def chercher_preuves(state: State):
    match = state.get("visual_hit")
    query = " ".join([state["question"], state.get("visual_query", ""), match.title if match else ""])
    hits = retrieve(query)
    if match:
        hits = [match, *[hit for hit in hits if hit != match]][:3]
    return {"hits": hits}


def repondre(state: State):
    hits = state["hits"]
    note = state.get("visual_note")
    response = render_answer(hits)
    if note:
        response += f"\n\n{note}"
    return {"response": {
        "reponse": response,
        "sources": [{"manuel": hit.file, "page": hit.page, "figure": hit.caption} for hit in hits[:1]],
    }}


def build():
    builder = StateGraph(State)
    builder.add_node("analyser_photo", analyser_photo)
    builder.add_node("chercher_preuves", chercher_preuves)
    builder.add_node("repondre", repondre)
    builder.add_conditional_edges(
        START, lambda state: "analyser_photo" if state.get("photo_ref") or state.get("photo_base64") else "chercher_preuves",
        ["analyser_photo", "chercher_preuves"],
    )
    builder.add_edge("analyser_photo", "chercher_preuves")
    builder.add_edge("chercher_preuves", "repondre")
    builder.add_edge("repondre", END)
    return builder.compile()


graph = build()

