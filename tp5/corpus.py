import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

import chromadb
import pymupdf as fitz
import numpy as np
from PIL import Image, ImageOps

from tp5.settings import DATA

PRODUCT = re.compile(r"\b([A-Z]{2})[ -]?(\d{2,3})\b", re.I)
SIGNAL = re.compile(r"\b[A-Z]\d{2}\b")
STOP = set("avec dans une les des est sur quel quoi quand comment mon mes son puis pour que qui cela affiche voyant imprimante signifie quel quelle".split())
SYNONYMS = {"coince": "bourrage", "bloquee": "bourrage", "bloque": "bourrage",
            "cartouche": "toner", "sansfil": "wifi", "reseau": "wifi",
            "chauffe": "temperature", "surchauffe": "temperature", "clignote": "clignotant"}


def normal(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def keywords(text: str) -> set[str]:
    text = re.sub(r"wi[ -]fi", "wifi", normal(text))
    text = PRODUCT.sub("", text)
    return {SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", text)
            if len(word) >= 3 and word not in STOP}


def embed(text: str) -> list[float]:
    vector = [0.0] * 512
    for word in keywords(text):
        index = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "little") % len(vector)
        vector[index] += 1
    return vector


@dataclass(frozen=True)
class Hit:
    model: str
    file: str
    page: int
    title: str
    passage: str
    caption: str
    figure: bytes

    @property
    def source(self) -> str:
        return f"{self.file}, p. {self.page}, {self.caption.split(' - ')[0]}"


@lru_cache(maxsize=1)
def index():
    records = {}
    chroma = chromadb.EphemeralClient()
    passages = chroma.get_or_create_collection("passages", embedding_function=None)
    figures = chroma.get_or_create_collection("figures", embedding_function=None)
    for path in sorted((DATA / "manuels").glob("*.pdf")):
        with fitz.open(path) as pdf:
            for page in pdf:
                rows = [line for line in page.get_text().splitlines() if line.strip()]
                caption = next(i for i, line in enumerate(rows) if line.startswith("Figure "))
                image_ref = page.get_images()[0][0]
                key = f"{path.stem}-{page.number + 1}"
                records[key] = Hit(
                    path.stem.removeprefix("manuel_"), path.name, page.number + 1,
                    rows[2], " ".join(rows[3:caption]), rows[caption],
                    pdf.extract_image(image_ref)["image"],
                )
    if not records:
        raise RuntimeError("Le corpus est absent. Lancez python -m tp5.fixtures.")
    for collection, attr in [(passages, "passage"), (figures, "caption")]:
        docs = [getattr(hit, attr) for hit in records.values()]
        collection.upsert(ids=list(records), documents=docs,
                          embeddings=[embed(f"{hit.title} {doc}") for hit, doc in zip(records.values(), docs)],
                          metadatas=[{"model": hit.model, "page": hit.page} for hit in records.values()])
    return passages, figures, records


def detect_product(text: str) -> str | None:
    match = PRODUCT.search(text)
    return f"{match[1].upper()}-{match[2]}" if match else None


def retrieve(query: str) -> list[Hit]:
    passages, figures, records = index()
    words = keywords(query)
    if not words:
        return []
    where = {"model": detect_product(query)} if detect_product(query) else None
    matches = {}
    for collection in (passages, figures):
        ids = collection.query(query_embeddings=[embed(query)], n_results=len(records), where=where)["ids"][0]
        for rank, key in enumerate(ids):
            record = records[key]
            overlap = words & keywords(f"{record.title} {record.passage}")
            if overlap:
                exact_code = bool(set(SIGNAL.findall(query.upper())) & set(SIGNAL.findall(record.title)))
                matches[key] = matches.get(key, 0) + len(overlap) + 1 / (rank + 1) + 10 * exact_code
    return [records[key] for key in sorted(matches, key=matches.get, reverse=True)[:3]]


def visual_signature(raw: bytes) -> np.ndarray:
    with Image.open(BytesIO(raw)) as source:
        if source.format not in {"PNG", "JPEG"} or source.width * source.height > 16_000_000:
            raise ValueError("Utilisez une photo PNG ou JPEG de moins de 16 mégapixels.")
        image = ImageOps.autocontrast(ImageOps.grayscale(source))
        mask = np.asarray(image) < 100
        # La photo fournie est cadrée sur l'écran clair, même si son boîtier est sombre.
        bright = np.asarray(image) > 180
        rows, cols = np.where(bright)
        if len(rows) == 0:
            raise ValueError("Aucun écran clair détecté.")
        mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        rows, cols = np.where(mask)
        if len(rows) == 0:
            raise ValueError("Aucun pictogramme détecté.")
        mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        return np.asarray(Image.fromarray(mask).resize((32, 32), Image.Resampling.NEAREST))


def match_image(raw: bytes, model: str | None) -> Hit | None:
    _, _, records = index()
    signature = visual_signature(raw)
    candidates = [(np.mean(signature != visual_signature(hit.figure)), hit)
                  for hit in records.values() if model is None or hit.model == model]
    candidates.sort(key=lambda pair: pair[0])
    if candidates and candidates[0][0] < 0.19 and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 0.08):
        return candidates[0][1]
    return None


def render_answer(hits: list[Hit]) -> str:
    if not hits:
        return "Je n’ai pas trouvé de passage pertinent. Indiquez le modèle et le code ou décrivez le voyant."
    best = hits[0]
    return f"{best.passage}\n\n**Source —** {best.source}."

