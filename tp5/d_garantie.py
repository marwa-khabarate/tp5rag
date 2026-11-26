"""Outil local : lecture seule d'un registre fictif."""
import csv
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool

from tp5.settings import DATA

SERIAL = re.compile(r"\b([A-Z]{2,5})\s*(\d{2,4})\s*-\s*(\d{4})\b", re.I)


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Lit le statut d'une garantie dans le registre local, à partir du numéro de série."""
    match = SERIAL.fullmatch(numero_serie.strip())
    if match is None:
        return {"statut": "invalide", "reponse": "Indiquez un numéro de série, par exemple PX200-0042."}
    serial = f"{match[1].upper()}{match[2]}-{match[3]}"
    with (DATA / "garanties.csv").open(encoding="utf-8", newline="") as register:
        entry = next((row for row in csv.DictReader(register) if row["numero_serie"] == serial), None)
    if entry is None:
        return {"statut": "introuvable", "reponse": f"Le numéro {serial} est absent du registre fictif."}
    start = date.fromisoformat(entry["debut"])
    end = date.fromisoformat(entry["fin"])
    try:
        today = datetime.now(ZoneInfo("Africa/Casablanca")).date()
    except ZoneInfoNotFoundError:
        today = date.today()
    if today < start:
        status, description = "a_venir", "commence le"
        relevant_date = start
    elif today <= end:
        status, description = "active", "est active jusqu’au"
        relevant_date = end
    else:
        status, description = "expiree", "a expiré le"
        relevant_date = end
    return {"statut": status, "date_debut": start.isoformat(), "date_fin": end.isoformat(),
            "reponse": f"La garantie de {serial} ({entry['modele']}) {description} {relevant_date:%d/%m/%Y}."}

