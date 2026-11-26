"""Partie C — Serveur MCP de démonstration, transport Streamable HTTP."""
import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("TP5 — services SAV", host=os.getenv("MCP_BIND", "127.0.0.1"),
                 port=8001, stateless_http=True, json_response=True)


@server.tool()
def estimer_devis(pieces: float, main_oeuvre: float) -> dict:
    """Additionne le prix des pièces et de la main-d'œuvre, en MAD."""
    if not (0 <= pieces <= 100_000 and 0 <= main_oeuvre <= 100_000):
        raise ValueError("Les montants doivent être compris entre 0 et 100 000 MAD.")
    return {"total": round(pieces + main_oeuvre, 2), "devise": "MAD"}


@server.tool()
def horaires_sav() -> dict:
    """Donne les horaires fictifs du service après-vente."""
    return {"jours": "lundi à vendredi", "ouverture": "09:00", "fermeture": "17:30", "zone": "Africa/Casablanca"}


if __name__ == "__main__":
    server.run(transport="streamable-http")

