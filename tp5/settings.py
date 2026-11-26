import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")


def chat(tier: str = "standard") -> ChatOllama:
    naming = {
        "standard": ("OLLAMA_MODEL", "qwen2.5:1.5b"),
        "advanced": ("OLLAMA_ADVANCED_MODEL", "qwen3-vl:2b-instruct"),
        "vision": ("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct"),
    }
    variable, default = naming[tier]
    return ChatOllama(
        model=os.getenv(variable, default),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0,
        num_predict=350,
    )

