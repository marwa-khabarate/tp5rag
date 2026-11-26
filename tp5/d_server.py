"""Expose le RAG multimodal sur /mcp via Streamable HTTP."""
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from tp5.d_rag_multimodal import graph

server = FastMCP("TP5 — Documentation multimodale", host="127.0.0.1",
                 port=8002, stateless_http=True, json_response=True)


class Source(BaseModel):
    manuel: str
    page: int
    figure: str


class RagResponse(BaseModel):
    reponse: str
    sources: list[Source]


@server.tool()
async def rag_multimodal(question: str, photo_ref: str | None = None, photo_base64: str | None = None) -> RagResponse:
    """Recherche des passages et des figures. Prend une question et une photo facultative."""
    result = await graph.ainvoke({"question": question, "photo_ref": photo_ref, "photo_base64": photo_base64})
    return RagResponse.model_validate(result["response"])


if __name__ == "__main__":
    server.run(transport="streamable-http")

