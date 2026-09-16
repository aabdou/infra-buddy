"""Prompt text shared by every implementation.

Behaviour rules only -- deliberately says nothing about how to cite. Each
agent's citation mechanism differs (the API's citations feature returns
sources as structured data; a self-reported result type needs an instruction),
so citation wording belongs with the agent, not here. What is shared is the
part that affects answer quality, which is what makes the agents comparable.
"""

from infra_core.types import Chunk

SYSTEM_PROMPT = """You answer questions about Amazon S3 using only the \
documentation excerpts provided to you.

Rules:
- Answer only from the excerpts. If they do not contain the answer, say so \
plainly rather than filling the gap from memory.
- Be concise. Prefer the documentation's own wording for exact names of \
settings, API calls and permissions."""


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as citable context.

    For agents that pass chunks as plain text. Agents using the API's document
    blocks send chunk text directly and do not need this.
    """
    return "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in chunks)
