"""Prompt text shared by every implementation.

Defined once so "answers must cite their source chunk" means the same thing
in all three agents.
"""

from infra_core.types import Chunk

SYSTEM_PROMPT = """You answer questions about Amazon S3 using only the \
documentation excerpts provided to you.

Rules:
- Answer only from the excerpts. If they do not contain the answer, say so \
plainly rather than filling the gap from memory.
- Cite the chunk id you used, in square brackets, immediately after the claim \
it supports. For example: [aws-s3-dev-421]
- Be concise. Prefer the documentation's own wording for exact names of \
settings, API calls and permissions."""


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as citable context."""
    return "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in chunks)
