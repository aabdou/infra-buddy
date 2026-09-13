"""The contract every agent implementation speaks.

Each agent package exposes `answer(question: str) -> Answer`. Keeping that
signature identical is what lets one eval harness score all of them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrieved passage from the vector store."""

    id: str
    text: str
    distance: float  # Chroma returns L2 distance: lower is closer.


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
