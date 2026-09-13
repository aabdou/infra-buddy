"""Retrieval through LangChain's own vector-store abstraction.

Approach: point langchain-chroma at the same store and build a retriever
chain, to see what the abstraction buys over calling Chroma directly.

The embedding function must match the one used at ingest time
(infra_core.config.EMBEDDING_MODEL). Handing this retriever a different
embeddings object is a silent failure, not an error.
"""

from infra_core.types import Answer


def answer(question: str) -> Answer:
    raise NotImplementedError(
        "agent-langchain: langchain-chroma retriever + chain, matching EMBEDDING_MODEL"
    )
