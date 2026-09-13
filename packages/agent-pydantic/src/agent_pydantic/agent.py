"""Retrieval as an agent tool.

Approach: register infra_core.retrieval.search() as an @agent.tool so the
model chooses whether and how often to search, and use a Pydantic result type
so citations come back structured instead of parsed out of prose.
"""

from infra_core.types import Answer


def answer(question: str) -> Answer:
    raise NotImplementedError(
        "agent-pydantic: register search() as @agent.tool, structured result type"
    )
