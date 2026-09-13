"""Retrieval-then-generate, with no framework in between.

Approach: call infra_core.retrieval.search() directly, format the chunks into
the prompt, and make a single Messages API call. The model never decides when
to search -- retrieval always happens, exactly once, before generation.
"""

from infra_core.types import Answer


def answer(question: str) -> Answer:
    raise NotImplementedError(
        "agent-raw: stuff retrieved chunks into a single Messages API call"
    )
