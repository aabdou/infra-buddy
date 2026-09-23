"""Retrieval-then-generate, with no framework in between.

Approach: call infra_core.retrieval.search() directly, format the chunks into
the prompt, and make a single Messages API call. The model never decides when
to search -- retrieval always happens, exactly once, before generation.
"""

from dataclasses import dataclass

import anthropic
from anthropic.types import DocumentBlockParam
from infra_core.blocks import to_question, to_user_turn
from infra_core.prompts import SYSTEM_PROMPT
from infra_core.retrieval import search
from infra_core.types import Answer, Chunk, Citation


@dataclass(frozen=True)
class Message:
    content: list[Chunk]
    question: str


messages: list[Message] = []
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def to_document(chunk: Chunk) -> DocumentBlockParam:
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": chunk.text,
        },
        "title": chunk.id,
        "citations": {"enabled": True},
    }


def answer(question: str) -> Answer:
    chunks = search(question)
    messages = [to_document(c) for c in chunks] + [to_question(question)]

    result = _get_client().messages.create(
        messages=[to_user_turn(messages)],
        system=SYSTEM_PROMPT,
        model="claude-opus-5",
        max_tokens=16000,
    )

    if result.stop_reason != "end_turn":
        raise RuntimeError(result.stop_reason)

    body: list[str] = []
    citations: list[Citation] = []
    for block in result.content:  # list[ContentBlock]
        if block.type == "text":
            body.append(block.text)
            for citation in block.citations or []:  # list[TextCitation] | None
                if citation.type != "char_location":
                    continue
                citations.append(
                    Citation(
                        citation.document_title or chunks[citation.document_index].id,
                        citation.cited_text,
                    )
                )

    return Answer("".join(body), citations)
