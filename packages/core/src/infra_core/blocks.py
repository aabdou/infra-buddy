"""Builders for Anthropic content blocks, shared by the SDK-based agents.

Deliberately the one Anthropic-specific module in infra-core: the agents that
talk to the Messages API directly all assemble the same message shapes, and
duplicating them per package is how the two copies drift apart. Agents built
on a framework (Pydantic AI, LangChain) do not import this.
"""

from collections.abc import Sequence

from anthropic.types import ContentBlockParam, MessageParam, TextBlockParam


def to_question(text: str) -> TextBlockParam:
    """A plain text block -- the user's question."""
    return {"type": "text", "text": text}


def to_user_turn(content: Sequence[ContentBlockParam]) -> MessageParam:
    """Wrap content blocks as one user turn in the messages list.

    Sequence, not list: a list[DocumentBlockParam | TextBlockParam] is not
    assignable to a list[ContentBlockParam] parameter, because list is
    invariant. Sequence is covariant, so the caller's mixed list fits.
    """
    return {"role": "user", "content": content}
