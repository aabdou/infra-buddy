"""Retrieval as a tool, with the agent loop written by hand.

Approach: describe search() to the model as a tool and let it decide whether
to search, what to search for, and how often. Compared with agent-raw -- which
always searches once, using the user's literal question -- the model can
rephrase the query, search several times, or skip searching entirely.

The loop this file owns is what Pydantic AI and LangChain automate:

    call create() -> stop_reason "tool_use"? -> run the tool, append the
    result, call create() again -> repeat until "end_turn"

MAX_ROUNDS caps that loop; without it, a model that keeps asking for searches
would spend money indefinitely.
"""

import anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock
from infra_core.blocks import to_question, to_user_turn
from infra_core.prompts import SYSTEM_PROMPT, format_context
from infra_core.retrieval import search
from infra_core.types import Answer, Citation
from pydantic import BaseModel


class CitationOut(BaseModel):
    chunk_id: str
    snippet: str


class AnswerOut(BaseModel):
    text: str
    citations: list[CitationOut]


MAX_ROUNDS = 5
MODEL = "claude-opus-5"

# The shared prompt says nothing about citing: agent-raw gets citations from
# the API, which tracks them itself. Here the model reports them, so it has to
# be told to -- the schema guarantees the shape of `citations`, not that the
# ids in it are real.
CITATION_RULES = """

- Populate `citations` with the id of each chunk you relied on and the exact \
sentence from it that supports the claim.
- Use only chunk ids returned by search_docs. Never invent one."""

SEARCH_DOCS: ToolParam = {
    "name": "search_docs",
    "description": (
        "Search the Amazon S3 documentation and return the most relevant "
        "excerpts. Use a short topical query describing what you need to "
        "know, not the user's question verbatim. Call this more than once if "
        "the question has several parts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up, e.g. 'enable bucket versioning'",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _run_search(block: ToolUseBlock) -> ToolResultBlockParam:
    """Execute one search_docs call and package the chunks as a tool result.

    tool_use_id ties the result to the request: one result per tool_use block,
    all of them in a single user message.
    """
    args = block.input if isinstance(block.input, dict) else {}
    chunks = search(str(args.get("query", "")))

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        # format_context prefixes each chunk with its id, which is what the
        # model cites from.
        "content": format_context(chunks),
    }


def answer(question: str) -> Answer:
    messages: list[MessageParam] = [to_user_turn([to_question(question)])]

    for _ in range(MAX_ROUNDS):
        result = _get_client().messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT + CITATION_RULES,
            tools=[SEARCH_DOCS],
            messages=messages,
            output_format=AnswerOut,
        )
        # Every turn goes back verbatim: without the assistant's tool request
        # in the history, the next call has no idea a search was asked for.
        messages.append({"role": "assistant", "content": result.content})

        if result.stop_reason != "tool_use":
            break

        messages.append(
            to_user_turn(
                [_run_search(b) for b in result.content if b.type == "tool_use"]
            )
        )
    else:
        raise RuntimeError(f"Still asking for searches after {MAX_ROUNDS} rounds")

    if result.stop_reason != "end_turn":
        raise RuntimeError(f"Incomplete answer: stop_reason={result.stop_reason}")

    answer_out = result.parsed_output
    if answer_out is None:
        raise RuntimeError("No structured output in the final response")

    return Answer(
        answer_out.text,
        [Citation(c.chunk_id, c.snippet) for c in answer_out.citations],
    )
