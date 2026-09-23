"""Ask a question, print the grounded answer and its citations."""

import argparse

from infra_core.config import load_env

from agent_raw_tools.agent import answer


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        description="RAG over AWS docs, Anthropic Messages API with a search tool"
    )
    parser.add_argument("question", help="natural-language question about S3")
    args = parser.parse_args()

    result = answer(args.question)
    print(result.text)

    if result.citations:
        print("\nSources:")
        for citation in result.citations:
            print(f"  [{citation.chunk_id}] {citation.snippet}")


if __name__ == "__main__":
    main()
