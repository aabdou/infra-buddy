"""Ask a question, print the grounded answer and its citations."""

import argparse

from agent_pydantic.agent import answer


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over AWS docs, Pydantic AI")
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
