"""The eval dataset every agent is scored against.

Rows live in questions.jsonl next to this file, so they are versioned in git
(data/ is gitignored). Each row's `question` column is what Weave passes to
`answer(question)`; the other columns are there for scorers to compare against.

    uv run --package infra-core python -m infra_core.evals.dataset validate
    uv run --package infra-core python -m infra_core.evals.dataset publish
"""

import argparse
import json
import re
import unicodedata
from importlib.resources import files

import weave

from infra_core.config import DATA_DIR, init_trace, load_env
from infra_core.retrieval import search

DATASET_NAME = "s3-devguide-qa"
CORPUS = DATA_DIR / "aws/s3-devguide.txt"


def load_rows() -> list[dict]:
    text = files("infra_core.evals").joinpath("questions.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize(text: str) -> str:
    """Make excerpts comparable with PDF-extracted text.

    The corpus came out of a PDF: it has ligatures ("speciﬁes") and hard line
    wraps mid-sentence. NFKC turns the ligatures back into plain letters and
    collapsing whitespace makes a wrapped sentence match an unwrapped one.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def validate() -> list[str]:
    """Check the rows against the local corpus and vector store.

    Not a test: both live under data/, which isn't in git. Returns problems
    found; an empty list means every row is answerable from what's ingested.
    """
    rows = load_rows()
    problems = []

    ids = [row["id"] for row in rows]
    for dup in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate id: {dup}")

    corpus = normalize(CORPUS.read_text())
    for row in rows:
        excerpt = row["source_excerpt"]
        if excerpt is None:
            continue  # out-of-scope rows have nothing to find
        needle = normalize(excerpt)
        if needle not in corpus:
            problems.append(f"{row['id']}: excerpt not in corpus")
            continue
        # The excerpt existing isn't enough: if search() can't surface it for
        # the question as asked, a low score measures retrieval, not the agent.
        if not any(needle in normalize(c.text) for c in search(row["question"])):
            problems.append(f"{row['id']}: excerpt not in top-N search results")

    return problems


def publish() -> weave.Dataset:
    """Push the rows to Weave. Unchanged rows reuse the existing version."""
    dataset = weave.Dataset(name=DATASET_NAME, rows=load_rows())
    weave.publish(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "publish"])
    args = parser.parse_args()

    if args.command == "validate":
        problems = validate()
        for problem in problems:
            print(problem)
        print(f"{len(load_rows())} rows, {len(problems)} problems")
        raise SystemExit(1 if problems else 0)

    load_env()
    init_trace()
    publish()


if __name__ == "__main__":
    main()
