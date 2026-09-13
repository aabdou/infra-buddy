"""Chunk a plain-text doc and load it into the shared Chroma store.

Already run for the S3 Developer Guide; the collection is populated. Kept
generalized so the User Guide (or a second service) can be added later.
"""

import argparse
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from infra_core.config import DB_PATH

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chroma rejects oversized single calls, so adds are batched. Ids are built
# from the batch offset rather than the position within the batch -- indexing
# per-batch restarts at 0 every time and silently overwrites earlier chunks.
BATCH_SIZE = 5000


def ingest(txt_path: Path, collection_name: str, id_prefix: str) -> int:
    """Load one text file into `collection_name`. Returns the chunk count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(txt_path.read_text())

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(collection_name)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[f"{id_prefix}-{i + x}" for x in range(len(batch))],
            documents=batch,
        )

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("txt_path", type=Path, help="plain-text doc to ingest")
    parser.add_argument("collection", help="Chroma collection to add to")
    parser.add_argument("id_prefix", help="prefix for generated chunk ids")
    args = parser.parse_args()

    count = ingest(args.txt_path, args.collection, args.id_prefix)
    print(f"Ingested {count} chunks into {args.collection}")


if __name__ == "__main__":
    main()
