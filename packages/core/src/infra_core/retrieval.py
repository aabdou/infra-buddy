"""The shared retrieval layer.

Every agent goes through `search()` so that a difference in answer quality
between two implementations is attributable to the framework, not to two
subtly different copies of this query code.
"""

import chromadb

from infra_core.config import COLLECTION, DB_PATH, TOP_N
from infra_core.types import Chunk

_collection = None


def _get_collection():
    """Open the store once; PersistentClient startup is not cheap."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = client.get_collection(COLLECTION)
    return _collection


def search(question: str, n: int = TOP_N) -> list[Chunk]:
    """Return the n passages most relevant to `question`."""
    results = _get_collection().query(query_texts=[question], n_results=n)

    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    return [
        Chunk(id=chunk_id, text=text, distance=distance)
        for chunk_id, text, distance in zip(ids, documents, distances)
    ]
