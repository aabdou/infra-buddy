"""Settings shared by every agent implementation.

These constants have to be identical across implementations, otherwise the
agents are not comparable: they would be answering from different corpora,
with different amounts of context, in different embedding spaces.
"""

import os
from pathlib import Path


def _find_data_dir() -> Path:
    """Locate the repo's data/ directory.

    Deliberately not cwd-relative: agents get run via `uv run --package ...`
    from wherever the user happens to be, so anchor on this file's location
    and walk up until we find the data/ directory the workspace root owns.
    """
    override = os.environ.get("INFRA_BUDDY_DATA")
    if override:
        return Path(override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate the data/ directory above "
        f"{Path(__file__).resolve()}. Set INFRA_BUDDY_DATA to point at it."
    )


DATA_DIR = _find_data_dir()
DB_PATH = DATA_DIR / "db/docs.db"

COLLECTION = "aws-s3-devguide"
TOP_N = 5

# The collection was ingested through Chroma's default embedding function,
# which is all-MiniLM-L6-v2 (384 dims) running on onnxruntime. Anything that
# queries this store has to embed with the same model. Using a different one
# fails silently: Chroma returns the nearest vectors in the wrong space, so
# you get confident, plausible, wrong chunks and no error anywhere.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
