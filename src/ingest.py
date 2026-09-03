from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_path = Path("data")
with open(data_path / "aws/s3-devguide.txt", "r") as f:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(f.read())

    chromadb_client = chromadb.PersistentClient(path=data_path / "db/docs.db")
    collection = chromadb_client.get_or_create_collection("aws-s3-devguide")
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[f"aws-s3-dev-{i + x}" for x in range(len(batch))],
            documents=batch,
        )
