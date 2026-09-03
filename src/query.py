from pathlib import Path

import chromadb

data_path = Path("data")
chromadb_client = chromadb.PersistentClient(path=data_path / "db/docs.db")
collection = chromadb_client.get_collection("aws-s3-devguide")
results = collection.query(query_texts=["blob"], n_results=2)
print(results)
