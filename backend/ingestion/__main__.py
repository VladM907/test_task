"""
Script to run the ingestion pipeline and print results (for testing).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.chromadb_store import ChromaDBStore

if __name__ == "__main__":
    # Resolve data directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(project_root, 'data')
    pipeline = IngestionPipeline(data_dir)
    docs = pipeline.run()

    # Prepare chunk dicts for ChromaDB
    chunk_dicts = []
    for doc in docs:
        for i, (chunk, embedding) in enumerate(zip(doc["chunks"], doc["embeddings"])):
            chunk_dicts.append({
                "path": doc["path"],
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding
            })

    # Store in ChromaDB
    store = ChromaDBStore()
    store.add_chunks(chunk_dicts)
    print(f"Added {len(chunk_dicts)} chunks to ChromaDB.")
