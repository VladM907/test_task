
"""
Script to test ChromaDBStore integration with the ingestion pipeline.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.chromadb_store import ChromaDBStore


if __name__ == "__main__":
    # Ingest documents and generate embeddings
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(project_root, 'data')
    pipeline = IngestionPipeline(data_dir)
    docs = pipeline.run()  # Each doc: path, full_content, chunks, embeddings

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

    # Test retrieval with the first chunk's embedding
    if chunk_dicts:
        query_embedding = chunk_dicts[0]["embedding"]
        results = store.query(query_embedding, n_results=3)
        print("Top 3 similar chunks:")
        for r in results:
            print(r["metadata"], "|", r["distance"])
            print(r["content"][:100], "...\n---")
    else:
        print("No chunks to query.")
