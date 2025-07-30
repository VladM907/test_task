"""
ChromaDB vector store integration for storing and retrieving document chunks and embeddings.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBStore:
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "documents"):
        """Initialize ChromaDB store with improved configuration."""
        self.persist_directory = persist_directory or os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self.collection_name = collection_name
        
        try:
            # Create client with persistent storage
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks and embeddings for RAG system"}
            )
            
            logger.info(f"Connected to ChromaDB collection '{self.collection_name}' at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_chunks(self, docs: List[Dict]) -> bool:
        """
        Adds document chunks and their embeddings to ChromaDB with improved error handling.
        Each doc should have: path, chunk_id, content, embedding
        Returns True if successful, False otherwise.
        """
        if not docs:
            logger.warning("No documents to add")
            return False
            
        try:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                # Validate required fields
                required_fields = ['path', 'chunk_id', 'content', 'embedding']
                if not all(field in doc for field in required_fields):
                    logger.warning(f"Skipping document with missing fields: {doc.keys()}")
                    continue
                
                # Generate unique ID
                chunk_id = f"{os.path.basename(doc['path'])}::chunk_{doc['chunk_id']}::{uuid.uuid4().hex[:8]}"
                
                # Validate embedding
                if not isinstance(doc['embedding'], list) or len(doc['embedding']) == 0:
                    logger.warning(f"Invalid embedding for chunk {chunk_id}")
                    continue
                
                ids.append(chunk_id)
                documents.append(str(doc["content"]))
                embeddings.append(doc["embedding"])
                metadatas.append({
                    "path": doc["path"],
                    "chunk_id": doc["chunk_id"],
                    "content_length": len(str(doc["content"])),
                    **doc.get("metadata", {})  # Include any additional metadata
                })
            
            if not ids:
                logger.warning("No valid documents to add after validation")
                return False
            
            # Add to collection in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                
                logger.info(f"Added batch {i//batch_size + 1}: {batch_end - i} chunks")
            
            logger.info(f"Successfully added {len(ids)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False

    def query(self, query_embedding: list, n_results: int = 5) -> List[Dict]:
        """
        Query ChromaDB for the most similar chunks to the query embedding.
        Returns a list of dicts with content and metadata.
        """
        try:
            if not isinstance(query_embedding, list) or len(query_embedding) == 0:
                logger.error("Invalid query embedding provided")
                return []
            
            # Ensure we don't request more results than exist
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Collection is empty")
                return []
            
            actual_n_results = min(n_results, collection_count)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n_results
            )
            
            # Validate results structure
            if not results:
                logger.warning("No results returned from ChromaDB query")
                return []
            
            # Check if required keys exist and are not None
            required_keys = ["documents", "metadatas", "distances"]
            for key in required_keys:
                if key not in results or results[key] is None:
                    logger.error(f"Missing or None key in results: {key}")
                    return []
            
            # Check if results have proper structure (list of lists)
            if (not isinstance(results["documents"], list) or 
                not isinstance(results["metadatas"], list) or 
                not isinstance(results["distances"], list) or
                len(results["documents"]) == 0):
                logger.error("Invalid results structure from ChromaDB")
                return []
            
            # Format results safely
            formatted_results = []
            docs = results["documents"][0]
            metas = results["metadatas"][0] 
            dists = results["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                formatted_results.append({
                    "content": doc,
                    "metadata": meta or {},  # Ensure metadata is not None
                    "distance": dist,
                    "similarity": 1 - dist if dist is not None else 0  # Convert distance to similarity
                })
            
            logger.info(f"Successfully retrieved {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Get all IDs and delete them
            results = self.collection.get()
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
                logger.info(f"Cleared {len(results['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
