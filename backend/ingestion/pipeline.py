"""
Pipeline for loading documents, generating embeddings, and preparing for storage.
"""
from pathlib import Path
from .loader import DocumentLoader
from .embedding import EmbeddingGenerator
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, data_folder: str, chunk_size: int = 800, chunk_overlap: int = 100):
        """Initialize ingestion pipeline with configurable parameters."""
        self.data_folder = Path(data_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.embedder = EmbeddingGenerator()
            logger.info(f"Initialized ingestion pipeline for folder: {self.data_folder}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            raise

    def run(self) -> List[Dict]:
        """
        Run the complete ingestion pipeline.
        Returns a list of dicts, each with:
            - path: file path
            - full_content: the full document text (for knowledge graph)
            - chunks: list of split text chunks
            - embeddings: list of embeddings for each chunk
            - metadata: document metadata
        """
        try:
            logger.info("Starting document ingestion pipeline")
            
            # Load and split documents
            docs = DocumentLoader.load_documents(
                self.data_folder, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            if not docs:
                logger.warning("No documents were loaded")
                return []
            
            logger.info(f"Loaded {len(docs)} documents, generating embeddings...")
            
            # Generate embeddings for each document's chunks
            for i, doc in enumerate(docs):
                try:
                    logger.info(f"Processing document {i+1}/{len(docs)}: {doc['path']}")
                    
                    if not doc["chunks"]:
                        logger.warning(f"No chunks found for {doc['path']}")
                        doc["embeddings"] = []
                        continue
                    
                    # Generate embeddings for all chunks
                    embeddings = self.embedder.embed(doc["chunks"])
                    doc["embeddings"] = embeddings
                    
                    # Validate embeddings
                    if len(embeddings) != len(doc["chunks"]):
                        logger.error(f"Mismatch: {len(embeddings)} embeddings for {len(doc['chunks'])} chunks in {doc['path']}")
                        continue
                    
                    logger.info(f"Generated {len(embeddings)} embeddings for {doc['path']}")
                    
                except Exception as e:
                    logger.error(f"Failed to process embeddings for {doc['path']}: {e}")
                    doc["embeddings"] = []
                    continue
            
            # Filter out documents with no valid embeddings
            valid_docs = [doc for doc in docs if doc.get("embeddings")]
            
            logger.info(f"Pipeline completed: {len(valid_docs)}/{len(docs)} documents processed successfully")
            return valid_docs
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return []
    
    def get_stats(self, docs: List[Dict]) -> Dict:
        """Get statistics about processed documents."""
        if not docs:
            return {}
        
        total_chunks = sum(len(doc.get("chunks", [])) for doc in docs)
        total_embeddings = sum(len(doc.get("embeddings", [])) for doc in docs)
        file_types = {}
        
        for doc in docs:
            file_ext = Path(doc["path"]).suffix.lower()
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "file_types": file_types,
            "avg_chunks_per_doc": total_chunks / len(docs) if docs else 0
        }
