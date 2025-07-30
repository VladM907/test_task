"""
Enhanced ingestion pipeline that includes knowledge graph building.
Combines the existing vector pipeline with graph creation.
"""
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .pipeline import IngestionPipeline
from ..retrieval.chromadb_store import ChromaDBStore
from ..knowledge_graph.graph_builder import SimpleGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIngestionPipeline:
    def __init__(
        self,
        data_folder: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        chroma_store: Optional[ChromaDBStore] = None,
        graph_builder: Optional[SimpleGraphBuilder] = None
    ):
        """Initialize enhanced pipeline with vector and graph storage."""
        self.base_pipeline = IngestionPipeline(data_folder, chunk_size, chunk_overlap)
        self.chroma_store = chroma_store or ChromaDBStore()
        self.graph_builder = graph_builder or SimpleGraphBuilder()
        
        logger.info(f"Initialized enhanced pipeline for: {data_folder}")

    def run_complete_pipeline(
        self, 
        store_in_vector_db: bool = True,
        build_knowledge_graph: bool = True,
        clear_existing: bool = False
    ) -> Dict:
        """
        Run the complete enhanced ingestion pipeline.
        
        Args:
            store_in_vector_db: Whether to store chunks in ChromaDB
            build_knowledge_graph: Whether to build knowledge graph in Neo4j
            clear_existing: Whether to clear existing data before processing
            
        Returns:
            Dict with comprehensive statistics
        """
        try:
            logger.info("Starting enhanced ingestion pipeline")
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing data...")
                if store_in_vector_db:
                    self.chroma_store.clear_collection()
                if build_knowledge_graph:
                    self.graph_builder.clear_graph()
            
            # Step 1: Run base document ingestion and embedding
            logger.info("Step 1: Document ingestion and embedding generation")
            docs = self.base_pipeline.run()
            
            if not docs:
                logger.warning("No documents processed by base pipeline")
                return {"success": False, "error": "No documents processed"}
            
            base_stats = self.base_pipeline.get_stats(docs)
            logger.info(f"Base pipeline stats: {base_stats}")
            
            # Step 2: Store in vector database
            vector_stats = {}
            if store_in_vector_db:
                logger.info("Step 2: Storing chunks in ChromaDB")
                vector_stats = self._store_in_vector_db(docs)
            
            # Step 3: Build knowledge graph
            graph_stats = {}
            if build_knowledge_graph:
                logger.info("Step 3: Building knowledge graph")
                graph_stats = self.graph_builder.build_graph_from_documents(docs)
            
            # Compile final statistics
            final_stats = {
                "success": True,
                "base_ingestion": base_stats,
                "vector_storage": vector_stats,
                "knowledge_graph": graph_stats,
                "total_processing_time": "N/A"  # Could add timing if needed
            }
            
            logger.info("Enhanced pipeline completed successfully")
            logger.info(f"Final stats: {final_stats}")
            
            return final_stats
            
        except Exception as e:
            error_msg = f"Enhanced pipeline failed: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def _store_in_vector_db(self, docs: List[Dict]) -> Dict:
        """Store document chunks in ChromaDB."""
        try:
            # Prepare chunks for ChromaDB
            chunk_dicts = []
            for doc in docs:
                chunks = doc.get("chunks", [])
                embeddings = doc.get("embeddings", [])
                
                if len(chunks) != len(embeddings):
                    logger.warning(f"Chunk/embedding mismatch for {doc['path']}")
                    continue
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_dicts.append({
                        "path": doc["path"],
                        "chunk_id": i,
                        "content": chunk,
                        "embedding": embedding,
                        "metadata": doc.get("metadata", {})
                    })
            
            if not chunk_dicts:
                return {"chunks_stored": 0, "success": False, "error": "No valid chunks to store"}
            
            # Store in ChromaDB
            success = self.chroma_store.add_chunks(chunk_dicts)
            
            return {
                "chunks_prepared": len(chunk_dicts),
                "chunks_stored": len(chunk_dicts) if success else 0,
                "success": success
            }
            
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return {"chunks_stored": 0, "success": False, "error": str(e)}

    def get_comprehensive_stats(self) -> Dict:
        """Get statistics from all components."""
        try:
            chroma_info = self.chroma_store.get_collection_info()
            graph_stats = self.graph_builder.get_graph_statistics()
            
            return {
                "vector_database": chroma_info,
                "knowledge_graph": graph_stats,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {"status": "error", "error": str(e)}

    def test_retrieval(self, query: str = "test query") -> Dict:
        """Test both vector and graph retrieval capabilities."""
        try:
            from ..retrieval.hybrid_retriever import HybridRetriever
            
            # Test hybrid retrieval
            retriever = HybridRetriever(
                chroma_store=self.chroma_store,
                neo4j_client=self.graph_builder.client
            )
            
            results = retriever.search(query, n_results=3)
            
            return {
                "query": query,
                "results_found": len(results),
                "sample_result": results[0] if results else None,
                "retrieval_working": len(results) > 0
            }
            
        except Exception as e:
            logger.error(f"Retrieval test failed: {e}")
            return {"query": query, "retrieval_working": False, "error": str(e)}

    def close(self):
        """Close all connections."""
        if self.graph_builder:
            self.graph_builder.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
