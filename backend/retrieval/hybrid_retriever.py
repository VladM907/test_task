"""
Hybrid retriever that combines ChromaDB vector search with Neo4j graph traversal.
Provides richer context by combining semantic similarity with knowledge graph relationships.
"""
from typing import List, Dict, Optional, Any
import logging
from ..retrieval.chromadb_store import ChromaDBStore
from ..ingestion.embedding import EmbeddingGenerator
from ..knowledge_graph.neo4j_client import Neo4jClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(
        self, 
        chroma_store: Optional[ChromaDBStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        embedder: Optional[EmbeddingGenerator] = None
    ):
        """Initialize hybrid retriever with vector and graph stores."""
        self.chroma_store = chroma_store or ChromaDBStore()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.embedder = embedder or EmbeddingGenerator()
        
        logger.info("Initialized hybrid retriever")

    def search(
        self, 
        query: str, 
        n_results: int = 5,
        use_graph_expansion: bool = True,
        graph_expansion_limit: int = 3
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and graph traversal.
        
        Args:
            query: Search query string
            n_results: Number of results to return from vector search
            use_graph_expansion: Whether to expand results using knowledge graph
            graph_expansion_limit: Number of related items to add from graph
            
        Returns:
            List of search results with combined vector and graph context
        """
        try:
            logger.info(f"Performing hybrid search for: '{query}'")
            
            # Step 1: Vector similarity search
            query_embedding = self.embedder.embed_single(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            vector_results = self.chroma_store.query(query_embedding, n_results)
            
            if not vector_results:
                logger.warning("No vector search results found")
                return []
            
            logger.info(f"Found {len(vector_results)} vector search results")
            
            # Step 2: Enhance with graph context if enabled
            enhanced_results = []
            
            for result in vector_results:
                enhanced_result = {
                    **result,
                    "source": "vector_search",
                    "graph_context": []
                }
                
                if use_graph_expansion:
                    try:
                        # Get document path from metadata
                        doc_path = result.get("metadata", {}).get("path")
                        chunk_index = result.get("metadata", {}).get("chunk_id", 0)
                        
                        if doc_path:
                            # Find related documents through graph
                            related_docs = self.neo4j_client.find_related_documents(
                                doc_path, 
                                limit=graph_expansion_limit
                            )
                            
                            # Get chunk-specific entities via MENTIONS relationships
                            chunk_id_str = self.neo4j_client._generate_chunk_id(doc_path, chunk_index)
                            chunk_entities = self._get_chunk_entities(chunk_id_str)
                            
                            enhanced_result["graph_context"] = {
                                "related_documents": related_docs,
                                "entities": chunk_entities[:5]  # Top 5 entities from this specific chunk
                            }
                            
                            logger.debug(f"Added graph context: {len(related_docs)} related docs, {len(chunk_entities)} chunk entities")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get graph context for result: {e}")
                
                enhanced_results.append(enhanced_result)
            
            # Step 3: Optionally add graph-only results for better coverage
            if use_graph_expansion and len(enhanced_results) < n_results:
                graph_results = self._get_graph_based_results(query, n_results - len(enhanced_results))
                enhanced_results.extend(graph_results)
            
            logger.info(f"Returning {len(enhanced_results)} enhanced results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _get_graph_based_results(self, query: str, limit: int) -> List[Dict]:
        """
        Get additional results based purely on graph traversal.
        Looks for entities mentioned in the query and finds related content.
        """
        try:
            # Extract potential entities from query (simple keyword approach)
            query_words = [word.lower().strip() for word in query.split() if len(word) > 2]
            
            graph_results = []
            
            for word in query_words[:3]:  # Check top 3 words as potential entities
                try:
                    connections = self._find_entity_connections_in_graph(word, limit=2)
                    
                    for connection in connections:
                        # Try to get the chunk content from ChromaDB for consistency
                        try:
                            # This is a simplified approach - in practice you might want
                            # to store chunk embeddings in the graph or have better linking
                            graph_results.append({
                                "content": connection["chunk_content"],
                                "metadata": {
                                    "path": connection["document_path"],
                                    "source": "graph_traversal",
                                    "entity_match": word
                                },
                                "distance": 0.5,  # Arbitrary score for graph results
                                "similarity": 0.5,
                                "source": "graph_search",
                                "graph_context": []
                            })
                            
                        except Exception as e:
                            logger.debug(f"Could not enhance graph result: {e}")
                            continue
                
                except Exception as e:
                    logger.debug(f"No graph connections found for '{word}': {e}")
                    continue
            
            return graph_results[:limit]
            
        except Exception as e:
            logger.error(f"Graph-based search failed: {e}")
            return []

    def _get_chunk_entities(self, chunk_id: str) -> List[Dict]:
        """Get entities mentioned in a specific chunk via MENTIONS relationships."""
        try:
            with self.neo4j_client.driver.session() as session:
                query = """
                MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
                RETURN e.name as name, e.type as type, e.frequency as frequency
                ORDER BY e.frequency DESC
                """
                
                result = session.run(query, {"chunk_id": chunk_id})
                
                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "frequency": record["frequency"]
                    })
                
                return entities
                
        except Exception as e:
            logger.debug(f"Failed to get chunk entities for {chunk_id}: {e}")
            return []

    def _find_entity_connections_in_graph(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """Find connections for an entity in the graph."""
        try:
            with self.neo4j_client.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN d.path as document_path, d.title as document_title, 
                       c.chunk_id as chunk_id, c.content as chunk_content
                LIMIT $limit
                """
                
                result = session.run(query, {"entity_name": entity_name, "limit": limit})
                
                connections = []
                for record in result:
                    connections.append({
                        "document_path": record["document_path"],
                        "document_title": record["document_title"],
                        "chunk_id": record["chunk_id"],
                        "chunk_content": record["chunk_content"]
                    })
                
                return connections
                
        except Exception as e:
            logger.debug(f"Entity connection search failed for {entity_name}: {e}")
            return []

    def get_document_neighborhood(self, document_path: str) -> Dict:
        """
        Get the complete neighborhood of a document including:
        - Related documents from graph
        - Document entities
        - Similar chunks from vector search
        """
        try:
            # Get graph-based relationships
            related_docs = self.neo4j_client.find_related_documents(document_path)
            entities = self.neo4j_client.get_document_entities(document_path)
            
            # Get similar content from vector search
            # (This would require storing document-level embeddings or aggregating chunk similarities)
            vector_neighbors = []  # Placeholder for now
            
            return {
                "document_path": document_path,
                "graph_related": related_docs,
                "entities": entities,
                "vector_neighbors": vector_neighbors
            }
            
        except Exception as e:
            logger.error(f"Failed to get document neighborhood: {e}")
            return {"document_path": document_path, "graph_related": [], "entities": [], "vector_neighbors": []}

    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """Search for content related to a specific entity."""
        try:
            connections = self._find_entity_connections_in_graph(entity_name, limit)
            
            results = []
            for connection in connections:
                results.append({
                    "content": connection["chunk_content"],
                    "metadata": {
                        "path": connection["document_path"],
                        "chunk_id": connection["chunk_id"],
                        "entity": entity_name
                    },
                    "source": "entity_search",
                    "distance": 0.0,  # Exact entity match
                    "similarity": 1.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Entity search failed for {entity_name}: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get statistics from both vector and graph stores."""
        try:
            chroma_info = self.chroma_store.get_collection_info()
            graph_stats = self.neo4j_client.get_statistics()
            
            return {
                "vector_store": chroma_info,
                "knowledge_graph": graph_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close connections to both stores."""
        if self.neo4j_client:
            self.neo4j_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
