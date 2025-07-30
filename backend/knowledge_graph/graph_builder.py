"""
Simple graph builder that integrates with the existing ingestion pipeline.
Creates document and chunk nodes with basic entity relationships.
"""
from typing import List, Dict, Optional
import logging
from .neo4j_client import Neo4jClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGraphBuilder:
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize graph builder with Neo4j client."""
        self.client = neo4j_client or Neo4jClient()
        
        # Initialize indexes for better performance
        try:
            self.client.create_indexes()
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def build_graph_from_documents(self, docs: List[Dict]) -> Dict:
        """
        Build knowledge graph from ingested documents.
        
        Args:
            docs: List of document dicts with path, full_content, chunks, metadata
            
        Returns:
            Dict with statistics about the graph building process
        """
        stats = {
            "documents_processed": 0,
            "chunks_added": 0,
            "entities_extracted": 0,
            "errors": []
        }
        
        if not docs:
            logger.warning("No documents to process")
            return stats
        
        logger.info(f"Building graph from {len(docs)} documents")
        
        for doc in docs:
            try:
                # Add document node
                success = self.client.add_document(
                    path=doc["path"],
                    content=doc["full_content"],
                    metadata=doc.get("metadata", {})
                )
                
                if not success:
                    stats["errors"].append(f"Failed to add document: {doc['path']}")
                    continue
                
                stats["documents_processed"] += 1
                
                # Add chunk nodes and extract entities
                chunks = doc.get("chunks", [])
                for i, chunk in enumerate(chunks):
                    try:
                        # Add chunk node
                        chunk_success = self.client.add_chunk(
                            document_path=doc["path"],
                            chunk_id=i,
                            content=chunk,
                            chunk_index=i
                        )
                        
                        if chunk_success:
                            stats["chunks_added"] += 1
                            
                            # Extract entities from chunk - use same chunk ID generation logic
                            chunk_id_str = self.client._generate_chunk_id(doc["path"], i)
                            logger.debug(f"Extracting entities for chunk: {chunk_id_str}")
                            
                            entities = self.client.extract_and_add_entities(
                                content=chunk,
                                chunk_id=chunk_id_str,  # Pass the generated chunk ID
                                document_path=doc["path"]
                            )
                            
                            stats["entities_extracted"] += len(entities)
                            
                        else:
                            stats["errors"].append(f"Failed to add chunk {i} for {doc['path']}")
                            
                    except Exception as e:
                        error_msg = f"Error processing chunk {i} in {doc['path']}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
                        continue
                
                # Extract document-level entities
                try:
                    doc_entities = self.client.extract_and_add_entities(
                        content=doc["full_content"][:2000],  # Limit for performance
                        document_path=doc["path"]
                    )
                    stats["entities_extracted"] += len(doc_entities)
                    
                except Exception as e:
                    error_msg = f"Error extracting document entities for {doc['path']}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                
            except Exception as e:
                error_msg = f"Error processing document {doc['path']}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                continue
        
        logger.info(f"Graph building completed: {stats}")
        return stats
    
    def get_document_context(self, document_path: str) -> Dict:
        """
        Get additional context for a document from the knowledge graph.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict with related documents and entities
        """
        try:
            # Get document entities
            entities = self.client.get_document_entities(document_path)
            
            # Get related documents
            related_docs = self.client.find_related_documents(document_path)
            
            return {
                "entities": entities,
                "related_documents": related_docs,
                "document_path": document_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get document context for {document_path}: {e}")
            return {"entities": [], "related_documents": [], "document_path": document_path}
    
    def find_entity_connections(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """
        Find documents and chunks that mention a specific entity.
        
        Args:
            entity_name: Name of the entity to search for
            limit: Maximum number of results to return
            
        Returns:
            List of documents/chunks mentioning the entity
        """
        try:
            with self.client.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN d.path as document_path, d.title as document_title, 
                       c.chunk_id as chunk_id, c.content as chunk_content
                LIMIT $limit
                """
                
                result = session.run(query, {"entity_name": entity_name.lower(), "limit": limit})
                
                connections = []
                for record in result:
                    connections.append({
                        "document_path": record["document_path"],
                        "document_title": record["document_title"],
                        "chunk_id": record["chunk_id"],
                        "chunk_content": record["chunk_content"][:200] + "..." if len(record["chunk_content"]) > 200 else record["chunk_content"]
                    })
                
                return connections
                
        except Exception as e:
            logger.error(f"Failed to find entity connections for {entity_name}: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        try:
            stats = self.client.get_statistics()
            return stats
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    def clear_graph(self) -> bool:
        """Clear the entire knowledge graph."""
        try:
            self.client.clear_database()
            return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False

    def close(self):
        """Close the Neo4j connection."""
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
