"""
Simple Neo4j client for knowledge graph operations.
Focuses on document relationships and basic entity connections.
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize Neo4j client with connection parameters."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                record = result.single()
                if record:
                    logger.info(f"Neo4j connection test: {record['message']}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        try:
            with self.driver.session() as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

    def create_indexes(self):
        """Create indexes for better performance."""
        indexes = [
            "CREATE INDEX document_path_index IF NOT EXISTS FOR (d:Document) ON (d.path)",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)", 
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE CONSTRAINT document_path_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.path IS UNIQUE"
        ]
        
        try:
            with self.driver.session() as session:
                for index_query in indexes:
                    try:
                        # Type ignore for neo4j query string compatibility
                        session.run(index_query)  # type: ignore
                        logger.info(f"Created index/constraint")
                    except Exception as e:
                        # Some indexes might already exist
                        logger.debug(f"Index creation note: {e}")
                        
                logger.info("Indexes and constraints created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

    def add_document(self, path: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document node to the graph."""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (d:Document {path: $path})
                SET d.content_length = $content_length,
                    d.file_type = $file_type,
                    d.created_at = datetime(),
                    d.title = $title
                RETURN d.path as path
                """
                
                # Extract metadata
                file_path = Path(path)
                params = {
                    "path": path,
                    "content_length": len(content),
                    "file_type": file_path.suffix.lower(),
                    "title": file_path.stem
                }
                
                # Add custom metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        if key not in params:  # Don't overwrite core fields
                            params[f"metadata_{key}"] = value
                
                result = session.run(query, params)
                record = result.single()
                
                if record:
                    logger.info(f"Added document: {record['path']}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to add document {path}: {e}")
            
        return False

    def add_chunk(self, document_path: str, chunk_id: int, content: str, chunk_index: Optional[int] = None) -> bool:
        """Add a chunk node and link it to its document."""
        try:
            with self.driver.session() as session:
                # Generate consistent chunk ID string
                if chunk_index is None:
                    chunk_index = chunk_id
                chunk_id_str = self._generate_chunk_id(document_path, chunk_index)
                
                query = """
                MATCH (d:Document {path: $document_path})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    content: $content,
                    content_length: $content_length,
                    chunk_index: $chunk_index
                })
                CREATE (d)-[:HAS_CHUNK]->(c)
                RETURN c.chunk_id as chunk_id
                """
                
                params = {
                    "document_path": document_path,
                    "chunk_id": chunk_id_str,
                    "content": content[:1000],  # Limit content size in graph
                    "content_length": len(content),
                    "chunk_index": chunk_index or chunk_id
                }
                
                result = session.run(query, params)  # type: ignore
                record = result.single()
                
                if record:
                    logger.debug(f"Added chunk: {record['chunk_id']}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id} for {document_path}: {e}")
            
        return False
    
    def _generate_chunk_id(self, document_path: str, chunk_index: int) -> str:
        """Generate consistent chunk ID."""
        doc_name = Path(document_path).stem
        return f"{doc_name}_chunk_{chunk_index}"

    def extract_and_add_entities(self, content: str, chunk_id: Optional[str] = None, document_path: Optional[str] = None) -> List[str]:
        """Extract simple entities (keywords) and add them to the graph."""
        try:
            # Simple keyword extraction (you can enhance this with NLP libraries)
            entities = self._extract_simple_keywords(content)
            
            if not entities:
                logger.debug(f"No entities extracted from content: {content[:100]}...")
                return []
            
            added_entities = []
            
            with self.driver.session() as session:
                for entity in entities[:10]:  # Limit to top 10 entities
                    try:
                        # Create entity node
                        entity_query = """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.frequency = COALESCE(e.frequency, 0) + 1
                        RETURN e.name as name
                        """
                        
                        entity_params = {
                            "name": entity.lower(),
                            "type": "keyword"
                        }
                        
                        result = session.run(entity_query, entity_params)  # type: ignore
                        entity_record = result.single()
                        
                        if not entity_record:
                            logger.warning(f"Failed to create entity: {entity}")
                            continue
                        
                        # Link entity to chunk if provided
                        if chunk_id:
                            # First verify the chunk exists
                            chunk_check_query = """
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            RETURN c.chunk_id
                            """
                            chunk_result = session.run(chunk_check_query, {"chunk_id": chunk_id})  # type: ignore
                            
                            if chunk_result.single():
                                link_query = """
                                MATCH (c:Chunk {chunk_id: $chunk_id})
                                MATCH (e:Entity {name: $entity_name})
                                MERGE (c)-[:MENTIONS]->(e)
                                RETURN c.chunk_id, e.name
                                """
                                link_result = session.run(link_query, {  # type: ignore
                                    "chunk_id": chunk_id,
                                    "entity_name": entity.lower()
                                })
                                
                                if link_result.single():
                                    logger.debug(f"Created MENTIONS relationship: {chunk_id} -> {entity}")
                                else:
                                    logger.warning(f"Failed to create MENTIONS relationship: {chunk_id} -> {entity}")
                            else:
                                logger.warning(f"Chunk {chunk_id} not found, cannot create MENTIONS relationship")
                        
                        # Link entity to document if provided
                        if document_path:
                            # First verify the document exists
                            doc_check_query = """
                            MATCH (d:Document {path: $document_path})
                            RETURN d.path
                            """
                            doc_result = session.run(doc_check_query, {"document_path": document_path})  # type: ignore
                            
                            if doc_result.single():
                                doc_link_query = """
                                MATCH (d:Document {path: $document_path})
                                MATCH (e:Entity {name: $entity_name})
                                MERGE (d)-[:CONTAINS]->(e)
                                RETURN d.path, e.name
                                """
                                doc_link_result = session.run(doc_link_query, {  # type: ignore
                                    "document_path": document_path,
                                    "entity_name": entity.lower()
                                })
                                
                                if doc_link_result.single():
                                    logger.debug(f"Created CONTAINS relationship: {document_path} -> {entity}")
                                else:
                                    logger.warning(f"Failed to create CONTAINS relationship: {document_path} -> {entity}")
                            else:
                                logger.warning(f"Document {document_path} not found, cannot create CONTAINS relationship")
                        
                        added_entities.append(entity)
                        
                    except Exception as e:
                        logger.debug(f"Failed to add entity {entity}: {e}")
                        continue
            
            logger.info(f"Added {len(added_entities)} entities: {added_entities[:5]}")
            return added_entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []

    def _extract_simple_keywords(self, content: str) -> List[str]:
        """Simple keyword extraction from content with improved logic."""
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'said', 'say', 'get', 'go', 'know', 'think', 'take', 'see', 'come', 'its', 'also',
            'back', 'use', 'two', 'way', 'even', 'new', 'want', 'because', 'any', 'day', 'most', 'us'
        }
        
        # Extract different types of potential keywords
        word_freq = {}
        
        # 1. Capitalized words (proper nouns, important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        for word in capitalized:
            if word.lower() not in stop_words and len(word) > 2:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 2  # Weight capitalized words more
        
        # 2. All words (regular keywords)
        all_words = re.findall(r'\b[A-Za-z]{3,}\b', content.lower())
        for word in all_words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 3. Multi-word phrases (simple approach - two consecutive non-stop words)
        words = re.findall(r'\b[A-Za-z]+\b', content.lower())
        for i in range(len(words) - 1):
            if (words[i] not in stop_words and words[i+1] not in stop_words and 
                len(words[i]) > 2 and len(words[i+1]) > 2):
                phrase = f"{words[i]} {words[i+1]}"
                word_freq[phrase] = word_freq.get(phrase, 0) + 1
        
        # Filter and sort by frequency, but also consider length and variety
        filtered_words = {}
        for word, freq in word_freq.items():
            # Boost score for longer words and phrases
            score = freq
            if len(word) > 6:
                score += 1
            if ' ' in word:  # Multi-word phrase
                score += 1
            
            # Only keep words that appear at least twice or are long/phrases
            if freq > 1 or len(word) > 6 or ' ' in word:
                filtered_words[word] = score
        
        # Return top keywords sorted by score
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        result = [word for word, score in sorted_words[:20]]
        
        logger.debug(f"Extracted keywords from {len(content)} chars: {result[:10]}")
        return result

    def find_related_documents(self, document_path: str, limit: int = 5) -> List[Dict]:
        """Find documents related through shared entities."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d1:Document {path: $document_path})-[:CONTAINS]->(e:Entity)<-[:CONTAINS]-(d2:Document)
                WHERE d1 <> d2
                WITH d2, COUNT(e) as shared_entities
                ORDER BY shared_entities DESC
                LIMIT $limit
                RETURN d2.path as path, d2.title as title, shared_entities
                """
                
                result = session.run(query, {"document_path": document_path, "limit": limit})
                
                related = []
                for record in result:
                    related.append({
                        "path": record["path"],
                        "title": record["title"],
                        "shared_entities": record["shared_entities"]
                    })
                
                return related
                
        except Exception as e:
            logger.error(f"Failed to find related documents: {e}")
            return []

    def get_document_entities(self, document_path: str) -> List[Dict]:
        """Get all entities mentioned in a document."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Document {path: $document_path})-[:CONTAINS]->(e:Entity)
                RETURN e.name as name, e.type as type, e.frequency as frequency
                ORDER BY e.frequency DESC
                """
                
                result = session.run(query, {"document_path": document_path})
                
                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "frequency": record["frequency"]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get document entities: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get basic statistics about the knowledge graph."""
        try:
            with self.driver.session() as session:
                stats_query = """
                MATCH (d:Document) WITH COUNT(d) as docs
                MATCH (c:Chunk) WITH docs, COUNT(c) as chunks
                MATCH (e:Entity) WITH docs, chunks, COUNT(e) as entities
                MATCH ()-[r]->() WITH docs, chunks, entities, COUNT(r) as relationships
                RETURN docs, chunks, entities, relationships
                """
                
                result = session.run(stats_query)
                record = result.single()
                
                if record:
                    return {
                        "documents": record["docs"],
                        "chunks": record["chunks"],
                        "entities": record["entities"],
                        "relationships": record["relationships"]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            
        return {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
