#!/usr/bin/env python3
"""
Interactive retrieval testing script.
Test both vector store and knowledge graph retrieval with custom queries.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.retrieval.chromadb_store import ChromaDBStore
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.knowledge_graph.graph_builder import SimpleGraphBuilder
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.ingestion.embedding import EmbeddingGenerator

def print_separator(title, char="=", width=60):
    """Print a formatted separator with title."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title, char="-", width=50):
    """Print a subsection header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def test_vector_store_only(query: str, chroma_store: ChromaDBStore, embedder: EmbeddingGenerator, n_results: int = 5):
    """Test vector store retrieval only."""
    print_subsection("🔍 VECTOR STORE RESULTS")
    
    try:
        # Generate query embedding
        print(f"Query: '{query}'")
        query_embedding = embedder.embed_single(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search vector store
        results = chroma_store.query(query_embedding, n_results)
        
        if not results:
            print("❌ No results found in vector store")
            return []
        
        print(f"✅ Found {len(results)} results from vector store:")
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            print(f"   Similarity: {result.get('similarity', 0):.4f}")
            print(f"   Distance: {result.get('distance', 0):.4f}")
            
            metadata = result.get('metadata', {})
            print(f"   Source: {metadata.get('path', 'unknown')}")
            print(f"   Chunk ID: {metadata.get('chunk_id', 'unknown')}")
            
            content = result.get('content', '')
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   Content: {preview}")
        
        return results
        
    except Exception as e:
        print(f"❌ Vector store search failed: {e}")
        return []

def test_knowledge_graph_only(query: str, graph_builder: SimpleGraphBuilder):
    """Test knowledge graph retrieval only."""
    print_subsection("🕸️  KNOWLEDGE GRAPH RESULTS")
    
    try:
        print(f"Query: '{query}'")
        
        # Extract potential entities from query
        query_words = [word.lower().strip() for word in query.split() if len(word) > 2]
        
        all_graph_results = []
        
        # Test entity-based search
        print("\n🏷️  Entity-based search:")
        for word in query_words[:3]:  # Test top 3 words as potential entities
            try:
                connections = graph_builder.find_entity_connections(word, limit=3)
                
                if connections:
                    print(f"   Entity '{word}': {len(connections)} connections found")
                    for conn in connections[:2]:  # Show top 2
                        print(f"     - {conn['document_path']}")
                        content_preview = conn['chunk_content'][:100] + "..." if len(conn['chunk_content']) > 100 else conn['chunk_content']
                        print(f"       {content_preview}")
                    
                    all_graph_results.extend(connections)
                else:
                    print(f"   Entity '{word}': No connections found")
                    
            except Exception as e:
                print(f"   Entity '{word}': Error - {e}")
        
        # Test direct entity search using the client method
        print("\n🔍 Direct entity search:")
        for word in query_words[:2]:
            try:
                with graph_builder.client.driver.session() as session:
                    # Simple entity search
                    entity_query = """
                    MATCH (e:Entity {name: $entity_name})
                    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(e)
                    RETURN e.name, e.frequency, COUNT(d) as doc_count
                    """
                    
                    result = session.run(entity_query, {"entity_name": word.lower()})  # type: ignore
                    record = result.single()
                    
                    if record:
                        print(f"   Entity '{word}': frequency={record['e.frequency']}, in {record['doc_count']} documents")
                    else:
                        print(f"   Entity '{word}': Not found in graph")
                        
            except Exception as e:
                print(f"   Entity '{word}': Error - {e}")
        
        # Get document relationships if any results found
        if all_graph_results:
            print(f"\n📊 Summary: Found {len(all_graph_results)} total graph connections")
        else:
            print("\n📊 Summary: No graph connections found")
        
        return all_graph_results
        
    except Exception as e:
        print(f"❌ Knowledge graph search failed: {e}")
        return []

def test_hybrid_retrieval(query: str, hybrid_retriever: HybridRetriever, n_results: int = 5):
    """Test hybrid retrieval combining both stores."""
    print_subsection("🔄 HYBRID RETRIEVAL RESULTS")
    
    try:
        print(f"Query: '{query}'")
        
        # Perform hybrid search
        results = hybrid_retriever.search(
            query=query,
            n_results=n_results,
            use_graph_expansion=True,
            graph_expansion_limit=3
        )
        
        if not results:
            print("❌ No results found from hybrid search")
            return []
        
        print(f"✅ Found {len(results)} results from hybrid search:")
        
        for i, result in enumerate(results, 1):
            print(f"\n🎯 Result {i}:")
            print(f"   Source: {result.get('source', 'unknown')}")
            print(f"   Similarity: {result.get('similarity', 0):.4f}")
            
            metadata = result.get('metadata', {})
            print(f"   Document: {metadata.get('path', 'unknown')}")
            
            content = result.get('content', '')
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   Content: {preview}")
            
            # Show graph context if available
            graph_context = result.get('graph_context', {})
            if graph_context:
                related_docs = graph_context.get('related_documents', [])
                entities = graph_context.get('entities', [])
                
                if entities:
                    entity_names = [e.get('name', '') for e in entities[:3]]
                    print(f"   Related entities: {', '.join(entity_names)}")
                
                if related_docs:
                    print(f"   Related documents: {len(related_docs)} found")
                    for doc in related_docs[:2]:
                        print(f"     - {doc.get('title', 'Unknown')} (shared entities: {doc.get('shared_entities', 0)})")
        
        return results
        
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return []

def get_system_status():
    """Check the status of both storage systems."""
    print_subsection("📊 SYSTEM STATUS")
    
    try:
        # Check ChromaDB
        chroma_store = ChromaDBStore()
        chroma_info = chroma_store.get_collection_info()
        print(f"Vector Store (ChromaDB): {chroma_info.get('count', 0)} chunks")
        
        # Check Neo4j
        neo4j_client = Neo4jClient()
        graph_stats = neo4j_client.get_statistics()
        print(f"Knowledge Graph (Neo4j): {graph_stats.get('documents', 0)} docs, {graph_stats.get('entities', 0)} entities, {graph_stats.get('relationships', 0)} relationships")
        
        neo4j_client.close()
        
        has_data = chroma_info.get('count', 0) > 0 and graph_stats.get('documents', 0) > 0
        return has_data
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def run_retrieval_test(query: str):
    """Run complete retrieval test for a given query."""
    print_separator(f"TESTING QUERY: '{query}'")
    
    try:
        # Initialize components
        chroma_store = ChromaDBStore()
        neo4j_client = Neo4jClient()
        graph_builder = SimpleGraphBuilder(neo4j_client)
        embedder = EmbeddingGenerator()
        hybrid_retriever = HybridRetriever(chroma_store, neo4j_client, embedder)
        
        # Test individual components
        vector_results = test_vector_store_only(query, chroma_store, embedder)
        graph_results = test_knowledge_graph_only(query, graph_builder)
        hybrid_results = test_hybrid_retrieval(query, hybrid_retriever)
        
        # Summary
        print_subsection("📋 QUERY SUMMARY")
        print(f"Vector Store Results: {len(vector_results)}")
        print(f"Knowledge Graph Results: {len(graph_results)}")
        print(f"Hybrid Results: {len(hybrid_results)}")
        
        # Close connections
        neo4j_client.close()
        hybrid_retriever.close()
        graph_builder.close()
        
        return {
            "vector_results": len(vector_results),
            "graph_results": len(graph_results),
            "hybrid_results": len(hybrid_results)
        }
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_mode():
    """Run in interactive mode for testing multiple queries."""
    print_separator("🚀 INTERACTIVE RETRIEVAL TESTING", "=", 70)
    print("Enter queries to test retrieval from both vector store and knowledge graph.")
    print("Type 'status' to check system status, 'quit' or 'exit' to stop.")
    
    # Check initial status
    system_ready = get_system_status()
    
    if not system_ready:
        print("\n⚠️  Warning: System may not be fully ready. Some components might be empty.")
        print("   Consider running the ingestion pipeline first: python test_pipeline.py")
    
    while True:
        try:
            print("\n" + "=" * 50)
            query = input("Enter your query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'status':
                get_system_status()
                continue
            elif not query:
                print("Please enter a query.")
                continue
            
            # Run the retrieval test
            results = run_retrieval_test(query)
            
            if results:
                print(f"\n✅ Test completed successfully!")
            else:
                print(f"\n❌ Test failed!")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function - can run with predefined queries or interactive mode."""
    if len(sys.argv) > 1:
        # Command line mode with query argument
        query = " ".join(sys.argv[1:])
        run_retrieval_test(query)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
