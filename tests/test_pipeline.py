#!/usr/bin/env python3
"""
Enhanced test script for the ingestion and retrieval pipeline with Neo4j integration.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.enhanced_pipeline import EnhancedIngestionPipeline
from backend.retrieval.chromadb_store import ChromaDBStore
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.knowledge_graph.graph_builder import SimpleGraphBuilder
from backend.ingestion.embedding import EmbeddingGenerator
"""
Enhanced test script for the ingestion and retrieval pipeline.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.chromadb_store import ChromaDBStore
from backend.ingestion.embedding import EmbeddingGenerator

def test_ingestion_pipeline():
    """Test the complete ingestion pipeline."""
    print("=" * 60)
    print("TESTING INGESTION PIPELINE")
    print("=" * 60)
    
    # Setup
    data_dir = project_root / "data"
    print(f"Data directory: {data_dir}")
    print(f"Data directory exists: {data_dir.exists()}")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return False
    
    # List files in data directory
    files = list(data_dir.rglob("*"))
    print(f"Files found: {len(files)}")
    for f in files:
        if f.is_file():
            print(f"  - {f.name} ({f.suffix}) - {f.stat().st_size} bytes")
    
    try:
        # Initialize pipeline
        print("\n🔄 Initializing pipeline...")
        pipeline = IngestionPipeline(str(data_dir), chunk_size=500, chunk_overlap=50)
        
        # Run pipeline
        print("🔄 Running ingestion pipeline...")
        docs = pipeline.run()
        
        # Show stats
        stats = pipeline.get_stats(docs)
        print(f"\n📊 Pipeline Results:")
        print(f"  - Documents processed: {stats.get('total_documents', 0)}")
        print(f"  - Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  - Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"  - Average chunks per doc: {stats.get('avg_chunks_per_doc', 0):.1f}")
        print(f"  - File types: {stats.get('file_types', {})}")
        
        # Show sample data
        if docs:
            print(f"\n📄 Sample document:")
            doc = docs[0]
            print(f"  - Path: {doc['path']}")
            print(f"  - Content length: {len(doc['full_content'])} chars")
            print(f"  - Number of chunks: {len(doc['chunks'])}")
            print(f"  - Number of embeddings: {len(doc['embeddings'])}")
            if doc['chunks']:
                print(f"  - First chunk preview: {doc['chunks'][0][:100]}...")
            if doc['embeddings']:
                print(f"  - Embedding dimension: {len(doc['embeddings'][0])}")
        
        return docs
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_chromadb_storage(docs):
    """Test ChromaDB storage and retrieval."""
    print("\n" + "=" * 60)
    print("TESTING CHROMADB STORAGE")
    print("=" * 60)
    
    if not docs:
        print("❌ No documents to test with!")
        return False
    
    try:
        # Initialize ChromaDB
        print("🔄 Initializing ChromaDB...")
        chroma_dir = project_root / "test_chroma_db"
        store = ChromaDBStore(str(chroma_dir), "test_collection")
        
        # Clear existing data
        print("🔄 Clearing existing collection...")
        store.clear_collection()
        
        # Prepare chunks for storage
        print("🔄 Preparing chunks for storage...")
        chunk_dicts = []
        for doc in docs:
            for i, (chunk, embedding) in enumerate(zip(doc["chunks"], doc["embeddings"])):
                chunk_dicts.append({
                    "path": doc["path"],
                    "chunk_id": i,
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": doc.get("metadata", {})
                })
        
        print(f"📊 Prepared {len(chunk_dicts)} chunks for storage")
        
        # Store in ChromaDB
        print("🔄 Storing chunks in ChromaDB...")
        success = store.add_chunks(chunk_dicts)
        
        if not success:
            print("❌ Failed to store chunks!")
            return False
        
        # Get collection info
        info = store.get_collection_info()
        print(f"📊 Collection info: {info}")
        
        # Test retrieval
        print("🔄 Testing retrieval...")
        if chunk_dicts:
            # Use first chunk's embedding for query
            query_embedding = chunk_dicts[0]["embedding"]
            results = store.query(query_embedding, n_results=3)
            
            print(f"📊 Retrieved {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    - Distance: {result['distance']:.4f}")
                print(f"    - Similarity: {result['similarity']:.4f}")
                print(f"    - Path: {result['metadata'].get('path', 'unknown')}")
                print(f"    - Content preview: {result['content'][:100]}...")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_connection():
    """Test Neo4j connection and basic operations."""
    print("\n" + "=" * 60)
    print("TESTING NEO4J CONNECTION")
    print("=" * 60)
    
    try:
        # Test connection
        print("🔄 Testing Neo4j connection...")
        with Neo4jClient() as client:
            stats = client.get_statistics()
            print(f"✅ Neo4j connected successfully")
            print(f"📊 Current graph stats: {stats}")
            
            # Test basic operations
            print("🔄 Testing basic operations...")
            
            # Clear and test
            client.clear_database()
            print("✅ Database cleared")
            
            # Create indexes
            client.create_indexes()
            print("✅ Indexes created")
            
            # Test document creation
            success = client.add_document(
                path="/test/document.txt",
                content="This is a test document about artificial intelligence and machine learning.",
                metadata={"test": True}
            )
            
            if success:
                print("✅ Test document added")
                
                # Test chunk creation
                chunk_success = client.add_chunk(
                    document_path="/test/document.txt",
                    chunk_id=0,
                    content="This is a test document about artificial intelligence.",
                    chunk_index=0
                )
                
                if chunk_success:
                    print("✅ Test chunk added")
                    
                    # Test entity extraction
                    entities = client.extract_and_add_entities(
                        content="This is a test document about artificial intelligence and machine learning.",
                        chunk_id="test_chunk_0",
                        document_path="/test/document.txt"
                    )
                    
                    print(f"✅ Extracted {len(entities)} entities: {entities[:5]}")
                    
                    # Test relationships
                    doc_entities = client.get_document_entities("/test/document.txt")
                    print(f"✅ Document has {len(doc_entities)} entities")
                    
                    return True
                
        return False
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_pipeline():
    """Test the complete enhanced pipeline with both vector and graph storage."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED PIPELINE")
    print("=" * 60)
    
    try:
        data_dir = project_root / "data"
        
        print("🔄 Initializing enhanced pipeline...")
        with EnhancedIngestionPipeline(str(data_dir)) as pipeline:
            
            # Run complete pipeline
            print("🔄 Running complete enhanced pipeline...")
            stats = pipeline.run_complete_pipeline(
                store_in_vector_db=True,
                build_knowledge_graph=True,
                clear_existing=True
            )
            
            print(f"📊 Enhanced Pipeline Results:")
            if stats.get("success"):
                print("  ✅ Pipeline completed successfully")
                
                base_stats = stats.get("base_ingestion", {})
                print(f"  📄 Documents processed: {base_stats.get('total_documents', 0)}")
                print(f"  📄 Total chunks: {base_stats.get('total_chunks', 0)}")
                
                vector_stats = stats.get("vector_storage", {})
                print(f"  🔍 Chunks stored in vector DB: {vector_stats.get('chunks_stored', 0)}")
                
                graph_stats = stats.get("knowledge_graph", {})
                print(f"  🕸️  Documents in graph: {graph_stats.get('documents_processed', 0)}")
                print(f"  🕸️  Entities extracted: {graph_stats.get('entities_extracted', 0)}")
                
                # Test retrieval
                print("\n🔄 Testing retrieval capabilities...")
                retrieval_test = pipeline.test_retrieval("Constitution United States")
                print(f"  🔍 Retrieval test: {retrieval_test.get('retrieval_working', False)}")
                print(f"  🔍 Results found: {retrieval_test.get('results_found', 0)}")
                
                return True
            else:
                print(f"  ❌ Pipeline failed: {stats.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_retrieval():
    """Test hybrid retrieval combining vector and graph search."""
    print("\n" + "=" * 60)
    print("TESTING HYBRID RETRIEVAL")
    print("=" * 60)
    
    try:
        print("🔄 Initializing hybrid retriever...")
        
        # Initialize components
        chroma_store = ChromaDBStore()
        neo4j_client = Neo4jClient()
        
        with HybridRetriever(chroma_store, neo4j_client) as retriever:
            
            # Test basic search
            print("🔄 Testing hybrid search...")
            results = retriever.search(
                query="Article 3, Section 2 of the Constitution United States government",
                n_results=3,
                use_graph_expansion=True
            )
            
            print(f"📊 Hybrid Search Results: {len(results)} found")
            
            for i, result in enumerate(results[:2]):  # Show top 2 results
                print(f"\n  Result {i+1}:")
                print(f"    Source: {result.get('source', 'unknown')}")
                print(f"    Similarity: {result.get('similarity', 0):.3f}")
                print(f"    Content preview: {result.get('content', '')[:100]}...")
                
                graph_context = result.get('graph_context', {})
                if graph_context:
                    related_docs = graph_context.get('related_documents', [])
                    entities = graph_context.get('entities', [])
                    print(f"    Related docs: {len(related_docs)}")
                    print(f"    Entities: {[e.get('name', '') for e in entities[:3]]}")
            
            # Test entity search
            print("\n🔄 Testing entity-based search...")
            entity_results = retriever.search_by_entity("constitution", limit=3)
            print(f"📊 Entity search results: {len(entity_results)} found")
            
            # Get statistics
            stats = retriever.get_statistics()
            print(f"\n📊 Retriever Statistics:")
            print(f"  Vector store: {stats.get('vector_store', {})}")
            print(f"  Knowledge graph: {stats.get('knowledge_graph', {})}")
            
            return len(results) > 0
            
    except Exception as e:
        print(f"❌ Hybrid retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    """Test embedding generation quality."""
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING QUALITY")
    print("=" * 60)
    
    try:
        embedder = EmbeddingGenerator()
        
        # Test similar texts
        texts = [
            "The United States Constitution is the supreme law of the United States.",
            "The Constitution of the United States is the fundamental law of America.",
            "Python is a programming language used for data science.",
            "Machine learning requires large datasets and computational power."
        ]
        
        print("🔄 Generating embeddings for test texts...")
        embeddings = embedder.embed(texts)
        
        print(f"📊 Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Calculate similarities
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print("\n📊 Similarity matrix:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i <= j:
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    print(f"  Text {i+1} vs Text {j+1}: {sim:.3f}")
                    if i != j and sim > 0.5:
                        print(f"    ✅ High similarity detected (expected for texts 1&2)")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_quality():
    """Test embedding generation quality."""
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING QUALITY")
    print("=" * 60)
    
    try:
        embedder = EmbeddingGenerator()
        
        # Test similar texts
        texts = [
            "The United States Constitution is the supreme law of the United States.",
            "The Constitution of the United States is the fundamental law of America.",
            "Python is a programming language used for data science.",
            "Machine learning requires large datasets and computational power."
        ]
        
        print("🔄 Generating embeddings for test texts...")
        embeddings = embedder.embed(texts)
        
        print(f"📊 Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Calculate similarities
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print("\n📊 Similarity matrix:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i <= j:
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    print(f"  Text {i+1} vs Text {j+1}: {sim:.3f}")
                    if i != j and sim > 0.5:
                        print(f"    ✅ High similarity detected (expected for texts 1&2)")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests including Neo4j integration."""
    print("🚀 Starting Enhanced Pipeline Tests with Neo4j")
    print(f"Project root: {project_root}")
    
    # Test Neo4j connection first
    neo4j_success = test_neo4j_connection()
    
    # Test basic ingestion
    docs = test_ingestion_pipeline()
    
    # Test ChromaDB
    if docs:
        chromadb_success = test_chromadb_storage(docs)
    else:
        chromadb_success = False
    
    # Test enhanced pipeline (if Neo4j is working)
    if neo4j_success:
        enhanced_success = test_enhanced_pipeline()
        hybrid_success = test_hybrid_retrieval()
    else:
        enhanced_success = False
        hybrid_success = False
    
    # Test embedding quality
    embedding_success = test_embedding_quality()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Neo4j Connection: {'PASSED' if neo4j_success else 'FAILED'}")
    print(f"✅ Ingestion Pipeline: {'PASSED' if docs else 'FAILED'}")
    print(f"✅ ChromaDB Storage: {'PASSED' if chromadb_success else 'FAILED'}")
    print(f"✅ Enhanced Pipeline: {'PASSED' if enhanced_success else 'FAILED'}")
    print(f"✅ Hybrid Retrieval: {'PASSED' if hybrid_success else 'FAILED'}")
    print(f"✅ Embedding Quality: {'PASSED' if embedding_success else 'FAILED'}")
    
    all_tests_passed = all([
        neo4j_success, docs, chromadb_success, 
        enhanced_success, hybrid_success, embedding_success
    ])
    
    if all_tests_passed:
        print("\n🎉 All tests passed! Your enhanced pipeline with Neo4j is ready to use.")
        print("\n📋 Next Steps:")
        print("  1. Start your services: docker-compose up -d")
        print("  2. Run the enhanced pipeline on your documents")
        print("  3. Use hybrid retrieval for better search results")
        print("  4. Integrate with your LLM layer")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        if not neo4j_success:
            print("   🔧 Make sure Neo4j is running: docker-compose up neo4j")

if __name__ == "__main__":
    main()
