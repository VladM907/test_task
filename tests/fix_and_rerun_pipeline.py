#!/usr/bin/env python3
"""
Script to re-run ingestion with fixed MENTIONS relationships
"""
import sys
import os

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ingestion.enhanced_pipeline import EnhancedIngestionPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🔄 Re-running ingestion pipeline with fixed MENTIONS relationships...")
    print()
    
    try:
        # Initialize enhanced pipeline
        pipeline = EnhancedIngestionPipeline("data")
        
        # Clear existing data and rebuild everything
        print("📊 Running complete pipeline (clearing existing data)...")
        stats = pipeline.run_complete_pipeline(
            store_in_vector_db=True,
            build_knowledge_graph=True,
            clear_existing=True  # This will clear and rebuild everything
        )
        
        print("\n✅ Pipeline Results:")
        print(f"Success: {stats.get('success', False)}")
        
        if stats.get('success'):
            base_stats = stats.get('base_ingestion', {})
            vector_stats = stats.get('vector_storage', {})
            graph_stats = stats.get('knowledge_graph', {})
            
            print(f"\n📄 Documents processed: {base_stats.get('total_documents', 0)}")
            print(f"📝 Total chunks: {base_stats.get('total_chunks', 0)}")
            print(f"🧮 Total embeddings: {base_stats.get('total_embeddings', 0)}")
            
            print(f"\n💾 Vector storage:")
            print(f"   Chunks stored: {vector_stats.get('chunks_stored', 0)}")
            print(f"   Success: {vector_stats.get('success', False)}")
            
            print(f"\n🕸️  Knowledge graph:")
            print(f"   Documents: {graph_stats.get('documents_processed', 0)}")
            print(f"   Chunks: {graph_stats.get('chunks_added', 0)}")
            print(f"   Entities: {graph_stats.get('entities_extracted', 0)}")
            
            errors = graph_stats.get('errors', [])
            if errors:
                print(f"   Errors: {len(errors)}")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"     - {error}")
        else:
            print(f"❌ Error: {stats.get('error', 'Unknown error')}")
        
        # Get final statistics
        print("\n📈 Final System Statistics:")
        final_stats = pipeline.get_comprehensive_stats()
        
        vector_info = final_stats.get('vector_database', {})
        graph_info = final_stats.get('knowledge_graph', {})
        
        print(f"Vector DB: {vector_info.get('count', 0)} chunks")
        print(f"Graph DB: {graph_info.get('documents', 0)} docs, {graph_info.get('chunks', 0)} chunks, {graph_info.get('entities', 0)} entities, {graph_info.get('relationships', 0)} relationships")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    print("\n🎯 Next step: Run 'python3 test_retrieval.py' to test the fixed MENTIONS relationships!")
    return 0

if __name__ == "__main__":
    exit(main())
