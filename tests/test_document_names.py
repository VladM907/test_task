#!/usr/bin/env python3
"""
Test script to check document name extraction in the RAG pipeline.
"""
import sys
import os
sys.path.append('/root/projects/test_task')

from backend.rag.pipeline import RAGPipeline, ModelProvider

def test_document_name_extraction():
    """Test if document names are properly extracted."""
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1"
        )
        
        # Test retrieval info to see the data structure
        print("Testing retrieval info...")
        info = pipeline.get_retrieval_info("What is the Constitution?")
        
        if "results" in info:
            print(f"\nFound {len(info['results'])} results")
            
            for i, result in enumerate(info['results'][:2]):  # Show first 2 results
                print(f"\n--- Result {i+1} ---")
                print(f"Content preview: {result.get('content', '')[:100]}...")
                print(f"Metadata: {result.get('metadata', {})}")
                print(f"Source field: {result.get('source', 'NOT_FOUND')}")
                print(f"Similarity: {result.get('similarity', 'N/A')}")
                
                # Test our extraction logic
                metadata = result.get("metadata", {})
                source = metadata.get("path", "unknown")
                if source != "unknown":
                    doc_name = os.path.basename(source)
                    doc_name = os.path.splitext(doc_name)[0]
                    print(f"Extracted document name: '{doc_name}'")
                else:
                    print("No path found in metadata")
        else:
            print(f"Error in retrieval: {info}")
        
        # Test a simple question
        print("\n" + "="*50)
        print("Testing actual RAG pipeline...")
        response = pipeline.ask("What is the Constitution?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_name_extraction()
