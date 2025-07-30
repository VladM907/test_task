#!/usr/bin/env python3
"""
Test script for the RAG pipeline with Ollama and OpenAI support.
"""
import sys
import os

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.rag.pipeline import RAGPipeline, ModelProvider
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama():
    """Test RAG with Ollama (local model)."""
    print("🦙 Testing RAG with Ollama...")
    try:
        # Initialize with Ollama
        rag = RAGPipeline(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1",  # Change to your available model
            temperature=0.1
        )
        
        # Test questions
        questions = [
            "What is the House of Representatives?",
            "How are Representatives chosen?",
            "What powers does Congress have?"
        ]
        
        for question in questions:
            print(f"\n📝 Question: {question}")
            print("🤖 Answer:")
            answer = rag.ask(question)
            print(answer)
            print("-" * 50)
        
        rag.close()
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print("💡 And that you have a model: ollama pull llama3.1")

def test_openai():
    """Test RAG with OpenAI."""
    print("🤖 Testing RAG with OpenAI...")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Initialize with OpenAI
        rag = RAGPipeline(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",  # or "gpt-4"
            temperature=0.1,
            openai_api_key=api_key
        )
        
        # Test question
        question = "What is the composition of the House of Representatives according to the Constitution?"
        
        print(f"\n📝 Question: {question}")
        print("🤖 Answer:")
        answer = rag.ask(question)
        print(answer)
        
        rag.close()
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")

def test_retrieval_only():
    """Test just the retrieval component."""
    print("🔍 Testing retrieval component...")
    
    try:
        rag = RAGPipeline(
            provider=ModelProvider.OLLAMA,  # Provider doesn't matter for retrieval test
            model_name="llama3.1"
        )
        
        question = "House of Representatives"
        retrieval_info = rag.get_retrieval_info(question)
        
        print(f"📝 Query: {question}")
        print(f"📊 Retrieved {retrieval_info['num_results']} results")
        print()
        
        for i, result in enumerate(retrieval_info['results'][:2]):  # Show first 2
            print(f"Result {i+1}:")
            print(f"  Source: {result['source']}")
            print(f"  Similarity: {result.get('similarity', 'N/A')}")
            print(f"  Content: {result['content'][:100]}...")
            
            graph_ctx = result.get('graph_context', {})
            entities = graph_ctx.get('entities', [])
            entity_names = [e['name'] if isinstance(e, dict) else str(e) for e in entities[:3]]
            print(f"  Entities: {entity_names}")
            print()
        
        rag.close()
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")

def interactive_mode():
    """Interactive chat mode."""
    print("💬 Interactive RAG Chat Mode")
    print("Type 'quit' to exit, 'switch' to change model provider")
    
    # Choose provider
    while True:
        provider_choice = input("\nChoose provider (ollama/openai): ").lower().strip()
        if provider_choice in ['ollama', 'openai']:
            break
        print("Please choose 'ollama' or 'openai'")
    
    try:
        if provider_choice == 'ollama':
            rag = RAGPipeline(
                provider=ModelProvider.OLLAMA,
                model_name="llama3.1",
                temperature=0.1
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                api_key = input("Enter OpenAI API key: ").strip()
            
            rag = RAGPipeline(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=api_key
            )
        
        chat_history = []
        
        while True:
            question = input(f"\n🤔 Your question ({provider_choice}): ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'switch':
                rag.close()
                return interactive_mode()
            elif not question:
                continue
            
            print("🤖 Thinking...")
            
            try:
                answer = rag.ask_with_context(question, chat_history)
                print(f"🤖 Answer: {answer}")
                
                # Update chat history
                chat_history.append({
                    "human": question,
                    "assistant": answer
                })
                
                # Keep only last 10 exchanges
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        rag.close()
        
    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")

def main():
    print("🚀 RAG Pipeline Testing")
    print("Choose a test mode:")
    print("1. Test Ollama (local)")
    print("2. Test OpenAI") 
    print("3. Test retrieval only")
    print("4. Interactive chat mode")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            test_ollama()
            break
        elif choice == '2':
            test_openai()
            break
        elif choice == '3':
            test_retrieval_only()
            break
        elif choice == '4':
            interactive_mode()
            break
        else:
            print("Please enter 1, 2, 3, or 4")

if __name__ == "__main__":
    main()
