#!/usr/bin/env python3
"""
Test script for chatbot memory functionality.
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

def test_chatbot_memory():
    """Test the chatbot memory functionality."""
    print("🧠 Testing Chatbot Memory...")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1",
            temperature=0.1
        )
        
        # Simulate a conversation
        chat_history = []
        
        # First question
        question1 = "What powers does Congress have?"
        print(f"\n👤 User: {question1}")
        answer1 = rag.ask(question1)
        print(f"🤖 Assistant: {answer1[:200]}...")
        
        # Add to history
        chat_history.append({
            "human": question1,
            "assistant": answer1
        })
        
        # Second question (context-dependent)
        question2 = "What was my last question about?"
        print(f"\n👤 User: {question2}")
        answer2 = rag.ask_with_context(question2, chat_history)
        print(f"🤖 Assistant: {answer2}")
        
        # Add to history
        chat_history.append({
            "human": question2,
            "assistant": answer2
        })
        
        # Third question (also context-dependent)
        question3 = "Can you elaborate on that first topic?"
        print(f"\n👤 User: {question3}")
        answer3 = rag.ask_with_context(question3, chat_history)
        print(f"🤖 Assistant: {answer3[:300]}...")
        
        rag.close()
        print("\n✅ Memory test completed!")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")

if __name__ == "__main__":
    test_chatbot_memory()
