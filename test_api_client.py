#!/usr/bin/env python3
"""
Example API client for testing the RAG System API.
"""
import requests
import json
import time
from typing import Optional

class RAGAPIClient:
    """Simple client for the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.current_session_id: Optional[str] = None
    
    def health_check(self):
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def configure_model(self, provider: str, model_name: str, temperature: float = 0.1, openai_api_key: Optional[str] = None):
        """Configure the RAG model."""
        data = {
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature
        }
        if openai_api_key:
            data["openai_api_key"] = openai_api_key
        
        try:
            response = self.session.post(f"{self.base_url}/configure", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, message: str, use_context: bool = True, session_id: Optional[str] = None):
        """Send a chat message."""
        data = {
            "message": message,
            "use_context": use_context,
            "session_id": session_id or self.current_session_id
        }
        
        try:
            response = self.session.post(f"{self.base_url}/chat", json=data)
            result = response.json()
            
            # Store session ID for future requests
            if "session_id" in result:
                self.current_session_id = result["session_id"]
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def search(self, query: str, n_results: int = 5, use_graph: bool = True):
        """Search the knowledge base."""
        data = {
            "query": query,
            "n_results": n_results,
            "use_graph": use_graph
        }
        
        try:
            response = self.session.post(f"{self.base_url}/search", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_sessions(self):
        """Get all chat sessions."""
        try:
            response = self.session.get(f"{self.base_url}/sessions")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_file(self, file_path: str):
        """Upload a document file."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/upload", files=files)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def ingest_documents(self, data_folder: str = "data", clear_existing: bool = False):
        """Trigger document ingestion."""
        data = {
            "data_folder": data_folder,
            "clear_existing": clear_existing
        }
        
        try:
            response = self.session.post(f"{self.base_url}/ingest", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Example usage of the API client."""
    print("🔗 RAG API Client Example")
    
    # Initialize client
    client = RAGAPIClient()
    
    # Check health
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"Status: {health.get('status', 'unknown')}")
    
    # Configure model (Ollama)
    print("\n2. Configuring Model:")
    config_result = client.configure_model("ollama", "llama3.1")
    print(f"Configuration: {config_result.get('status', 'failed')}")
    
    # Wait a moment for initialization
    time.sleep(2)
    
    # Test chat
    print("\n3. Testing Chat:")
    
    # First message
    response1 = client.chat("What is the House of Representatives?")
    if "response" in response1:
        print(f"🤖 Response: {response1['response'][:200]}...")
        print(f"📎 Sources: {len(response1.get('sources', []))}")
    else:
        print(f"❌ Error: {response1.get('error', 'Unknown error')}")
    
    # Context-dependent message
    response2 = client.chat("What was my last question about?")
    if "response" in response2:
        print(f"🤖 Context Response: {response2['response']}")
    else:
        print(f"❌ Error: {response2.get('error', 'Unknown error')}")
    
    # Test search
    print("\n4. Testing Search:")
    search_result = client.search("congressional powers")
    if "results" in search_result:
        print(f"🔍 Found {search_result['total_results']} results")
        for i, result in enumerate(search_result['results'][:2]):
            content = result.get('content', '') if isinstance(result, dict) else str(result)
            print(f"   Result {i+1}: {content[:100]}...")
    else:
        print(f"❌ Search Error: {search_result.get('error', 'Unknown error')}")
    
    # List sessions
    print("\n5. Session Info:")
    sessions = client.get_sessions()
    if "sessions" in sessions:
        print(f"📝 Active sessions: {sessions['total']}")
        if sessions['sessions']:
            session = sessions['sessions'][0]
            if isinstance(session, dict):
                session_id = session.get('session_id', 'unknown')
                message_count = session.get('message_count', 0)
                print(f"   Session ID: {session_id[:8]}...")
                print(f"   Messages: {message_count}")
    else:
        print(f"❌ Sessions Error: {sessions.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
