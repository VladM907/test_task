"""
Streamlit UI for the RAG System.
Provides a user-friendly interface for chatting, searching, and managing the system.
"""
import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .search-result {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #4caf50;
    }
    
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #f0f0f0, #ffffff);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# API Client Class
class RAGAPIClient:
    """Simple client for the RAG API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self):
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
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
            response = requests.post(f"{self.base_url}/configure", json=data, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def chat(self, message: str, use_context: bool = True, session_id: Optional[str] = None):
        """Send a chat message."""
        data = {
            "message": message,
            "use_context": use_context,
            "session_id": session_id
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=30)
            return response.json()
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
            response = requests.post(f"{self.base_url}/search", json=data, timeout=15)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_sessions(self):
        """Get all chat sessions."""
        try:
            response = requests.get(f"{self.base_url}/sessions", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_file(self, file_bytes, filename: str):
        """Upload a document file."""
        try:
            files = {'file': (filename, file_bytes)}
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=60)
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
            response = requests.post(f"{self.base_url}/ingest", json=data, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    return RAGAPIClient()

# Helper functions
def format_timestamp(timestamp):
    """Format timestamp for display."""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp
    return str(timestamp)

def display_chat_message(role: str, content: str, timestamp: Optional[str] = None):
    """Display a chat message with styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You</strong> {f"({format_timestamp(timestamp)})" if timestamp else ""}<br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant</strong> {f"({format_timestamp(timestamp)})" if timestamp else ""}<br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_search_result(result: Dict, index: int):
    """Display a search result."""
    content = result.get('content', '')
    similarity = result.get('similarity', 'N/A')
    source = result.get('source', 'unknown')
    
    # Format similarity for display
    if isinstance(similarity, (int, float)):
        similarity_display = f"{similarity:.3f}"
    else:
        similarity_display = str(similarity)
    
    # Truncate content for display
    truncated_content = content[:300] + "..." if len(content) > 300 else content
    
    st.markdown(f"""
    <div class="search-result">
        <strong>Result {index + 1}</strong> (Similarity: {similarity_display})<br>
        <strong>Source:</strong> {source}<br>
        <em>{truncated_content}</em>
    </div>
    """, unsafe_allow_html=True)

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG System</h1>
        <p>Retrieval-Augmented Generation with Knowledge Graphs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    client = get_api_client()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System Status
        st.subheader("📊 System Status")
        
        if st.button("🔄 Refresh Status"):
            st.cache_resource.clear()
        
        # Get health status
        health_status = client.health_check()
        
        if health_status.get("status") == "healthy":
            st.markdown('<span class="status-healthy">✅ Healthy</span>', unsafe_allow_html=True)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", health_status.get("total_documents", 0))
                st.metric("📝 Chunks", health_status.get("total_chunks", 0))
            with col2:
                st.metric("🕸️ Entities", health_status.get("total_entities", 0))
                st.metric("🤖 Model", health_status.get("model_name", "Unknown"))
        else:
            st.markdown('<span class="status-error">❌ Error</span>', unsafe_allow_html=True)
            if "error" in health_status:
                st.error(f"Error: {health_status['error']}")
        
        st.divider()
        
        # Model Configuration
        st.subheader("🔧 Model Configuration")
        
        provider = st.selectbox(
            "Provider",
            ["ollama", "openai"],
            help="Choose between local Ollama or cloud OpenAI"
        )
        
        if provider == "ollama":
            model_name = st.text_input("Model Name", value="llama3.1")
            openai_key = None
        else:
            model_name = st.selectbox("Model Name", ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini"])
            openai_key = st.text_input("OpenAI API Key", type="password")
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        if st.button("🚀 Configure Model"):
            with st.spinner("Configuring model..."):
                result = client.configure_model(provider, model_name, temperature, openai_key)
                if result.get("status") == "success":
                    st.success("✅ Model configured successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Configuration failed: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # File Upload
        st.subheader("📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload & Process"):
                with st.spinner("Uploading file..."):
                    result = client.upload_file(uploaded_file.getvalue(), uploaded_file.name)
                    if result.get("status") == "success":
                        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                        
                        # Trigger ingestion
                        with st.spinner("Processing document..."):
                            ingest_result = client.ingest_documents()
                            if ingest_result.get("status") == "started":
                                st.info("🔄 Document processing started in background")
                    else:
                        st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
        
        # Data Management
        st.subheader("🗂️ Data Management")
        
        if st.button("🔄 Reprocess Documents"):
            with st.spinner("Reprocessing documents..."):
                result = client.ingest_documents(clear_existing=True)
                if result.get("status") == "started":
                    st.success("✅ Document reprocessing started!")
                else:
                    st.error(f"❌ Reprocessing failed: {result.get('error', 'Unknown error')}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Search", "📝 Sessions", "📊 Analytics"])
    
    # Chat Tab
    with tab1:
        st.header("💬 Chat with the RAG System")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(
                    message["role"],
                    message["content"],
                    message.get("timestamp")
                )
        
        # Chat input form to prevent rerunning
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ask me anything about your documents...",
                    key="chat_input_form"
                )
            
            with col2:
                use_context = st.checkbox("Use Context", value=True, help="Use conversation history for context")
            
            send_button = st.form_submit_button("📤 Send", type="primary")
        
        # Process chat submission
        if send_button and user_input.strip():
            # Add user message to history
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            
            # Get AI response
            with st.spinner("🤖 Thinking..."):
                response = client.chat(
                    user_input,
                    use_context=use_context,
                    session_id=st.session_state.current_session_id
                )
            
            if "response" in response:
                # Update session ID
                if "session_id" in response:
                    st.session_state.current_session_id = response["session_id"]
                
                # Add assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": response["response"],
                    "timestamp": datetime.now().isoformat(),
                    "sources": response.get("sources", [])
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Show sources if available
                if response.get("sources"):
                    with st.expander(f"📎 Sources ({len(response['sources'])})"):
                        for i, source in enumerate(response["sources"]):
                            st.write(f"**Source {i+1}:**")
                            if isinstance(source, dict):
                                st.write(f"Content: {source.get('content', '')[:200]}...")
                                # Format similarity properly
                                similarity = source.get('similarity', 'N/A')
                                if isinstance(similarity, (int, float)):
                                    similarity_text = f"{similarity:.3f}"
                                else:
                                    similarity_text = str(similarity)
                                st.write(f"Similarity: {similarity_text}")
                            else:
                                st.write(f"Content: {str(source)[:200]}...")
                            st.divider()
            
            else:
                st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
            
            # Rerun to show new messages
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.rerun()
    
    # Search Tab
    with tab2:
        st.header("🔍 Search Knowledge Base")
        
        # Search form to prevent rerunning
        with st.form("search_form"):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input("Search Query", placeholder="Enter your search query...")
            
            with col2:
                n_results = st.number_input("Results", min_value=1, max_value=20, value=5)
            
            with col3:
                use_graph = st.checkbox("Use Graph", value=True, help="Include knowledge graph enhancement")
            
            search_button = st.form_submit_button("🔍 Search", type="primary")
        
        # Process search submission
        if search_button and search_query.strip():
            with st.spinner("🔍 Searching..."):
                search_results = client.search(search_query, n_results, use_graph)
            
            if "results" in search_results:
                st.session_state.search_results = search_results["results"]
                st.success(f"✅ Found {search_results.get('total_results', len(search_results['results']))} results")
                
                # Display results
                for i, result in enumerate(search_results["results"]):
                    if isinstance(result, dict):
                        display_search_result(result, i)
                    else:
                        st.write(f"**Result {i+1}:** {str(result)}")
            
            else:
                st.error(f"❌ Search failed: {search_results.get('error', 'Unknown error')}")
        
        # Display previous search results if any
        elif st.session_state.search_results:
            st.subheader("Previous Search Results")
            for i, result in enumerate(st.session_state.search_results):
                if isinstance(result, dict):
                    display_search_result(result, i)
                else:
                    st.write(f"**Result {i+1}:** {str(result)}")
    
    # Sessions Tab
    with tab3:
        st.header("📝 Chat Sessions")
        
        if st.button("🔄 Refresh Sessions (currently don't do anything...)"):
            pass
        
        # Get sessions
        sessions_data = client.get_sessions()
        
        if "sessions" in sessions_data:
            if sessions_data["sessions"]:
                # Create DataFrame for sessions
                sessions_df = pd.DataFrame(sessions_data["sessions"])
                sessions_df["created_at"] = pd.to_datetime(sessions_df["created_at"])
                sessions_df["last_activity"] = pd.to_datetime(sessions_df["last_activity"])
                
                st.dataframe(
                    sessions_df,
                    use_container_width=True,
                    column_config={
                        "session_id": st.column_config.TextColumn("Session ID", width="medium"),
                        "created_at": st.column_config.DatetimeColumn("Created", width="medium"),
                        "last_activity": st.column_config.DatetimeColumn("Last Activity", width="medium"),
                        "message_count": st.column_config.NumberColumn("Messages", width="small")
                    }
                )
                
                st.metric("📊 Total Sessions", len(sessions_data["sessions"]))
            else:
                st.info("No active sessions found.")
        else:
            st.error(f"❌ Failed to get sessions: {sessions_data.get('error', 'Unknown error')}")
    
    # Analytics Tab
    with tab4:
        st.header("📊 System Analytics")
        
        # System metrics
        health_status = client.health_check()
        
        if health_status.get("status") == "healthy":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>📄 Documents</h3>
                    <h2>{}</h2>
                </div>
                """.format(health_status.get("total_documents", 0)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>📝 Chunks</h3>
                    <h2>{}</h2>
                </div>
                """.format(health_status.get("total_chunks", 0)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>🕸️ Entities</h3>
                    <h2>{}</h2>
                </div>
                """.format(health_status.get("total_entities", 0)), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>🤖 Model</h3>
                    <h2>{}</h2>
                </div>
                """.format(health_status.get("model_name", "Unknown")), unsafe_allow_html=True)
            
            # Session analytics
            sessions_data = client.get_sessions()
            if "sessions" in sessions_data and sessions_data["sessions"]:
                st.subheader("📈 Session Analytics")
                
                sessions_df = pd.DataFrame(sessions_data["sessions"])
                sessions_df["created_at"] = pd.to_datetime(sessions_df["created_at"])
                
                # Messages per session chart
                st.bar_chart(sessions_df.set_index("session_id")["message_count"])
                
                # Sessions over time
                sessions_by_hour = sessions_df.groupby(sessions_df["created_at"].dt.hour).size()
                st.line_chart(sessions_by_hour)
        
        else:
            st.error("Cannot load analytics - system not healthy")

if __name__ == "__main__":
    main()
