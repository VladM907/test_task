"""
Configuration for RAG pipeline.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGConfig:
    """Configuration settings for RAG pipeline."""
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_DEFAULT_MODEL: str = "gpt-4.1-nano"
    
    # Ollama settings  
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    
    # RAG settings
    DEFAULT_TEMPERATURE: float = 0.1
    MAX_RETRIEVAL_RESULTS: int = 5
    GRAPH_EXPANSION_LIMIT: int = 3
    
    # Paths
    DATA_FOLDER: str = "data"
    CHROMA_DB_PATH: str = "./chroma_db"
    
    # Neo4j settings (from existing setup)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

# Example .env file content
ENV_TEMPLATE = """
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Neo4j Configuration (optional, defaults provided)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
"""
