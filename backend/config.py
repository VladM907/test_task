"""
Configuration management for models, sources, and API keys.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Neo4j Configuration
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password123"))
    
    # ChromaDB Configuration
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    chroma_collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base"))

@dataclass
class LLMConfig:
    """LLM configuration."""
    # Default provider and model
    default_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "ollama"))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "llama3.1"))
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.1")))
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    
    # Ollama Configuration
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    
    # Model limits
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4000")))
    context_window: int = field(default_factory=lambda: int(os.getenv("CONTEXT_WINDOW", "8000")))

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    cache_dir: str = field(default_factory=lambda: os.getenv("EMBEDDING_CACHE_DIR", "./data/embedding_cache"))
    batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))

@dataclass
class IngestionConfig:
    """Document ingestion configuration."""
    data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    supported_extensions: list = field(default_factory=lambda: [".pdf", ".txt", ".md", ".docx"])
    
    # Graph extraction settings
    enable_entity_extraction: bool = field(default_factory=lambda: os.getenv("ENABLE_ENTITY_EXTRACTION", "true").lower() == "true")
    entity_extraction_model: str = field(default_factory=lambda: os.getenv("ENTITY_EXTRACTION_MODEL", "en_core_web_sm"))

@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    default_n_results: int = field(default_factory=lambda: int(os.getenv("DEFAULT_N_RESULTS", "5")))
    max_n_results: int = field(default_factory=lambda: int(os.getenv("MAX_N_RESULTS", "20")))
    
    # Hybrid retrieval settings
    enable_graph_expansion: bool = field(default_factory=lambda: os.getenv("ENABLE_GRAPH_EXPANSION", "true").lower() == "true")
    graph_expansion_limit: int = field(default_factory=lambda: int(os.getenv("GRAPH_EXPANSION_LIMIT", "3")))
    
    # Similarity thresholds
    min_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.5")))

@dataclass
class APIConfig:
    """API configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true")
    
    # CORS settings
    cors_origins: list = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    
    # Rate limiting
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60")))
    
    # Session management
    session_timeout: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT", "3600")))  # 1 hour

@dataclass
class UIConfig:
    """UI configuration."""
    streamlit_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))
    streamlit_host: str = field(default_factory=lambda: os.getenv("STREAMLIT_HOST", "0.0.0.0"))
    
    # Theme settings
    primary_color: str = field(default_factory=lambda: os.getenv("UI_PRIMARY_COLOR", "#667eea"))
    background_color: str = field(default_factory=lambda: os.getenv("UI_BACKGROUND_COLOR", "#ffffff"))

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Component-specific logging
    enable_api_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_API_LOGGING", "true").lower() == "true")
    enable_rag_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_RAG_LOGGING", "true").lower() == "true")
    enable_db_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_DB_LOGGING", "false").lower() == "true")

@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    jwt_secret: Optional[str] = field(default_factory=lambda: os.getenv("JWT_SECRET"))
    allowed_hosts: list = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","))
    
    # File upload security
    max_file_size: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE", "50")))  # MB
    allowed_file_types: list = field(default_factory=lambda: os.getenv("ALLOWED_FILE_TYPES", ".pdf,.txt,.md,.docx").split(","))

class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.ingestion = IngestionConfig()
        self.retrieval = RetrievalConfig()
        self.api = APIConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.database.chroma_persist_dir,
            self.embedding.cache_dir,
            self.ingestion.data_folder,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_env_vars_info(self) -> Dict[str, Any]:
        """Get information about environment variables and their current values."""
        return {
            "database": {
                "neo4j_uri": self.database.neo4j_uri,
                "neo4j_user": self.database.neo4j_user,
                "neo4j_password": "***" if self.database.neo4j_password else None,
                "chroma_persist_dir": self.database.chroma_persist_dir,
                "chroma_collection_name": self.database.chroma_collection_name,
            },
            "llm": {
                "default_provider": self.llm.default_provider,
                "default_model": self.llm.default_model,
                "default_temperature": self.llm.default_temperature,
                "openai_api_key": "***" if self.llm.openai_api_key else None,
                "ollama_base_url": self.llm.ollama_base_url,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
            },
            "ui": {
                "streamlit_port": self.ui.streamlit_port,
                "streamlit_host": self.ui.streamlit_host,
            }
        }
    
    def validate(self) -> Dict[str, list]:
        """Validate configuration and return any issues."""
        issues = {
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        if not Path(self.ingestion.data_folder).exists():
            issues["warnings"].append(f"Data folder does not exist: {self.ingestion.data_folder}")
        
        # Check LLM configuration
        if self.llm.default_provider == "openai" and not self.llm.openai_api_key:
            issues["errors"].append("OpenAI API key is required when using OpenAI provider")
        
        # Check Neo4j configuration
        if not self.database.neo4j_password or self.database.neo4j_password == "password":
            issues["warnings"].append("Using default Neo4j password - consider changing for security")
        
        # Check chunk sizes
        if self.ingestion.chunk_overlap >= self.ingestion.chunk_size:
            issues["errors"].append("Chunk overlap must be smaller than chunk size")
        
        return issues

# Global configuration instance
config = Config()
