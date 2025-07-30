"""
FastAPI REST API for the RAG system.
Provides endpoints for document ingestion, retrieval, and chat functionality.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
import asyncio
import json
import logging
from datetime import datetime
import os
from pathlib import Path
import tempfile
import shutil

# Import our RAG components
from ..rag.pipeline import RAGPipeline, ModelProvider
from ..ingestion.enhanced_pipeline import EnhancedIngestionPipeline
from ..retrieval.hybrid_retriever import HybridRetriever
from ..config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG System API",
    description="REST API for Retrieval-Augmented Generation with Knowledge Graphs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
rag_pipeline: Optional[RAGPipeline] = None
sessions: Dict[str, Dict] = {}  # Store chat sessions

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'human' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    use_context: bool = Field(True, description="Whether to use conversation context")
    stream: bool = Field(False, description="Whether to stream the response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session ID")
    sources: List[Dict] = Field(default_factory=list, description="Source documents")
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    n_results: int = Field(5, description="Number of results to return")
    use_graph: bool = Field(True, description="Whether to use knowledge graph enhancement")

class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[Dict] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")

class ConfigRequest(BaseModel):
    provider: str = Field(..., description="Model provider: 'ollama' or 'openai'")
    model_name: str = Field(..., description="Model name")
    temperature: float = Field(0.1, description="Sampling temperature")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key if using OpenAI")

class StatusResponse(BaseModel):
    status: str = Field(..., description="System status")
    model_provider: Optional[str] = Field(None, description="Current model provider")
    model_name: Optional[str] = Field(None, description="Current model name")
    total_documents: int = Field(0, description="Total documents indexed")
    total_chunks: int = Field(0, description="Total chunks indexed")
    total_entities: int = Field(0, description="Total entities in knowledge graph")

class IngestionRequest(BaseModel):
    data_folder: str = Field("data", description="Folder containing documents to process")
    clear_existing: bool = Field(False, description="Whether to clear existing data")
    chunk_size: int = Field(800, description="Text chunk size")
    chunk_overlap: int = Field(100, description="Chunk overlap size")

class IngestionResponse(BaseModel):
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(0, description="Number of documents processed")
    chunks_created: int = Field(0, description="Number of chunks created")
    entities_extracted: int = Field(0, description="Number of entities extracted")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup."""
    global rag_pipeline
    try:
        # Initialize with configuration from .env file
        provider = ModelProvider.OLLAMA if config.llm.default_provider.lower() == "ollama" else ModelProvider.OPENAI
        
        rag_pipeline = RAGPipeline(
            provider=provider,
            model_name=config.llm.default_model,
            temperature=config.llm.default_temperature,
            openai_api_key=config.llm.openai_api_key
        )
        logger.info(f"RAG pipeline initialized with {config.llm.default_provider} ({config.llm.default_model})")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Don't fail startup, allow manual configuration later
        rag_pipeline = None
        # Don't fail startup, allow manual configuration via API

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global rag_pipeline
    if rag_pipeline:
        rag_pipeline.close()

# Helper functions
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return session_id
    
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "created_at": datetime.now(),
        "messages": [],
        "last_activity": datetime.now()
    }
    return new_session_id

def add_message_to_session(session_id: str, role: str, content: str):
    """Add a message to the session history."""
    if session_id in sessions:
        sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        sessions[session_id]["last_activity"] = datetime.now()
        
        # Keep only last 20 messages per session
        if len(sessions[session_id]["messages"]) > 20:
            sessions[session_id]["messages"] = sessions[session_id]["messages"][-20:]

def format_chat_history(session_id: str) -> List[Dict]:
    """Format session messages for RAG pipeline."""
    if session_id not in sessions:
        return []
    
    history = []
    messages = sessions[session_id]["messages"]
    
    # Group messages into human-assistant pairs
    for i in range(0, len(messages), 2):
        if i < len(messages):
            human_msg = messages[i] if messages[i]["role"] == "human" else None
            assistant_msg = messages[i+1] if i+1 < len(messages) and messages[i+1]["role"] == "assistant" else None
            
            if human_msg and assistant_msg:
                history.append({
                    "human": human_msg["content"],
                    "assistant": assistant_msg["content"]
                })
    
    return history

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    global rag_pipeline
    
    try:
        # Get system statistics
        stats = {"total_documents": 0, "total_chunks": 0, "total_entities": 0}
        
        if rag_pipeline:
            try:
                retrieval_stats = rag_pipeline.get_retrieval_info("test")
                if "retrieval_stats" in retrieval_stats:
                    vector_stats = retrieval_stats["retrieval_stats"].get("vector_database", {})
                    graph_stats = retrieval_stats["retrieval_stats"].get("knowledge_graph", {})
                    
                    stats["total_documents"] = graph_stats.get("documents", 0)
                    stats["total_chunks"] = vector_stats.get("count", 0)
                    stats["total_entities"] = graph_stats.get("entities", 0)
            except Exception as e:
                logger.warning(f"Failed to get stats: {e}")
        
        return StatusResponse(
            status="healthy" if rag_pipeline else "not_configured",
            model_provider=rag_pipeline.provider.value if rag_pipeline else None,
            model_name=rag_pipeline.model_name if rag_pipeline else None,
            **stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return StatusResponse(
            status="error",
            model_provider=None,
            model_name=None
        ) # type: ignore

@app.post("/configure", response_model=Dict[str, str])
async def configure_model(config: ConfigRequest):
    """Configure the RAG model."""
    global rag_pipeline
    
    try:
        # Close existing pipeline
        if rag_pipeline:
            rag_pipeline.close()
        
        # Create new pipeline
        provider = ModelProvider.OLLAMA if config.provider.lower() == "ollama" else ModelProvider.OPENAI
        
        rag_pipeline = RAGPipeline(
            provider=provider,
            model_name=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key
        )
        
        return {"status": "success", "message": f"Configured {config.provider} with {config.model_name}"}
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system."""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not configured. Use /configure endpoint.")
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Add user message to session
        add_message_to_session(session_id, "human", request.message)
        
        # Get chat history if using context
        chat_history = format_chat_history(session_id) if request.use_context else None
        
        # Get response from RAG
        if request.use_context and chat_history:
            response = rag_pipeline.ask_with_context(request.message, chat_history)
        else:
            response = rag_pipeline.ask(request.message)
        
        # Add assistant response to session
        add_message_to_session(session_id, "assistant", response)
        
        # Get retrieval info for sources
        retrieval_info = rag_pipeline.get_retrieval_info(request.message)
        sources = retrieval_info.get("results", [])[:3]  # Top 3 sources
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the knowledge base."""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not configured. Use /configure endpoint.")
    
    try:
        # Get search results
        retrieval_info = rag_pipeline.get_retrieval_info(request.query)
        results = retrieval_info.get("results", [])
        
        # Limit results
        limited_results = results[:request.n_results]
        
        return SearchResponse(
            query=request.query,
            results=limited_results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/sessions", response_model=Dict[str, Any])
async def list_sessions():
    """List all chat sessions."""
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append({
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "last_activity": session_data["last_activity"],
            "message_count": len(session_data["messages"])
        })
    
    return {"sessions": session_list, "total": len(session_list)}

@app.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """Get a specific chat session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "session_data": sessions[session_id]
    }

@app.delete("/sessions/{session_id}", response_model=Dict[str, str])
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(background_tasks: BackgroundTasks, request: IngestionRequest):
    """Ingest documents into the system."""
    try:
        # Check if data folder exists
        data_path = Path(request.data_folder)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data folder '{request.data_folder}' not found")
        
        # Run ingestion in background
        def run_ingestion():
            try:
                pipeline = EnhancedIngestionPipeline(
                    data_folder=request.data_folder,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap
                )
                
                result = pipeline.run_complete_pipeline(
                    store_in_vector_db=True,
                    build_knowledge_graph=True,
                    clear_existing=request.clear_existing
                )
                
                # Log the result
                logger.info(f"Ingestion completed: {result}")
                
            except Exception as e:
                logger.error(f"Background ingestion failed: {e}")
        
        background_tasks.add_task(run_ingestion)
        
        return IngestionResponse(
            status="started",
            documents_processed=0,
            chunks_created=0,
            entities_extracted=0,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Ingestion request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file for processing."""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = data_dir / file.filename # type: ignore
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "message": f"File '{file.filename}' uploaded successfully", "path": str(file_path)}
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
