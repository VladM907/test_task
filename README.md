# 🤖 Advanced RAG System with Knowledge Graphs

A production-ready Retrieval-Augmented Generation (RAG) system that combines vector similarity search with knowledge graph enhancement for intelligent document Q&A. Features hybrid retrieval, multiple LLM providers, and both REST API and web UI interfaces.

## ✨ Features

### 🧠 **Advanced RAG Architecture**
- **Hybrid Retrieval**: Combines ChromaDB vector search with Neo4j knowledge graph
- **Smart Context**: Entity-aware document chunking with relationship mapping
- **Multi-format Support**: PDF, TXT, MD, DOCX document ingestion
- **Conversation Memory**: Context-aware chat with session management

### 🔄 **Multiple LLM Providers**
- **Ollama**: Local models (llama3.1, mistral, etc.)
- **OpenAI**: Cloud models (GPT-4.1, etc.)
- **Runtime Switching**: Change models without restart
- **Configuration Management**: Environment-based setup

### 🌐 **Complete Interface Stack**
- **REST API**: Endpoints for programmatic access
- **Web UI**: Streamlit interface
- **Interactive Chat**: Real-time conversations
- **Admin Panel**: System monitoring and configuration

### 📊 **Production Features**
- **Session Management**: Conversation tracking
- **Health Monitoring**: System status and statistics
- **File Upload**: Direct document ingestion via UI/API
- **Comprehensive Logging**: Debug and monitoring capabilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   REST API      │    │   Documents     │
│                 │◄──►│   (FastAPI)     │◄──►│   (PDF/TXT/MD)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  RAG Pipeline   │
                    │   (LangChain)   │
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌─────────────────┐  ┌─────────────────┐
            │   ChromaDB      │  │     Neo4j       │
            │ (Vector Store)  │  │ (Knowledge Graph)│
            └─────────────────┘  └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- 8GB+ RAM recommended

### 1. **Clone & Setup**
```bash
git clone <repository>
cd test_task
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. **Environment Configuration**
```bash
# Copy and customize environment file
cp .env.example .env
nano .env  # Edit configuration
```

**Key Settings:**
```env
# LLM Provider (ollama or openai)
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4
OPENAI_API_KEY=your_key_here

# Database Settings
NEO4J_PASSWORD=your_password
DATA_FOLDER=./data
```

### 3. **Start Services**
```bash
# Start databases
docker-compose up -d

# Wait for Neo4j to be ready (30 seconds)
sleep 30
```

### 4. **Add Documents**
```bash
# Place your documents in the data folder
cp your_documents.pdf ./data/

# Or use the API/UI for upload
```

### 5. **Launch the System**
```bash
python start_api.py
# Access: http://localhost:8000/docs
```

```bash
python start_ui.py
# Access: http://localhost:8501
```

### 6. **First Use**
1. **Upload Documents**: Via UI or place in `./data/` folder
2. **Process Documents**: Click "Reprocess Documents" in UI
3. **Start Chatting**: Ask questions about your documents!

## 📁 Project Structure

```
test_task/
├── backend/                    # Core backend code
│   ├── api/                   # REST API (FastAPI)
│   │   └── main.py           # API endpoints
│   ├── ingestion/            # Document processing
│   │   ├── loader.py         # Document loaders
│   │   ├── splitter.py       # Text chunking
│   │   ├── embedding.py      # Vector embeddings
│   │   └── pipeline.py       # Full pipeline
│   ├── knowledge_graph/      # Neo4j integration
│   │   ├── neo4j_client.py   # Database client
│   │   └── graph_builder.py  # Entity extraction
│   ├── retrieval/            # Hybrid retrieval
│   │   ├── chromadb_store.py # Vector storage
│   │   └── hybrid_retriever.py # Combined search
│   ├── rag/                  # RAG implementation
│   │   └── pipeline.py       # LangChain RAG
│   └── config.py             # Configuration management
├── data/                     # Document storage
├── streamlit_app.py          # Web UI application
├── start_api.py              # API startup script
├── start_ui.py               # UI startup script
├── .env                      # Environment configuration
├── docker-compose.yml        # Database services
├── README.md                 # This file
└── requirements.txt      # Dependencies
```

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
CHROMA_PERSIST_DIR=./data/chroma_db

# LLM Configuration
DEFAULT_LLM_PROVIDER=openai          # ollama or openai
DEFAULT_LLM_MODEL=gpt-4              # Model name
OPENAI_API_KEY=sk-...                # Your OpenAI key
OLLAMA_BASE_URL=http://localhost:11434

# Document Processing
DATA_FOLDER=./data
CHUNK_SIZE=800
CHUNK_OVERLAP=100
ENABLE_ENTITY_EXTRACTION=true

# API Settings
API_PORT=8000
STREAMLIT_PORT=8501
CORS_ORIGINS=*
```

### Supported Models

**Ollama (Local):**
- llama3.1, llama3.2
- mistral, codellama
- Custom models

**OpenAI (Cloud):**
- gpt-4, gpt-4-turbo etc.

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status and statistics |
| POST | `/configure` | Change LLM provider/model |
| POST | `/chat` | Ask questions with context |
| POST | `/search` | Search documents |
| POST | `/upload` | Upload new documents |
| POST | `/ingest` | Process documents |
| GET | `/sessions` | List chat sessions |

## 🖥️ Web UI Guide

### Main Features

1. **💬 Chat Interface**
   - Natural language questions
   - Conversation history
   - Source citations
   - Context toggle

2. **🔍 Search Panel**
   - Document search
   - Similarity scores
   - Graph enhancement toggle

3. **⚙️ System Control**
   - Model configuration
   - File upload
   - Health monitoring
   - Session management

4. **📊 Analytics**
   - System metrics
   - Session statistics
   - Performance monitoring

## 🧪 Testing

### Validate Configuration
```bash
python validate_config.py
```

### Test API
```bash
python test_api_client.py
```

## 📈 Performance

### Optimization Tips
- **Embedding Model**: `all-MiniLM-L6-v2` for speed/quality balance
- **Graph Expansion**: Limit to 3 for response time
- **Database**: Use SSD storage for better performance

### Scaling Considerations
- **Vector Database**: ChromaDB scales to millions of documents
- **Knowledge Graph**: Neo4j handles complex relationships efficiently
- **API**: FastAPI supports async for high concurrency
- **Caching**: Embedding cache reduces reprocessing time

## 🐛 Troubleshooting

### Common Issues

**1. Neo4j Connection Failed**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart services
docker compose restart neo4j
```

**2. OpenAI API Key Issues**
```bash
# Verify key in environment
echo $OPENAI_API_KEY

# Check .env file
cat .env | grep OPENAI_API_KEY
```

**3. Document Processing Errors**
```bash

# Validate configuration
python validate_config.py
```

**4. Memory Issues**
```bash
# Monitor memory usage
docker stats

# Reduce chunk size if needed
CHUNK_SIZE=400
```

## 🙏 Acknowledgments

- **LangChain**: RAG framework and LLM integration
- **ChromaDB**: Vector database for similarity search
- **Neo4j**: Graph database for knowledge relationships
- **Streamlit**: Web UI framework
- **FastAPI**: High-performance API framework
- **Sentence Transformers**: Document embedding models

---

## 🎯 Demo Usage

### Example Questions to Try

**About the Constitution (if using provided document):**
- "What are the main principles of the Constitution?"
- "How many amendments are there?"
- "What is the Bill of Rights?"
- "Who has the power to declare war?"

**Follow-up Context Questions:**
- "Can you explain that in more detail?" 
- "What was my last question about?"
- "How does this relate to what we discussed before?"

### Expected Performance
- **Response Time**: 2-5 seconds per query
- **Context**: Maintains conversation history
- **Sources**: Shows relevant document sections

Ready to explore your documents with AI! 🚀
