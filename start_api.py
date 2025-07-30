#!/usr/bin/env python3
"""
Startup script for the RAG System API.
"""
import sys
import os
import uvicorn

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
