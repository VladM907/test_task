"""
RAG (Retrieval-Augmented Generation) module.
Integrates LangChain with Ollama/OpenAI and the hybrid retrieval system.
"""

from .pipeline import RAGPipeline, ModelProvider
from .config import RAGConfig

__all__ = ['RAGPipeline', 'ModelProvider', 'RAGConfig']
