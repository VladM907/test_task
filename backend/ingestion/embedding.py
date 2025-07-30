"""
Embedding generation using sentence-transformers (all-MiniLM-L6-v2) with improved error handling.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model with error handling."""
        self.model_name = model_name
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def embed(self, texts: List[str]) -> List[list]:
        """
        Generate embeddings for a list of texts with error handling.
        Returns list of embedding vectors (as lists).
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        # Filter out empty or None texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping invalid text at index {i}: {type(text)}")
        
        if not valid_texts:
            logger.warning("No valid texts found for embedding")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            embeddings = self.model.encode(
                valid_texts, 
                show_progress_bar=True,
                convert_to_tensor=False,  # Return as numpy arrays
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            # Convert numpy arrays to lists
            embedding_lists = []
            for embedding in embeddings:
                if hasattr(embedding, 'tolist'):
                    embedding_lists.append(embedding.tolist())
                else:
                    embedding_lists.append(list(embedding))
            
            # Create result list with proper indexing
            result_embeddings: List[list] = []
            embedding_dim = len(embedding_lists[0]) if embedding_lists else 384  # Default dimension
            
            # Initialize with zero vectors
            for i in range(len(texts)):
                result_embeddings.append([0.0] * embedding_dim)
            
            # Fill in actual embeddings for valid texts
            for i, valid_idx in enumerate(valid_indices):
                result_embeddings[valid_idx] = embedding_lists[i]
            
            # Log warnings for texts that got zero vectors
            for i in range(len(texts)):
                if i not in valid_indices:
                    logger.warning(f"Using zero vector for invalid text at index {i}")
            
            logger.info(f"Successfully generated {len(embedding_lists)} embeddings")
            return result_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
            return [[0.0] * embedding_dim for _ in texts]
    
    def embed_single(self, text: str) -> Optional[list]:
        """
        Generate embedding for a single text.
        Returns embedding vector as list or None if failed.
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Invalid text provided for single embedding")
            return None
        
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        try:
            test_embedding = self.embed(["test"])
            return len(test_embedding[0]) if test_embedding and test_embedding[0] else 384
        except Exception:
            return 384  # Default dimension for all-MiniLM-L6-v2
