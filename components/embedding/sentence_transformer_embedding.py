# /Users/murseltasgin/projects/chat_rag/components/embedding/sentence_transformer_embedding.py
"""
Sentence Transformer embedding implementation
"""
import os
from typing import List, Union
import numpy as np

# Set OpenMP environment variables BEFORE importing SentenceTransformer
# This prevents OMP errors when multiple instances are created
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding
from core.exceptions import EmbeddingException


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embedding implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer model
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        
        try:
            # Use CPU device and disable multiprocessing to avoid OMP conflicts
            # Set these environment variables here as a fallback (should be set in app.py already)
            self.model = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            raise EmbeddingException(f"Failed to load model {model_name}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        prefix: str = None,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) to embedding(s)"""
        try:
            # Disable multiprocessing and use single-threaded encoding to avoid OMP conflicts
            if prefix and "e5" in self.model_name.lower():
                texts = [prefix + t for t in texts] if isinstance(texts, list) else prefix + texts
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=False,
                normalize_embeddings=False,
                **kwargs
            )
            return embeddings
        except Exception as e:
            raise EmbeddingException(f"Encoding failed: {e}")
    
    def get_name(self) -> str:
        """Get the embedding model name"""
        return f"SentenceTransformer-{self.model_name}"
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

