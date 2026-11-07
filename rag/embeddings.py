import os
from typing import List
import numpy as np

class LocalSentenceEmbeddings:
    """
    Local embedding model powered by sentence-transformers.
    Default model: 'all-MiniLM-L6-v2' (384-dimensional vector space,
    fast inference on CPU/GPU, high semantic similarity accuracy).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Try offline local cache first to avoid network timeouts / DNS issues
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        embedding = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
