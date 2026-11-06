import os
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

from .chunker import DocumentChunk
from .embeddings import LocalSentenceEmbeddings

class FaissVectorStore:
    """
    Production-grade Vector Store using Meta FAISS.
    Operates in cosine-similarity mode via normalized Inner Product (IndexFlatIP).
    """
    def __init__(self, embedding_model: LocalSentenceEmbeddings):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384

    def build_from_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.dimension = embeddings.shape[1]

        # Inner Product on normalized vectors is exact Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or len(self.chunks) == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
            chunks_data = [c.to_dict() for c in self.chunks]
            with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    def load(self, directory: str) -> bool:
        index_file = os.path.join(directory, "index.faiss")
        chunks_file = os.path.join(directory, "chunks.json")
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            self.index = faiss.read_index(index_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [
                DocumentChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    metadata=c["metadata"]
                ) for c in chunks_data
            ]
            return True
        return False
