import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from .chunker import DocumentChunk

class BM25Retriever:
    """
    Keyword-based BM25 retriever for exact term, model code,
    and specification lookups (e.g., 'A18 Pro', '120Hz', 'IP68', '$29').
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric, preserving numbers and hyphens
        tokens = re.findall(r'[a-zA-Z0-9\-\$]+', text.lower())
        return tokens

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        corpus = [
            self._tokenize(
                f"{c.text} {c.metadata.get('name', '')} {c.metadata.get('keywords', '')}"
            ) for c in chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))
        return results
