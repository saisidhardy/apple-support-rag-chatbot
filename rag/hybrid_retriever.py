from typing import List, Tuple, Dict, Any
from .chunker import DocumentChunk
from .vector_store import FaissVectorStore
from .bm25_retriever import BM25Retriever

class HybridRetriever:
    """
    Hybrid RAG Retriever combining Dense Semantic Search (FAISS)
    with Sparse Lexical Search (BM25) via Reciprocal Rank Fusion (RRF).
    
    RRF Formula:
        RRF_Score(doc) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
        where k = 60 (standard IR smoothing parameter)
    """
    def __init__(
        self,
        vector_store: FaissVectorStore,
        bm25_retriever: BM25Retriever,
        k_constant: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.k_constant = k_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float, Dict[str, Any]]]:
        # Fetch candidate pools from both retrievers
        candidate_k = max(top_k * 3, 10)
        dense_results = self.vector_store.similarity_search(query, top_k=candidate_k)
        sparse_results = self.bm25_retriever.search(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        provenance_map: Dict[str, Dict[str, Any]] = {}

        # Process dense rankings
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_contrib = self.dense_weight / (self.k_constant + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_contrib
            provenance_map[cid] = {
                "dense_rank": rank,
                "dense_score": score,
                "sparse_rank": None,
                "sparse_score": None
            }

        # Process sparse rankings
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_contrib = self.sparse_weight / (self.k_constant + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_contrib
            if cid in provenance_map:
                provenance_map[cid]["sparse_rank"] = rank
                provenance_map[cid]["sparse_score"] = score
            else:
                provenance_map[cid] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": score
                }

        # Sort combined results descending by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, fused_score in sorted_chunks:
            chunk = chunk_map[cid]
            prov = provenance_map[cid]
            results.append((chunk, fused_score, prov))

        return results
