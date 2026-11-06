import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from .chunker import AppleKnowledgeChunker, DocumentChunk
from .embeddings import LocalSentenceEmbeddings
from .vector_store import FaissVectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .query_decomposer import QueryDecomposer
from .reranker import CrossEncoderReranker
from .conversational import ConversationContextManager
from .verifier import GroundednessVerifier
from .followup import SmartFollowUpGenerator
from .generator import GroundedGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "apple_knowledge_base.json")
INDEX_DIR = os.path.join(BASE_DIR, "data", "vector_cache")

class AppleRAGPipeline:
    """
    Tier-1 Enterprise RAG Pipeline for Apple Genius AI.
    
    Architecture Highlights:
      1. Two-Stage Retrieval: Hybrid Dense (FAISS) + Sparse (BM25) RRF Recall
         followed by Deep Transformer Cross-Encoder Re-Ranking.
      2. Multi-turn Conversational Coreference Resolution.
      3. Claim-Level Self-Correction Grounding Verifier (>99% target precision).
      4. Dynamic Contextual Follow-Up Suggestions.
      5. Multi-Intent Query Decomposition.
    """
    def __init__(
        self,
        retrieval_mode: str = "hybrid_rerank",
        generator_backend: str = "ollama",
        top_k: int = 4
    ):
        self.retrieval_mode = retrieval_mode
        self.generator_backend = generator_backend
        self.top_k = top_k

        # Core Components
        self.chunker = AppleKnowledgeChunker(max_chunk_size=600, overlap=100)
        self.embeddings = LocalSentenceEmbeddings()
        self.vector_store = FaissVectorStore(self.embeddings)
        self.bm25_retriever = BM25Retriever()

        # Advanced Modules
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_retriever)
        self.query_decomposer = QueryDecomposer(self.hybrid_retriever)
        self.reranker = CrossEncoderReranker()
        self.context_manager = ConversationContextManager()
        self.verifier = GroundednessVerifier()
        self.followup_generator = SmartFollowUpGenerator()

        # Generator
        self.generator = GroundedGenerator(backend=generator_backend)

        # Knowledge Base Initialization
        self.chunks: List[DocumentChunk] = []
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Knowledge base data file not found at {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.chunks = self.chunker.chunk_records(records)

        # Load or build FAISS index
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            try:
                self.vector_store.load(INDEX_DIR)
            except Exception:
                self.vector_store.build_from_chunks(self.chunks)
                self.vector_store.save(INDEX_DIR)
        else:
            self.vector_store.build_from_chunks(self.chunks)
            self.vector_store.save(INDEX_DIR)

        # Build BM25 index
        self.bm25_retriever.build_index(self.chunks)

    def query(self, raw_user_question: str) -> Dict[str, Any]:
        """
        Executes full end-to-end multi-stage RAG workflow.
        """
        t0 = time.time()

        # Step 1: Multi-turn Context & Pronoun Coreference Resolution
        resolved_question = self.context_manager.resolve_query(raw_user_question)

        subqueries = []
        candidates = []
        confidence_score = 0.99

        # Step 2: First-Stage Retrieval (Recall Layer)
        candidate_pool_size = max(self.top_k * 3, 12)

        if self.retrieval_mode == "baseline":
            raw_results = self.vector_store.similarity_search(resolved_question, top_k=candidate_pool_size)
            candidates = [
                (chunk, score, {"dense_score": score, "mode": "semantic"})
                for chunk, score in raw_results
            ]
            final_chunks = candidates[:self.top_k]

        elif self.retrieval_mode == "decomposed":
            subqueries, candidates = self.query_decomposer.retrieve_decomposed(
                resolved_question, top_k_per_subquery=max(2, self.top_k // 2)
            )
            # Re-rank decomposed candidates for maximum precision
            final_chunks = self.reranker.rerank(resolved_question, candidates, top_k=self.top_k)

        else: # Default: 'hybrid_rerank' (Two-Stage Top-Tier Mode)
            hybrid_candidates = self.hybrid_retriever.retrieve(resolved_question, top_k=candidate_pool_size)
            final_chunks = self.reranker.rerank(resolved_question, hybrid_candidates, top_k=self.top_k)

        retrieval_latency = time.time() - t0

        if final_chunks:
            confidence_score = final_chunks[0][1]
        else:
            confidence_score = 0.0

        # Step 3: Grounded Generation Layer
        t1 = time.time()
        gen_result = self.generator.generate(
            query=resolved_question,
            retrieved_chunks=final_chunks,
            confidence_score=confidence_score
        )
        generation_latency = time.time() - t1

        # Step 4: Claim-Level Self-Correction & Grounding Verification Loop
        verification = self.verifier.verify_answer(
            query=resolved_question,
            answer=gen_result["answer"],
            retrieved_chunks=final_chunks,
            is_fallback=gen_result["is_fallback"]
        )

        final_answer = verification["verified_answer"]
        grounded_accuracy_pct = f"{verification['grounding_score'] * 100:.1f}%"

        # Step 5: Dynamic Follow-Up Questions
        follow_ups = self.followup_generator.generate_suggestions(
            query=resolved_question,
            retrieved_chunks=final_chunks,
            is_fallback=gen_result["is_fallback"]
        )

        # Update Conversation Memory
        detected_entity = final_chunks[0][0].metadata.get("name") if final_chunks else None
        self.context_manager.add_turn("user", raw_user_question, detected_entity)
        self.context_manager.add_turn("assistant", final_answer, detected_entity)

        total_latency = time.time() - t0

        return {
            "query": raw_user_question,
            "resolved_query": resolved_question,
            "answer": final_answer,
            "is_fallback": gen_result["is_fallback"],
            "grounded_accuracy": grounded_accuracy_pct,
            "grounding_score": verification["grounding_score"],
            "retrieval_mode": self.retrieval_mode,
            "generator_backend": gen_result["backend_used"],
            "subqueries": subqueries,
            "retrieved_chunks": [
                {
                    "chunk_id": item[0].chunk_id,
                    "name": item[0].metadata.get("name"),
                    "category": item[0].metadata.get("category"),
                    "section": item[0].metadata.get("section"),
                    "price": item[0].metadata.get("price"),
                    "score": round(item[1], 4),
                    "confidence_pct": f"{min(99.9, item[1] * 100):.1f}%" if item[1] <= 1.0 else f"{min(99.9, item[1]):.1f}%",
                    "text": item[0].text,
                    "provenance": item[2] if len(item) > 2 else {}
                }
                for item in final_chunks
            ],
            "sources": gen_result["sources"],
            "follow_ups": follow_ups,
            "verification": verification,
            "latency": {
                "retrieval_ms": round(retrieval_latency * 1000, 1),
                "generation_ms": round(generation_latency * 1000, 1),
                "total_ms": round(total_latency * 1000, 1)
            }
        }
