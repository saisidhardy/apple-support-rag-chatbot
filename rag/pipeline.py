import os
import json
import time
from typing import List, Dict, Any, Optional

from .chunker import AppleKnowledgeChunker, DocumentChunk
from .embeddings import LocalSentenceEmbeddings
from .vector_store import FaissVectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .query_decomposer import QueryDecomposer
from .generator import GroundedGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "apple_knowledge_base.json")
INDEX_DIR = os.path.join(BASE_DIR, "data", "vector_cache")

class AppleRAGPipeline:
    """
    Unified Production RAG Pipeline for Apple Genius AI.
    
    Supports 3 retrieval modalities:
      1. 'baseline'    -> FAISS Dense Semantic Search
      2. 'hybrid'      -> FAISS + BM25 with Reciprocal Rank Fusion (RRF)
      3. 'decomposed'  -> Multi-intent Query Decomposition + Hybrid RRF
      
    And 4 generation backends:
      1. 'ollama'      -> Local Ollama (qwen3.5:4b)
      2. 'gemini'      -> Google Gemini API
      3. 'openai'      -> OpenAI API
      4. 'extractive'  -> Offline Grounded Synthesizer
    """
    def __init__(
        self,
        retrieval_mode: str = "hybrid",
        generator_backend: str = "ollama",
        top_k: int = 4
    ):
        self.retrieval_mode = retrieval_mode
        self.generator_backend = generator_backend
        self.top_k = top_k

        # 1. Initialize Chunker & Embedding Model
        self.chunker = AppleKnowledgeChunker(max_chunk_size=600, overlap=100)
        self.embeddings = LocalSentenceEmbeddings()

        # 2. Initialize Vector Store & BM25 Retriever
        self.vector_store = FaissVectorStore(self.embeddings)
        self.bm25_retriever = BM25Retriever()

        # 3. Initialize Advanced Enhancements
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_retriever)
        self.query_decomposer = QueryDecomposer(self.hybrid_retriever)

        # 4. Initialize Generator
        self.generator = GroundedGenerator(backend=generator_backend)

        # 5. Build or load knowledge index
        self.chunks: List[DocumentChunk] = []
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Knowledge base data file not found at {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.chunks = self.chunker.chunk_records(records)

        # Build or load FAISS Index
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            try:
                self.vector_store.load(INDEX_DIR)
            except Exception:
                self.vector_store.build_from_chunks(self.chunks)
                self.vector_store.save(INDEX_DIR)
        else:
            self.vector_store.build_from_chunks(self.chunks)
            self.vector_store.save(INDEX_DIR)

        # Build BM25 Index
        self.bm25_retriever.build_index(self.chunks)

    def query(self, user_question: str) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query pipeline.
        Returns detailed execution trace including retrieved chunks,
        provenance scores, subqueries (if any), answer, and latency.
        """
        t0 = time.time()
        retrieved_data = []
        subqueries = []
        confidence_score = 1.0

        # Retrieval phase
        if self.retrieval_mode == "baseline":
            raw_results = self.vector_store.similarity_search(user_question, top_k=self.top_k)
            retrieved_data = [
                (chunk, score, {"dense_score": score, "mode": "semantic"})
                for chunk, score in raw_results
            ]
            if retrieved_data:
                confidence_score = retrieved_data[0][1] # Cosine similarity of top hit
            else:
                confidence_score = 0.0

        elif self.retrieval_mode == "decomposed":
            subqueries, retrieved_data = self.query_decomposer.retrieve_decomposed(
                user_question, top_k_per_subquery=max(2, self.top_k // 2)
            )
            if retrieved_data:
                confidence_score = retrieved_data[0][1]
            else:
                confidence_score = 0.0

        else: # Default: 'hybrid'
            retrieved_data = self.hybrid_retriever.retrieve(user_question, top_k=self.top_k)
            if retrieved_data:
                confidence_score = retrieved_data[0][1]
            else:
                confidence_score = 0.0

        retrieval_latency = time.time() - t0

        # Generation phase
        t1 = time.time()
        gen_result = self.generator.generate(
            query=user_question,
            retrieved_chunks=retrieved_data,
            confidence_score=confidence_score
        )
        generation_latency = time.time() - t1
        total_latency = time.time() - t0

        return {
            "query": user_question,
            "answer": gen_result["answer"],
            "is_fallback": gen_result["is_fallback"],
            "confidence_score": confidence_score,
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
                    "text": item[0].text,
                    "provenance": item[2] if len(item) > 2 else {}
                }
                for item in retrieved_data
            ],
            "sources": gen_result["sources"],
            "latency": {
                "retrieval_ms": round(retrieval_latency * 1000, 1),
                "generation_ms": round(generation_latency * 1000, 1),
                "total_ms": round(total_latency * 1000, 1)
            }
        }
