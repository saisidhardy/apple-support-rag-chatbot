import re
from typing import List, Dict, Any, Tuple
from .chunker import DocumentChunk
from .hybrid_retriever import HybridRetriever

class QueryDecomposer:
    """
    Query Decomposition RAG (Enhancement Requirement).
    Breaks complex, comparative, or multi-faceted questions into
    focused sub-queries, retrieves targeted evidence for each, and
    assembles a unified context packet.
    
    Example:
      User: "Compare iPhone 16 and 16 Pro cameras, and what is the return policy?"
      Sub-query 1: "iPhone 16 camera specifications"
      Sub-query 2: "iPhone 16 Pro camera specifications"
      Sub-query 3: "Apple return and refund policy 14 days"
    """
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever

    def decompose(self, query: str) -> List[str]:
        q_lower = query.lower()

        # Check for comparative queries: "X vs Y" or "compare X and Y"
        compare_match = re.search(
            r'(?:compare|difference between)\s+([a-zA-Z0-9\s]+?)\s+(?:and|with|to)\s+([a-zA-Z0-9\s\?]+)',
            query,
            re.IGNORECASE
        )
        if compare_match:
            entity_a = compare_match.group(1).strip()
            entity_b = compare_match.group(2).strip().rstrip("?")
            return [
                f"{entity_a} specifications features overview",
                f"{entity_b} specifications features overview"
            ]

        vs_match = re.search(r'([a-zA-Z0-9\s]+?)\s+(?:vs\.?|versus)\s+([a-zA-Z0-9\s\?]+)', query, re.IGNORECASE)
        if vs_match:
            entity_a = vs_match.group(1).strip()
            entity_b = vs_match.group(2).strip().rstrip("?")
            return [
                f"{entity_a} specifications features",
                f"{entity_b} specifications features"
            ]

        # Check for multi-clause questions connected by "and also", "as well as", ";", ", and", or multiple '?'
        parts = re.split(r'\?|\band also\b|\bas well as\b|;|\bwhat about\b|,\s*and\b|\band\s+what\b|\band\s+how\b', query, flags=re.IGNORECASE)
        cleaned_parts = [p.strip() for p in parts if len(p.strip()) > 8]

        if len(cleaned_parts) > 1:
            return cleaned_parts[:3]

        # Single focused query
        return [query]

    def retrieve_decomposed(self, query: str, top_k_per_subquery: int = 2) -> Tuple[List[str], List[Tuple[DocumentChunk, float, Dict[str, Any]]]]:
        subqueries = self.decompose(query)
        seen_chunk_ids = set()
        aggregated_results = []

        for sq in subqueries:
            results = self.retriever.retrieve(sq, top_k=top_k_per_subquery)
            for chunk, score, prov in results:
                if chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.chunk_id)
                    prov["subquery_origin"] = sq
                    aggregated_results.append((chunk, score, prov))

        # If only 1 subquery or very few chunks retrieved, ensure top_k baseline is met
        if len(aggregated_results) < 3:
            baseline = self.retriever.retrieve(query, top_k=4)
            for chunk, score, prov in baseline:
                if chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.chunk_id)
                    prov["subquery_origin"] = "primary_query"
                    aggregated_results.append((chunk, score, prov))

        return subqueries, aggregated_results
