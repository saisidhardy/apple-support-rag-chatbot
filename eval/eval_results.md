# RAG Chatbot Evaluation & Verification Report

## 1. Executive Summary
This evaluation benchmarks the **Apple Genius Support & Product Discovery RAG Chatbot** across multiple retrieval modalities: Baseline Dense Semantic Search, Hybrid RAG with Reciprocal Rank Fusion (RRF), and Multi-intent Query Decomposition RAG.

| Evaluation Metric | Baseline RAG | Hybrid RAG (BM25 + Dense RRF) | Query Decomposition RAG |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate (Supported Queries)** | 75.0% | 75.0% | 100.0% |
| **Fallback Precision (Unsupported Queries)** | 100.0% | 100.0% | 100.0% |
| **Average End-to-End Latency** | 2046.6 ms | 825.4 ms | 817.9 ms |

## 2. Test Cases and Grounding Verification
| Test ID | Category | Question | Hit | Keyword Cov | Fallback Verified |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **TEST-01-FACTUAL** | Factual Lookup | What is the battery life, charging port, and ... | PASS (Yes) | 100% | Supported |
| **TEST-02-COMPARISON** | Comparison Query | Compare the display refresh rate, optical zoo... | PASS (Yes) | 83% | Supported |
| **TEST-03-MULTIPART-POLICY** | Multi-Part Policy Query | What is Apple standard return window for open... | FAIL (No) | 75% | Supported |
| **TEST-04-TROUBLESHOOTING** | Hardware Troubleshooting | How do I force restart a frozen iPhone 16, an... | PASS (Yes) | 25% | Supported |
| **TEST-05-OUT-OF-SCOPE-COMPETITOR** | Out-of-Scope Fallback | What are the camera specifications and batter... | PASS (Yes) | 100% | PASS (Fallback) |
| **TEST-06-OUT-OF-SCOPE-POLICY** | Unsupported Policy Query | Does Apple offer free lifetime warranty and r... | PASS (Yes) | 100% | Grounded Refusal |

## 3. Failure Case Analysis & Engineering Mitigation

### Failure Case 1: Complex Multi-Entity Comparison Under Pure Semantic Search
* **Symptom:** In pure Baseline Dense Semantic retrieval, when querying: *'Compare the display refresh rate, optical zoom, and materials between iPhone 16 and iPhone 16 Pro'*, the vector space similarity was dominated by the iPhone 16 Pro chunks, causing the standard iPhone 16 specifications to be ranked outside the top-k cutoff.
* **Root Cause:** Dense embedding representations of closely related product names can cluster tightly, allowing the first entity with higher lexical density to monopolize nearest neighbors.
* **Mitigation Implemented:** 
  1. **Query Decomposition RAG:** The query decomposer isolates 'iPhone 16 specifications' and 'iPhone 16 Pro specifications' into discrete sub-queries, retrieving targeted context for both models before synthesizing the answer.
  2. **Hybrid RRF Ranking:** BM25 lexical indexing assigns strong inverse document frequency weights to distinct model numbers ('16' vs '16 Pro'), ensuring balanced retrieval.

---

### Failure Case 2: Exact Currency and Deductible Value Mismatches
* **Symptom:** Dense embeddings occasionally treat numerical figures (e.g., '$29 screen deductible' vs '$99 accidental damage deductible') with identical semantic valence, occasionally retrieving general terms rather than exact dollar amounts.
* **Root Cause:** Cosine angle of small numbers and currency symbols in pre-trained transformer embeddings exhibit high cosine proximity.
* **Mitigation Implemented:**
  1. **BM25 Lexical Tokenization:** Preserved dollar signs and numerical tokens (`$29`, `$99`, `14-day`) as explicit lexical index terms.
  2. **Reciprocal Rank Fusion:** Fusion ensures chunks possessing both semantic relevance and exact numerical tokens rise to the top 1 position.
