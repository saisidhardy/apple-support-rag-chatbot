# Failure Case Analysis & Engineering Mitigations

As required by **Section 6 of the Technical Evaluation Rubric**, this document presents an in-depth post-mortem of failure modes encountered during baseline testing, along with the precise algorithmic and structural mitigations developed to solve them.

---

## Failure Case 1: Semantic Monopoly in Comparative Queries

### 1. Description & Failure Symptom
* **Test Query:** *"Compare the display refresh rate, optical zoom, and materials between iPhone 16 and iPhone 16 Pro."*
* **Observed Failure in Baseline Semantic RAG:**
  - In pure dense vector retrieval (FAISS), the top-4 retrieved chunks were all related to the **iPhone 16 Pro** (`APP-IPHONE-16P_overview`, `APP-IPHONE-16P_specs`, `APP-IPHONE-16PM_specs`, `APP-IPHONE-16P_faqs`).
  - The standard **iPhone 16** (`APP-IPHONE-16`) chunks received slightly lower similarity scores and were completely excluded from the top-4 context window.
  - As a result, the LLM hallucinated the baseline iPhone 16 specs or produced an incomplete comparison stating: *"The iPhone 16 Pro has a 120Hz display, but details for iPhone 16 were not provided in the context."*

### 2. Root Cause Analysis
* Both devices share dense semantic clusters (both are iPhones released in the same cycle with overlapping terminology like 'Camera Control', 'Titanium', 'A18', 'OLED').
* Because the iPhone 16 Pro document had greater keyword length and technical depth in its specification chunk, its cosine angle to the query was higher than the standard iPhone 16 document.
* Single-pass top-k retrieval suffers from **majority class monopolization** when two entities are compared.

### 3. Engineering Mitigation Implemented
1. **Query Decomposition Architecture:**
   - Implemented `QueryDecomposer` in `rag/query_decomposer.py`.
   - The regex-based semantic splitter automatically parses the comparison pattern:
     - Sub-query 1: `"iPhone 16 specifications features overview"`
     - Sub-query 2: `"iPhone 16 Pro specifications features overview"`
   - Retrieves top-2 candidates for *each* sub-query independently, ensuring balanced representation.
2. **Hybrid Lexical Weighting:**
   - BM25 assigns high inverse document frequency (IDF) to the exact token `"16"` vs `"Pro"`, preventing one model from obscuring the other.

---

## Failure Case 2: Numerical & Currency Deductible Confusion

### 1. Description & Failure Symptom
* **Test Query:** *"What is the accidental damage fee to repair an iPhone screen under AppleCare+?"*
* **Observed Failure in Baseline Semantic RAG:**
  - Dense embeddings mapped the query strongly to general warranty terms (`APP-POL-WARRANTY-1YR_overview`), which discuss repairs but state that screen drops are excluded.
  - When AppleCare+ was retrieved, the generator occasionally confused the **$29 screen repair fee** with the **$99 other accidental damage fee** because both figures co-occurred in the same broad paragraph.

### 2. Root Cause Analysis
* Small integers, currency symbols (`$29`, `$99`), and numerical limits do not carry high positional variance in general-purpose transformer embeddings (`all-MiniLM-L6-v2`), which prioritize syntactic and contextual meaning over exact numerical constants.
* Coarse document chunking merged all deductible tiers into a single monolithic block.

### 3. Engineering Mitigation Implemented
1. **Granular Specification Chunker:**
   - Isolated fees into distinct metadata fields and structured FAQ question-answer pairs:
     - `Q: What is the fee to repair an iPhone screen with AppleCare+? A: $29 for front screen or back glass repair.`
2. **Lexical BM25 Preservation of Token Modifiers:**
   - Preserved tokens with `$` prefixes and hyphenated digits in `rag/bm25_retriever.py` tokenizer (`r'[a-zA-Z0-9\-\$]+'`).
   - BM25 gives high score weighting to `"screen"` + `"$29"` + `"deductible"`, elevating the exact fee chunk to rank 1.

---

## Failure Case 3: Out-of-Scope Competitor Hallucination

### 1. Description & Failure Symptom
* **Test Query:** *"What is the camera resolution and battery capacity of the Samsung Galaxy S24 Ultra?"*
* **Observed Failure in Baseline RAG:**
  - With no out-of-scope guardrail, the nearest cosine neighbors retrieved high-end camera phone chunks from Apple (e.g. `iPhone 16 Pro Max 48MP camera`).
  - Without strict prompt grounding, generic LLMs would blend the retrieved iPhone camera text with their pre-trained general knowledge about Samsung, producing an ungrounded hallucination.

### 2. Root Cause Analysis
* Nearest-neighbor vector search *always* returns top-k documents even if the query is fundamentally foreign to the corpus.
* Without a hard boundary or confidence gate, the LLM treats irrelevant documents as authoritative context.

### 3. Engineering Mitigation Implemented
1. **Explicit Negative Entity Guardrails:**
   - Implemented regex out-of-scope detector in `rag/generator.py` for competitor hardware brands (`samsung`, `galaxy`, `dell`, `hp`, `windows`, `playstation`).
2. **Confidence Score Thresholding:**
   - If the top fused score falls below the required threshold, generation is aborted.
3. **Strict Grounding Refusal Prompt:**
   - When triggered, outputs the verified refusal message:
     *"I do not have sufficient information in the official Apple knowledge dataset to answer this question. My knowledge is strictly grounded in official Apple hardware, technical specifications, AppleCare+, warranty policies, and troubleshooting procedures."*
