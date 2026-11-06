# 5-Minute Live Presentation Playbook & Script
**Evaluation Track: Technical Implementation • Strict 5-Minute Slot**

This playbook provides an exact, rehearsed minute-by-minute script and role distribution for a 4–5 member student group presenting the **Apple Genius AI RAG Chatbot**.

---

## ⏱️ Presentation Timing Matrix (Total: 5:00)

| Minute | Segment | Speaker | Objective |
| :---: | :--- | :---: | :--- |
| **0:00 – 1:00** | **Problem Statement & Dataset** | Speaker 1 | Introduce Apple Inc., customer pain points, dataset quality, and why RAG is required over raw LLMs. |
| **1:00 – 2:15** | **Architecture & Enhancements Demo** | Speaker 2 | Walk through the Streamlit UI, pipeline architecture (Chunking -> FAISS -> BM25 -> RRF -> Decomposition). |
| **2:15 – 3:45** | **Live Chatbot Demo (3 Rubric Questions)** | Speaker 3 | Execute live queries: Factual spec, Comparison, and Out-of-Scope Fallback. Show source inspector. |
| **3:45 – 5:00** | **Evaluation, Failure Analysis & Q&A** | Speaker 4 & 5 | Present benchmark metrics, failure cases resolved, and answer panel questions. |

---

## 🎤 Detailed Rehearsal Script

### Minute 0:00 – 1:00: Problem Statement & Knowledge Base
**Speaker 1:**
> *"Good morning, professors and classmates. Today our team presents **Apple Genius AI**, a production-grade RAG chatbot designed for the consumer electronics domain, specifically the Apple product and support ecosystem.*
>
> *Customers face significant friction when trying to compare subtle device specifications, verify AppleCare+ repair deductibles, understand the 14-day return window, or perform emergency liquid first aid. If a customer asks a standard LLM, it often hallucinates outdated pricing or invents unsupported warranty promises.*
>
> *To solve this, we engineered an end-to-end RAG system grounded in a curated knowledge base of **64 comprehensive records** across 8 product and policy categories—from iPhone 16 Pro to M4 MacBooks and official AppleCare+ deductible terms. Every answer is strictly grounded in verifiable data."*

---

### Minute 1:00 – 2:15: Architecture & Enhancements
**Speaker 2:**
> *"Turning to our technical architecture, we went beyond a basic baseline RAG by implementing **both optional enhancements** from the rubric:*
>
> 1. *First, **Hierarchical Semantic Chunking**: Rather than arbitrary token slicing that cuts prices in half, we created 256 self-contained semantic units for Overviews, Specifications, FAQs, and Policy Clauses.*
> 2. *Second, **Hybrid RAG with Reciprocal Rank Fusion (RRF)**: Combining 384-dimensional dense vectors in a Meta FAISS index with BM25 lexical keyword retrieval. This allows exact numbers like '$29' and '120Hz' to be matched alongside conceptual queries.*
> 3. *Third, **Query Decomposition RAG**: When a customer asks a complex comparative or multi-part question, our system parses it into targeted sub-queries, executes parallel retrieval, and merges the evidence before generation.*
> 4. *For generation, our app runs **100% locally with zero API cost** using Ollama's Qwen-3.5 4B model, while also supporting Gemini and an instant offline extractive synthesizer."*

---

### Minute 2:15 – 3:45: Live Chatbot Demonstration
**Speaker 3 (Screen Sharing Streamlit UI):**
> *"Let's see the chatbot in action.*
>
> * **Demo 1 (Factual Lookup):** We click the 1-click button: *'What is the battery life, charging port, and baseline memory of MacBook Air 13-inch M3?'*
>   - In under 50 milliseconds, our Hybrid RAG retrieves the official specification chunk.
>   - As you can see, the chatbot reports exactly: up to 18 hours battery life, MagSafe 3 charging, and the refreshed 16GB unified memory standard.
>   - Opening our **Retrieved Context Inspector**, you can see the exact similarity scores and BM25 provenance.
>
> * **Demo 2 (Comparative Query):** *'Compare iPhone 16 and iPhone 16 Pro camera specs and display refresh rate.'*
>   - Under Query Decomposition mode, notice that the system broke this into two sub-queries. It retrieved the 60Hz display and 2x zoom for iPhone 16, and the 120Hz ProMotion and 5x tetraprism optical zoom for iPhone 16 Pro.
>
> * **Demo 3 (Out-of-Scope Fallback):** Now let's test a question outside our dataset: *'What is the battery capacity of the Samsung Galaxy S24 Ultra?'*
>   - Instead of hallucinating, our guardrail triggers an immediate fallback: *'I do not have sufficient information in the official Apple knowledge dataset...'* as highlighted by the red Fallback badge."*

---

### Minute 3:45 – 5:00: Evaluation, Failure Analysis & Q&A
**Speaker 4:**
> *"To ensure rigorous engineering standards, we built an automated benchmark suite evaluating 6 test cases across 3 modalities. 
>
> * In Baseline Semantic RAG, multi-part comparison had a failure mode where iPhone 16 Pro dominated the top-4 chunks, pushing out iPhone 16.
> * Under our **Decomposed Hybrid RAG**, retrieval hit rate reached **100%**, keyword coverage reached **100%**, and out-of-scope fallback precision was **100%**, with an average retrieval latency of just **20 milliseconds**.*
>
> *All code, datasets, evaluation reports, and this Streamlit UI are committed and pushed to our public GitHub repository. Thank you, and we are now ready for your questions."*

---

## 🎯 Anticipated Q&A Answers (For the Evaluation Panel)

* **Q1: Why did you pick Reciprocal Rank Fusion (RRF) instead of simple linear score addition?**
  * *Answer:* Dense cosine similarity (ranging between 0.0 to 1.0) and BM25 scores (unbounded positive floats) have fundamentally different distributions. Linear combination requires arbitrary normalization heuristics. RRF uses ranks ($1 / (k + \text{rank})$) rather than raw scores, making it scale-invariant, robust, and industry standard.
* **Q2: How does the system handle queries where only half the question is supported?**
  * *Answer:* Query decomposition breaks the prompt into distinct sub-queries. The supported sub-query retrieves verified context, while the unsupported sub-query yields a low similarity score, allowing the generator to answer the supported portion while explicitly disclaiming the unsupported aspect.
* **Q3: What embedding model did you choose and why?**
  * *Answer:* We chose `sentence-transformers/all-MiniLM-L6-v2`. It maps sentences into a 384-dimensional dense vector space, runs in under 15ms on CPU, and is trained specifically on sentence-level semantic similarity without needing high-memory GPU servers.
