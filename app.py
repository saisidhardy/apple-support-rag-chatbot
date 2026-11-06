import os
import sys
import streamlit as st
import time

# Ensure root directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline import AppleRAGPipeline

st.set_page_config(
    page_title="Apple Genius AI | Support & Product Discovery RAG",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #fbfbfd;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .apple-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-semantic { background-color: #e0f2fe; color: #0369a1; }
    .badge-hybrid { background-color: #fef3c7; color: #b45309; }
    .badge-decomposed { background-color: #ede9fe; color: #6d28d9; }
    .badge-fallback { background-color: #fee2e2; color: #b91c1c; }
    .badge-source { background-color: #f1f5f9; color: #334155; }
</style>
""", unsafe_allow_html=True)

# Cache pipeline instance based on parameters
@st.cache_resource(show_spinner=False)
def get_rag_pipeline(retrieval_mode: str, generator_backend: str, top_k: int):
    return AppleRAGPipeline(
        retrieval_mode=retrieval_mode,
        generator_backend=generator_backend,
        top_k=top_k
    )

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=42)
    st.title("Apple Genius RAG")
    st.caption("Technical Implementation Track • Production RAG")

    st.markdown("---")
    st.subheader("⚙️ Pipeline Configuration")

    retrieval_mode_label = st.selectbox(
        "Retrieval Technique:",
        [
            "⚡ Hybrid RAG (BM25 + FAISS RRF)",
            "🔍 Query Decomposition RAG (Multi-intent)",
            "🎯 Baseline Semantic RAG (Dense FAISS)"
        ],
        index=0,
        help="Select between baseline vector search, hybrid lexical+dense fusion, or query decomposition."
    )

    mode_map = {
        "⚡ Hybrid RAG (BM25 + FAISS RRF)": "hybrid",
        "🔍 Query Decomposition RAG (Multi-intent)": "decomposed",
        "🎯 Baseline Semantic RAG (Dense FAISS)": "baseline"
    }
    selected_mode = mode_map[retrieval_mode_label]

    backend_label = st.selectbox(
        "LLM Generator Backend:",
        [
            "🤖 Local Ollama (qwen3.5:4b)",
            "⚡ Local Grounded Synthesizer (Fast Extractive)",
            "☁️ Google Gemini API",
            "☁️ OpenAI API"
        ],
        index=0
    )

    backend_map = {
        "🤖 Local Ollama (qwen3.5:4b)": "ollama",
        "⚡ Local Grounded Synthesizer (Fast Extractive)": "extractive",
        "☁️ Google Gemini API": "gemini",
        "☁️ OpenAI API": "openai"
    }
    selected_backend = backend_map[backend_label]

    top_k = st.slider("Top-K Retrieved Context Chunks:", min_value=2, max_value=8, value=4, step=1)

    st.markdown("---")
    st.subheader("🧪 1-Click Evaluation Prompts")
    st.caption("Pre-configured rubric benchmark questions:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📌 Factual Specs", use_container_width=True):
            st.session_state["prompt_input"] = "What is the battery life, charging port, and baseline memory of the MacBook Air 13-inch M3?"
        if st.button("📋 Return Policy", use_container_width=True):
            st.session_state["prompt_input"] = "What is Apple standard return window for opened items, and what is the accidental damage deductible for iPhone screen repair under AppleCare+?"
    with col_btn2:
        if st.button("⚖️ Comparison", use_container_width=True):
            st.session_state["prompt_input"] = "Compare the display refresh rate, optical zoom, and materials between iPhone 16 and iPhone 16 Pro."
        if st.button("🚫 Out-of-Scope", use_container_width=True):
            st.session_state["prompt_input"] = "What is the camera resolution and battery life of the Samsung Galaxy S24 Ultra?"

    st.markdown("---")
    with st.expander("📖 RAG Pipeline Architecture"):
        st.markdown("""
        **End-to-End Pipeline Workflow:**
        1. **Data Layer**: Cleaned knowledge base (64 records) covering iPhones, Macs, iPads, Watches, AirPods, Services, and Policies.
        2. **Hierarchical Chunker**: Produces 256 discrete semantic chunks (Overview, Specs, FAQs, Policy clauses).
        3. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` generates 384-dim normalized dense vectors.
        4. **Vector Store**: Meta `FAISS` Inner-Product Cosine Index for dense vector similarity.
        5. **Lexical Index**: `rank_bm25` (BM25Okapi) for exact keyword and number matching ($29, 120Hz, M4).
        6. **Hybrid RRF / Decomposition**: Reciprocal Rank Fusion fuses vector and lexical ranks with $k=60$. Multi-intent questions are split into targeted sub-queries.
        7. **Grounded Generation & Guardrail**: Strict context conditioning refuses out-of-scope queries (e.g. competitor devices) and flags low-confidence retrievals.
        """)

    with st.expander("📊 Dataset & Index Statistics"):
        st.write({
            "Total Catalog Items": 64,
            "Total Document Chunks": 256,
            "Vector Dimension": 384,
            "Dense Index": "FAISS IndexFlatIP",
            "Lexical Index": "BM25Okapi",
            "Embedding Model": "all-MiniLM-L6-v2"
        })

# Main Chat Interface
st.title("🍎 Apple Genius AI Support Assistant")
st.markdown("Empowering customers and technicians with **grounded retrieval** across Apple hardware, technical specifications, AppleCare+ policies, and troubleshooting protocols.")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AppleCare Genius AI assistant. I can help answer questions regarding Apple products, detailed technical specifications, warranty terms, AppleCare+ fees, 14-day return policies, or hardware troubleshooting procedures. What can I assist you with today?",
            "metadata": None
        }
    ]

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🍎" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            meta = msg["metadata"]
            badge_class = "badge-decomposed" if meta.get("mode") == "decomposed" else ("badge-hybrid" if meta.get("mode") == "hybrid" else "badge-semantic")
            
            # Badges
            badges_html = f"<span class='apple-badge {badge_class}'>Mode: {meta.get('mode', '').upper()}</span>"
            if meta.get("is_fallback"):
                badges_html += "<span class='apple-badge badge-fallback'>⚠️ Fallback Triggered</span>"
            badges_html += f"<span class='apple-badge badge-source'>⏱️ Total: {meta.get('latency', {}).get('total_ms', 0)}ms</span>"
            badges_html += f"<span class='apple-badge badge-source'>Engine: {meta.get('backend', '')}</span>"
            st.markdown(badges_html, unsafe_allow_html=True)

            # Sources Expander
            if meta.get("chunks"):
                with st.expander(f"🔍 Inspect Retrieved Context ({len(meta['chunks'])} Chunks)"):
                    if meta.get("subqueries"):
                        st.markdown(f"**Decomposed Sub-queries:** `{', '.join(meta['subqueries'])}`")
                    for i, ch in enumerate(meta["chunks"], start=1):
                        st.markdown(f"**[{i}] {ch.get('name')}** • *Category:* `{ch.get('category')}` • *Section:* `{ch.get('section')}` • *Score:* `{ch.get('score')}`")
                        st.code(ch.get("text"), language="markdown")

# Process User Input
user_query = st.chat_input("Ask a question about Apple products, specs, policies, or troubleshooting...")

# If triggered by a quick button
if "prompt_input" in st.session_state and st.session_state["prompt_input"]:
    user_query = st.session_state["prompt_input"]
    del st.session_state["prompt_input"]

if user_query:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Run Pipeline
    with st.chat_message("assistant", avatar="🍎"):
        with st.spinner("Retrieving grounded knowledge and formulating response..."):
            pipeline = get_rag_pipeline(
                retrieval_mode=selected_mode,
                generator_backend=selected_backend,
                top_k=top_k
            )
            result = pipeline.query(user_query)

        # Display assistant response
        st.markdown(result["answer"])

        meta_info = {
            "mode": result["retrieval_mode"],
            "backend": result["generator_backend"],
            "is_fallback": result["is_fallback"],
            "latency": result["latency"],
            "chunks": result["retrieved_chunks"],
            "subqueries": result["subqueries"]
        }

        # Render metadata badges
        badge_class = "badge-decomposed" if result["retrieval_mode"] == "decomposed" else ("badge-hybrid" if result["retrieval_mode"] == "hybrid" else "badge-semantic")
        badges_html = f"<span class='apple-badge {badge_class}'>Mode: {result['retrieval_mode'].upper()}</span>"
        if result["is_fallback"]:
            badges_html += "<span class='apple-badge badge-fallback'>⚠️ Fallback / Refusal Triggered</span>"
        badges_html += f"<span class='apple-badge badge-source'>⏱️ Retrieval: {result['latency']['retrieval_ms']}ms | Generation: {result['latency']['generation_ms']}ms</span>"
        badges_html += f"<span class='apple-badge badge-source'>Engine: {result['generator_backend']}</span>"
        st.markdown(badges_html, unsafe_allow_html=True)

        if result["retrieved_chunks"]:
            with st.expander(f"🔍 Inspect Retrieved Context ({len(result['retrieved_chunks'])} Chunks)"):
                if result.get("subqueries"):
                    st.markdown(f"**Decomposed Sub-queries:** `{', '.join(result['subqueries'])}`")
                for i, ch in enumerate(result["retrieved_chunks"], start=1):
                    st.markdown(f"**[{i}] {ch.get('name')}** • *Category:* `{ch.get('category')}` • *Section:* `{ch.get('section')}` • *Score:* `{ch.get('score')}`")
                    st.code(ch.get("text"), language="markdown")

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "metadata": meta_info
    })
