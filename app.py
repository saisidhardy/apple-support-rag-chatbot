import os
import sys
import streamlit as st

# Ensure root directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline import AppleRAGPipeline

st.set_page_config(
    page_title="Apple Genius AI | Support & Discovery",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-Premium Cupertino Glassmorphism & Dark Aesthetic CSS
st.markdown("""
<style>
    /* Global SF Pro Typography */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #171821 0%, #0c0d12 100%);
        color: #f5f5f7;
    }

    /* Completely Remove Sidebar & Collapsed Toggle Chevron */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Centered, Optimal Conversational Layout */
    .main .block-container {
        max-width: 960px !important;
        margin: 0 auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
    }

    /* Top Hero Header */
    .apple-hero {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 16px;
        backdrop-filter: blur(24px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .apple-glow-text {
        background: linear-gradient(135deg, #ffffff 0%, #a1a1a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 28px;
        letter-spacing: -0.5px;
    }

    .apple-intelligence-pill {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(59, 130, 246, 0.2) 50%, rgba(6, 182, 212, 0.2) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        color: #e9d5ff;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Message Bubbles */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        backdrop-filter: blur(12px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache Pipeline Instance with Best Re-Ranking Engine Locked as Default
@st.cache_resource(show_spinner=False)
def get_advanced_rag_pipeline():
    return AppleRAGPipeline(
        retrieval_mode="hybrid_rerank",
        generator_backend="ollama",
        top_k=4
    )

# Top Hero Bar
st.markdown("""
<div class="apple-hero">
    <div>
        <div class="apple-glow-text">🍎 Apple Genius AI</div>
        <div style="color: #86868b; font-size: 13px; margin-top: 4px;">
            Official Technical & Support Advisor for Apple Products & Services
        </div>
    </div>
    <div style="text-align: right;">
        <div class="apple-intelligence-pill">
            <span class="pulse-dot"></span>
            APPLE INTELLIGENCE
        </div>
        <div style="color: #64748b; font-size: 11px; margin-top: 6px;">
            Grounded & Verified Support
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Prompts & Reset Header
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.markdown("##### ⚡ Quick Prompts")
with col_head2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Quick Discovery Grid
qcol1, qcol2, qcol3, qcol4 = st.columns(4)

with qcol1:
    if st.button("📱 iPhone 16 Pro Max", use_container_width=True):
        st.session_state["prompt_input"] = "What are the camera specifications, battery life, and materials of the iPhone 16 Pro Max?"
    if st.button("🛡️ AppleCare+ Fees", use_container_width=True):
        st.session_state["prompt_input"] = "What is the deductible for repairing a cracked screen under AppleCare+, and how many incidents are allowed?"

with qcol2:
    if st.button("💻 MacBook Pro M4", use_container_width=True):
        st.session_state["prompt_input"] = "What is the maximum battery life and external port setup on the 14-inch MacBook Pro M4?"
    if st.button("🔄 14-Day Return Window", use_container_width=True):
        st.session_state["prompt_input"] = "Can I return an opened Apple product within 14 days, and can I return it to a physical Apple Store?"

with qcol3:
    if st.button("⌚ Apple Watch Ultra 2", use_container_width=True):
        st.session_state["prompt_input"] = "How deep can I scuba dive with the Apple Watch Ultra 2, and what does the Action button do?"
    if st.button("🛠️ Frozen Force Restart", use_container_width=True):
        st.session_state["prompt_input"] = "How do I force restart an iPhone 16 that is stuck on a black screen without losing data?"

with qcol4:
    if st.button("🎧 AirPods Pro 2 Hearing", use_container_width=True):
        st.session_state["prompt_input"] = "What is the new FDA-authorized Hearing Aid feature on AirPods Pro 2, and are they water resistant?"
    if st.button("🚫 Out-of-Scope Test", use_container_width=True):
        st.session_state["prompt_input"] = "What is the battery capacity and camera megapixel count of the Samsung Galaxy S24 Ultra?"

st.markdown("---")

# Session State for Messages
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to Apple Genius AI.**\n\n"
                "I am your official technical advisor grounded strictly in Apple's verified product catalog, hardware specifications, "
                "AppleCare+ coverage terms, store return guidelines, and troubleshooting protocols.\n\n"
                "**How can I assist you with your Apple devices today?**"
            ),
            "follow_ups": [
                "What is the battery life and price of MacBook Air M3?",
                "Compare iPhone 16 and iPhone 16 Pro cameras",
                "What is Apple standard return and refund window?"
            ]
        }
    ]

# Render Message Stream
for idx, msg in enumerate(st.session_state.messages):
    is_assistant = (msg["role"] == "assistant")
    with st.chat_message(msg["role"], avatar="🍎" if is_assistant else "👤"):
        st.markdown(msg["content"])

        # Render Dynamic Follow-Up Buttons
        if is_assistant and msg.get("follow_ups"):
            st.markdown("<div style='font-size: 11px; color: #86868b; margin-top: 8px;'>Suggested Questions:</div>", unsafe_allow_html=True)
            fcols = st.columns(len(msg["follow_ups"]))
            for fi, fq in enumerate(msg["follow_ups"]):
                with fcols[fi]:
                    if st.button(fq, key=f"fup_{idx}_{fi}", use_container_width=True):
                        st.session_state["prompt_input"] = fq
                        st.rerun()

# User Input Handling
user_input = st.chat_input("Ask about Apple devices, specs, AppleCare+, return window, or troubleshooting...")

if "prompt_input" in st.session_state and st.session_state["prompt_input"]:
    user_input = st.session_state["prompt_input"]
    del st.session_state["prompt_input"]

if user_input:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Execute RAG Pipeline with High-Precision Multi-Stage Engine
    with st.chat_message("assistant", avatar="🍎"):
        with st.spinner("Thinking..."):
            pipeline = get_advanced_rag_pipeline()
            result = pipeline.query(user_input)

        st.markdown(result["answer"])

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "follow_ups": result.get("follow_ups", [])
    })
    st.rerun()
