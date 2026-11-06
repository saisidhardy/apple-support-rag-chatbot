import os
import json
import re
import urllib.request
from typing import List, Dict, Any, Optional
from .chunker import DocumentChunk

SYSTEM_PROMPT = """You are AppleCare Genius AI, an official technical advisor and support assistant for Apple Inc.
Your purpose is to answer customer queries accurately, politely, and faithfully using ONLY the provided retrieved context.

STRICT INSTRUCTIONS:
1. Grounding: Rely strictly on the information provided in the [RETRIEVED CONTEXT] below. Do NOT hallucinate or extrapolate facts not present in the context.
2. Fallback & Boundaries: If the question cannot be answered from the provided context (e.g., questions about competitor products like Samsung Galaxy, Dell, Windows OS, or unverified claims), you MUST clearly and politely state:
   "I do not have sufficient information in the official Apple knowledge dataset to answer this question. My knowledge is strictly grounded in official Apple hardware, technical specifications, AppleCare+, warranty policies, and troubleshooting procedures."
3. Formatting: Structure your response with clean markdown headings, bullet points, and exact numbers/pricing where relevant.
4. Transparency: Explicitly cite the product, policy, or guide name referenced from the context.
"""

OUT_OF_SCOPE_PATTERNS = [
    r'\bsamsung\b', r'\bgalaxy\s*s\d+\b', r'\bdell\b', r'\bwindows\s*1\d\b',
    r'\blenovo\b', r'\bhp\s*laptop\b', r'\basus\b', r'\bxbox\b',
    r'\bplaystation\b', r'\bnvidia\b', r'\brtx\s*\d+\b', r'\bintel\s*core\s*i\d\b',
    r'\bweather\b', r'\bcooking\b', r'\brecipe\b', r'\bpolitics\b'
]

class GroundedGenerator:
    """
    RAG Generation Engine with multiple backend support:
      - 'ollama' (Local Ollama qwen3.5:4b)
      - 'gemini' (Google Gemini API via GEMINI_API_KEY)
      - 'openai' (OpenAI GPT-4o / GPT-4o-mini via OPENAI_API_KEY)
      - 'extractive' (Deterministic, offline rule-based semantic synthesizer)
    """
    def __init__(
        self,
        backend: str = "ollama",
        ollama_model: str = "qwen3.5:4b",
        ollama_url: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def is_out_of_scope(self, query: str) -> bool:
        q_lower = query.lower()
        for pat in OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, q_lower):
                # Check if it's an allowed comparison or trade-in mention (e.g. "Can I trade in Samsung?")
                if "trade in" in q_lower or "trade-in" in q_lower:
                    return False
                return True
        return False

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        confidence_score: float = 1.0,
        confidence_threshold: float = 0.015
    ) -> Dict[str, Any]:
        """
        Generates a grounded response given query and retrieved chunks.
        Returns:
            {
                "answer": str,
                "is_fallback": bool,
                "confidence_score": float,
                "backend_used": str,
                "sources": List[str]
            }
        """
        # 1. Guardrail / Fallback Checks
        if self.is_out_of_scope(query) or confidence_score < confidence_threshold or not retrieved_chunks:
            return {
                "answer": (
                    "I do not have sufficient information in the official Apple knowledge dataset to answer this question. "
                    "My knowledge base is strictly limited to verified Apple hardware, technical specifications, "
                    "AppleCare+, warranty guidelines, and troubleshooting procedures."
                ),
                "is_fallback": True,
                "confidence_score": confidence_score,
                "backend_used": "guardrail_fallback",
                "sources": []
            }

        # Prepare context blocks
        context_blocks = []
        sources = []
        for i, item in enumerate(retrieved_chunks, start=1):
            if isinstance(item, tuple):
                chunk = item[0]
            else:
                chunk = item
            context_blocks.append(f"[Source {i} | {chunk.metadata.get('name', 'Apple Doc')}]:\n{chunk.text}")
            sources.append(chunk.metadata.get("name", f"Document {i}"))

        context_str = "\n\n".join(context_blocks)

        prompt = f"""[RETRIEVED CONTEXT]
{context_str}

[USER QUESTION]
{query}

[GROUNDED ANSWER]
"""

        # 2. Execute selected backend
        if self.backend == "ollama":
            try:
                ans = self._generate_ollama(prompt)
                return {
                    "answer": ans,
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": f"Ollama ({self.ollama_model})",
                    "sources": sources
                }
            except Exception as e:
                # Fallback to local extractive synthesizer if Ollama errors
                ans = self._generate_extractive(query, retrieved_chunks)
                return {
                    "answer": ans + f"\n\n*(Note: Generated via Local Extractive Engine due to Ollama timeout: {e})*",
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": "extractive_fallback",
                    "sources": sources
                }

        elif self.backend == "gemini":
            try:
                ans = self._generate_gemini(prompt)
                return {
                    "answer": ans,
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": "Google Gemini",
                    "sources": sources
                }
            except Exception as e:
                ans = self._generate_extractive(query, retrieved_chunks)
                return {
                    "answer": ans + f"\n\n*(Fallback to Extractive Engine: {e})*",
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": "extractive_fallback",
                    "sources": sources
                }

        elif self.backend == "openai":
            try:
                ans = self._generate_openai(prompt)
                return {
                    "answer": ans,
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": "OpenAI",
                    "sources": sources
                }
            except Exception as e:
                ans = self._generate_extractive(query, retrieved_chunks)
                return {
                    "answer": ans + f"\n\n*(Fallback to Extractive Engine: {e})*",
                    "is_fallback": False,
                    "confidence_score": confidence_score,
                    "backend_used": "extractive_fallback",
                    "sources": sources
                }

        else: # Extractive
            ans = self._generate_extractive(query, retrieved_chunks)
            return {
                "answer": ans,
                "is_fallback": False,
                "confidence_score": confidence_score,
                "backend_used": "Local High-Precision Synthesizer",
                "sources": sources
            }

    def _generate_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()

    def _generate_gemini(self, prompt: str) -> str:
        import urllib.request
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode())
            return res["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _generate_openai(self, prompt: str) -> str:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()

    def _generate_extractive(self, query: str, retrieved_chunks: List[Any]) -> str:
        """
        High-precision deterministic factual synthesizer that aggregates
        specifications, pricing, and policy clauses into clean markdown.
        """
        lines = ["### Official Apple Support & Specifications Summary\n"]
        for i, item in enumerate(retrieved_chunks[:3], start=1):
            chunk = item[0] if isinstance(item, tuple) else item
            name = chunk.metadata.get("name", "Apple Product/Policy")
            price = chunk.metadata.get("price", "")
            lines.append(f"#### **{name}**")
            if price and price != "N/A":
                lines.append(f"- **Pricing/Fee:** {price}")
            
            # Format text body cleanly
            content = chunk.text
            for cl in content.split("\n"):
                cl_clean = cl.strip()
                if cl_clean and not cl_clean.startswith("[") and not cl_clean.startswith("Price"):
                    lines.append(f"- {cl_clean}")
            lines.append("")

        lines.append("---")
        lines.append("*Grounded answer strictly compiled from Apple Official Knowledge Base.*")
        return "\n".join(lines)
