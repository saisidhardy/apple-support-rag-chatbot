from typing import List, Dict, Any

class SmartFollowUpGenerator:
    """
    Generates dynamic, highly contextual follow-up questions
    based on the current topic, category, and retrieved context.
    """
    def generate_suggestions(self, query: str, retrieved_chunks: List[Any], is_fallback: bool) -> List[str]:
        if is_fallback:
            return [
                "What Apple products and specifications can you help me with?",
                "What is Apple standard 14-day return policy?",
                "How do AppleCare+ repair deductibles work?"
            ]

        q_lower = query.lower()

        # Categorical and Entity checks
        if "iphone" in q_lower:
            return [
                "Compare the camera specs and refresh rate of iPhone 16 vs 16 Pro",
                "What are the AppleCare+ accidental damage repair fees for iPhone?",
                "What is Apple trade-in value policy for older devices?"
            ]
        elif "mac" in q_lower or "macbook" in q_lower:
            return [
                "How many external monitors does MacBook Air M3 support?",
                "What are the differences between M4, M4 Pro, and M4 Max?",
                "What is the AppleCare+ deductible for Mac screen damage?"
            ]
        elif "ipad" in q_lower:
            return [
                "Which Apple Pencil works with iPad Pro M4?",
                "What is the Tandem OLED display technology in iPad Pro?",
                "Does iPad Air M2 have 120Hz ProMotion?"
            ]
        elif "watch" in q_lower:
            return [
                "How deep can I dive with Apple Watch Ultra 2?",
                "Can Apple Watch Series 10 play audio through its built-in speaker?",
                "What are AppleCare+ deductibles for Apple Watch?"
            ]
        elif "airpod" in q_lower or "audio" in q_lower:
            return [
                "How does the FDA-authorized Hearing Aid feature work on AirPods Pro 2?",
                "Do AirPods 4 ANC have silicone ear tips?",
                "Are AirPods Pro 2 and their case water resistant?"
            ]
        elif "return" in q_lower or "refund" in q_lower or "policy" in q_lower:
            return [
                "Can I return an opened Apple product to a physical Apple Store?",
                "How long does a refund take to process?",
                "What are the exact deductibles under AppleCare+?"
            ]
        elif "restart" in q_lower or "frozen" in q_lower or "liquid" in q_lower:
            return [
                "Why does Apple advise against putting a wet phone in rice?",
                "Will a force restart delete my personal data or photos?",
                "How do technicians check for water damage?"
            ]
        else:
            return [
                "What is Apple official 14-day return window?",
                "What does the 1-year limited hardware warranty cover?",
                "Compare MacBook Air M3 and MacBook Pro 14 M4"
            ]
