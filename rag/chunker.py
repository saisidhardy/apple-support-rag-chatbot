import re
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }

class AppleKnowledgeChunker:
    """
    Hierarchical Semantic Chunker designed for structured product,
    specification, policy, and troubleshooting records.
    
    Rather than blind arbitrary-length splitting, this chunker creates
    coherent, self-contained semantic passages with rich metadata headers.
    """
    def __init__(self, max_chunk_size: int = 600, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_records(self, records: List[Dict[str, Any]]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        for record in records:
            item_id = record.get("id", "UNKNOWN")
            category = record.get("category", "General")
            name = record.get("name", "Unknown Item")
            price = record.get("price", "N/A")

            # 1. Product / Policy Overview Chunk
            overview_text = (
                f"[{category}] {name}\n"
                f"Price/Cost: {price}\n"
                f"Overview: {record.get('description', '')}"
            )
            chunks.append(DocumentChunk(
                chunk_id=f"{item_id}_overview",
                text=overview_text.strip(),
                metadata={
                    "item_id": item_id,
                    "name": name,
                    "category": category,
                    "section": "overview",
                    "price": price,
                    "keywords": record.get("keywords", "")
                }
            ))

            # 2. Technical Specifications / Core Attributes Chunk
            specs = record.get("specs", "").strip()
            if specs:
                specs_text = (
                    f"[{category}] {name} - Technical Specifications\n"
                    f"Specifications: {specs}"
                )
                chunks.append(DocumentChunk(
                    chunk_id=f"{item_id}_specs",
                    text=specs_text.strip(),
                    metadata={
                        "item_id": item_id,
                        "name": name,
                        "category": category,
                        "section": "specifications",
                        "price": price,
                        "keywords": record.get("keywords", "")
                    }
                ))

            # 3. FAQs & Common Inquiries Chunk
            faqs = record.get("faqs", "").strip()
            if faqs:
                faq_text = (
                    f"[{category}] {name} - Frequently Asked Questions\n"
                    f"{faqs}"
                )
                chunks.append(DocumentChunk(
                    chunk_id=f"{item_id}_faqs",
                    text=faq_text.strip(),
                    metadata={
                        "item_id": item_id,
                        "name": name,
                        "category": category,
                        "section": "faqs",
                        "price": price,
                        "keywords": record.get("keywords", "")
                    }
                ))

            # 4. Policy Terms & Warranty Details Chunk
            terms = record.get("policy_terms", "").strip()
            if terms:
                terms_text = (
                    f"[{category}] {name} - Policy Terms & Warranty Coverage\n"
                    f"{terms}"
                )
                chunks.append(DocumentChunk(
                    chunk_id=f"{item_id}_terms",
                    text=terms_text.strip(),
                    metadata={
                        "item_id": item_id,
                        "name": name,
                        "category": category,
                        "section": "terms_and_policy",
                        "price": price,
                        "keywords": record.get("keywords", "")
                    }
                ))

        return chunks
