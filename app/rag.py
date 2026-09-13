"""
Khối: RAG (Retrieval-Augmented Generation).

Đây là kỹ năng lõi số một của AI Engineer. Luồng gồm 4 bước:

1. CHUNK  : chia tài liệu dài thành các đoạn nhỏ (vì prompt có giới hạn,
            và tìm kiếm đoạn nhỏ thì chính xác hơn).
2. INDEX  : embedding từng đoạn rồi lưu vào vector store.
3. RETRIEVE: với câu hỏi, tìm các đoạn liên quan nhất.
4. GENERATE: nhét các đoạn đó vào prompt làm "ngữ cảnh", rồi hỏi model.

Điểm cốt lõi: model KHÔNG trả lời bằng kiến thức của nó, mà bằng tài liệu
bạn cung cấp. Nhờ vậy nó dùng được dữ liệu nội bộ và ít bịa hơn.
"""
from __future__ import annotations

import os
import re

from .llm import LLM
from .vectorstore import VectorStore


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Chia theo đoạn/tiêu đề, gộp lại sao cho mỗi chunk không quá dài."""
    blocks = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            for i in range(0, len(block), max_chars):
                chunks.append(block[i : i + max_chars])
    return chunks


def build_store(data_dir: str = "data") -> VectorStore:
    store = VectorStore()
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        store.add(chunks, source=name)
    return store


def answer(question: str, store: VectorStore, top_k: int = 3) -> dict:
    hits = store.search(question, top_k=top_k)
    context = "\n".join(f"- {doc.text}" for doc, _ in hits)

    system = (
        "Bạn là trợ lý trả lời dựa trên NGỮ CẢNH được cung cấp. "
        "Chỉ dùng thông tin trong ngữ cảnh, không bịa. Nếu không có, hãy nói không biết.\n\n"
        f"NGỮ CẢNH:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    reply = LLM().chat(messages)
    return {
        "answer": reply,
        "sources": [{"source": doc.source, "score": round(score, 3), "text": doc.text} for doc, score in hits],
    }
