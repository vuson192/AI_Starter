"""
Khối: VECTOR STORE (tìm kiếm theo ngữ nghĩa).

Một vector store làm 2 việc:
1. Lưu các đoạn văn bản kèm vector của chúng.
2. Khi có câu hỏi, tính vector câu hỏi rồi tìm các đoạn có vector gần nhất.

Ở production người ta dùng Pinecone, pgvector, Milvus, Qdrant... Bản chất
giống hệt cái dưới đây, chỉ khác là chúng scale lên hàng triệu vector và
tìm nhanh bằng thuật toán xấp xỉ (ANN). Ở đây ta làm bản in-memory để hiểu lõi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embeddings import embed, embed_many, cosine_similarity


@dataclass
class Doc:
    text: str
    source: str
    vector: np.ndarray | None = None


@dataclass
class VectorStore:
    docs: list[Doc] = field(default_factory=list)

    def add(self, texts: list[str], source: str = "unknown") -> None:
        vectors = embed_many(texts)
        for t, v in zip(texts, vectors):
            self.docs.append(Doc(text=t, source=source, vector=v))

    def search(self, query: str, top_k: int = 3) -> list[tuple[Doc, float]]:
        qv = embed(query)
        scored = [(d, cosine_similarity(qv, d.vector)) for d in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
