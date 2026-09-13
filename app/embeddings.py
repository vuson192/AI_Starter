"""
Khối: EMBEDDING.

Embedding = biến một đoạn chữ thành một vector số. Ý tưởng: hai đoạn chữ
gần nghĩa nhau sẽ có vector gần nhau trong không gian. Nhờ đó máy "so sánh
ngữ nghĩa" được, thay vì chỉ khớp từ khóa.

Ở chế độ mock, ta tự tạo một embedding đơn giản dựa trên tần suất ký tự
(hash-based bag of words). Nó KHÔNG tốt như embedding thật, nhưng đủ để
minh họa cơ chế: text -> vector -> so sánh bằng cosine.

Khi bật OpenAI, ta gọi model embedding thật (chất lượng cao hơn nhiều).
"""
from __future__ import annotations

import hashlib
import math
import os
import re

import numpy as np

_DIM = 256  # số chiều vector ở chế độ mock


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


def embed(text: str) -> np.ndarray:
    if _provider() == "openai":
        return _embed_openai(text)
    return _embed_mock(text)


def embed_many(texts: list[str]) -> np.ndarray:
    return np.vstack([embed(t) for t in texts])


def _embed_openai(text: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _embed_mock(text: str) -> np.ndarray:
    """Bag-of-words hashing: mỗi từ được hash vào một chiều, cộng dồn."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm  # chuẩn hóa để cosine tính đơn giản
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
