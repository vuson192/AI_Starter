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
    provider = _provider()
    if provider == "openai":
        return _embed_openai(text)
    if provider == "ollama":
        return _embed_ollama(text)
    return _embed_mock(text)


def _embed_ollama(text: str) -> np.ndarray:
    """Gọi Ollama để tạo embedding thật. Cần một embedding model, vd nomic-embed-text."""
    import json as _json
    import urllib.request

    from .llm import ollama_host

    host = ollama_host()
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    payload = {"model": model, "prompt": text}
    req = urllib.request.Request(
        f"{host}/api/embeddings",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return np.array(data["embedding"], dtype=np.float32)


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
