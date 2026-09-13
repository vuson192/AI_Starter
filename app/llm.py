"""
Khối: GỌI MODEL.

Đây là "component" trung tâm mà AI Engineer làm việc cùng. Điểm mấu chốt:
ta bọc model sau một interface chung (chat) để phần còn lại của hệ thống
không quan tâm đang dùng model nào. Đây là kỹ thuật dev thuần túy:
trừu tượng hóa một dependency không ổn định.

Hai chế độ:
- "mock": không cần mạng/API key. Trả lời theo luật đơn giản để bạn thấy luồng.
- "openai": gọi model thật khi bạn đặt OPENAI_API_KEY và LLM_PROVIDER=openai.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


class LLM:
    """Interface tối giản: đưa vào danh sách message, nhận lại text."""

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None) -> str:
        if _provider() == "openai":
            return self._chat_openai(messages, tools)
        return self._chat_mock(messages, tools)

    # ---- OpenAI thật ----
    def _chat_openai(self, messages: list[dict[str, str]], tools: list[dict] | None) -> str:
        from openai import OpenAI  # import trong hàm để mock không cần cài openai

        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    # ---- Mock: giả lập hành vi model bằng luật đơn giản ----
    def _chat_mock(self, messages: list[dict[str, str]], tools: list[dict] | None) -> str:
        user = ""
        context = ""
        for m in messages:
            if m["role"] == "user":
                user = m["content"]
            if m["role"] == "system" and "NGỮ CẢNH" in m["content"]:
                context = m["content"]

        # Nếu prompt yêu cầu quyết định gọi tool (agent), trả JSON theo luật.
        if tools is not None:
            return self._mock_tool_decision(user)

        # Chế độ RAG: trả lời bằng cách trích câu liên quan nhất từ ngữ cảnh.
        if context:
            return self._mock_answer_from_context(user, context)

        return "Tôi là model mock. Hãy đặt LLM_PROVIDER=openai để dùng model thật."

    def _mock_answer_from_context(self, question: str, context: str) -> str:
        q_words = set(re.findall(r"\w+", question.lower()))
        best_line, best_score = "", -1
        for line in context.splitlines():
            line = line.strip("- ").strip()
            if not line:
                continue
            score = len(q_words & set(re.findall(r"\w+", line.lower())))
            if score > best_score:
                best_line, best_score = line, score
        if best_score <= 0:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
        return best_line

    def _mock_tool_decision(self, user: str) -> str:
        """Giả lập việc model chọn tool. Trả JSON để agent thực thi."""
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:của|cua|of)\s*(\d[\d.,]*)", user.lower())
        if m:
            pct = float(m.group(1).replace(",", "."))
            base = float(m.group(2).replace(".", "").replace(",", ""))
            return json.dumps({"tool": "calculator", "args": {"expression": f"{base}*{pct}/100"}})
        return json.dumps({"tool": "knowledge_lookup", "args": {"query": user}})
