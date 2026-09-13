"""
Khối: AGENT.

Khác với RAG (chỉ trả lời), agent *hành động*. Vòng lặp cơ bản của agent:

1. Đưa câu hỏi + danh sách tool cho model.
2. Model quyết định: gọi tool nào, với tham số gì (trả về dưới dạng JSON).
3. Ta thực thi tool đó, lấy kết quả.
4. (Có thể lặp lại nhiều bước) rồi tổng hợp thành câu trả lời cuối.

Đây là khung tối giản 1 bước để bạn thấy rõ cơ chế "model quyết định -> code
thực thi". Agent thật (LangChain, ReAct...) chỉ là lặp bước này nhiều lần và
xử lý lỗi kỹ hơn.
"""
from __future__ import annotations

import json

from .llm import LLM
from .tools import build_toolset
from .vectorstore import VectorStore


def run_agent(user_query: str, store: VectorStore) -> dict:
    tools = build_toolset(store)
    tool_desc = "\n".join(f"- {name}: {meta['desc']}" for name, meta in tools.items())

    system = (
        "Bạn là agent. Với yêu cầu của người dùng, hãy chọn MỘT tool để dùng. "
        "Trả về DUY NHẤT một JSON dạng {\"tool\": <tên>, \"args\": {...}}.\n\n"
        f"Các tool có sẵn:\n{tool_desc}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    raw = LLM().chat(messages, tools=list(tools.keys()))
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Model không trả về JSON hợp lệ", "raw": raw}

    tool_name = decision.get("tool")
    args = decision.get("args", {})
    if tool_name not in tools:
        return {"error": f"Tool không tồn tại: {tool_name}", "decision": decision}

    try:
        result = tools[tool_name]["fn"](args)
    except Exception as e:  # noqa: BLE001 - demo: bắt để báo lỗi gọn
        return {"error": f"Lỗi khi chạy tool {tool_name}: {e}", "decision": decision}

    return {
        "tool_used": tool_name,
        "args": args,
        "result": result,
        "final_answer": f"Đã dùng tool '{tool_name}', kết quả: {result}",
    }
