"""
Khối: CÔNG CỤ (tools) cho agent.

Tool là các hàm mà agent được phép gọi để *làm việc thật*: tính toán,
tra cứu dữ liệu, gọi API nội bộ... Mỗi tool có:
- tên (để model gọi)
- mô tả (để model biết khi nào dùng)
- một hàm Python thực thi

Lưu ý an toàn: KHÔNG bao giờ dùng eval() trực tiếp lên chuỗi từ model.
Ta chỉ cho phép biểu thức số học đơn giản qua một parser an toàn.
"""
from __future__ import annotations

import ast
import operator

from .vectorstore import VectorStore

# --- Tool 1: máy tính an toàn ---
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Chỉ chấp nhận số.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Biểu thức không được phép.")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    # bỏ .0 thừa cho gọn
    if result == int(result):
        result = int(result)
    return str(result)


# --- Tool 2: tra cứu kiến thức (dùng lại RAG store) ---
def make_knowledge_lookup(store: VectorStore):
    def knowledge_lookup(query: str) -> str:
        hits = store.search(query, top_k=1)
        if not hits:
            return "Không tìm thấy."
        return hits[0][0].text

    return knowledge_lookup


def build_toolset(store: VectorStore) -> dict:
    """Trả về map tên_tool -> hàm, và mô tả để agent biết có gì dùng."""
    return {
        "calculator": {
            "fn": lambda args: calculator(args["expression"]),
            "desc": "Tính biểu thức số học. args: {expression: str}",
        },
        "knowledge_lookup": {
            "fn": lambda args: make_knowledge_lookup(store)(args["query"]),
            "desc": "Tra cứu thông tin trong tài liệu công ty. args: {query: str}",
        },
    }
