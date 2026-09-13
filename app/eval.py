"""
Khối: EVAL (đánh giá chất lượng).

Vì output của model không cố định, ta không test bằng `assert x == 5`.
Thay vào đó, ta xây một bộ (câu hỏi, từ khóa kỳ vọng) rồi chấm điểm: câu
trả lời có chứa thông tin đúng không.

Đây là thứ phân biệt người làm AI nghiêm túc: không có eval thì bạn không
biết sản phẩm tốt hay tệ, và mỗi lần chỉnh prompt/model bạn không biết
mình đang cải thiện hay làm hỏng. Production dùng eval tinh vi hơn (LLM chấm
LLM, so với đáp án chuẩn, đo tỉ lệ bịa...), nhưng ý tưởng gốc là cái dưới đây.
"""
from __future__ import annotations

from .rag import answer, build_store

# Bộ test: câu hỏi + các từ khóa PHẢI xuất hiện trong câu trả lời đúng.
TEST_CASES = [
    {"q": "Được nghỉ phép bao nhiêu ngày mỗi năm?", "expect_any": ["12"]},
    {"q": "Sản phẩm được bảo hành bao lâu?", "expect_any": ["24"]},
    {"q": "Chính sách hoàn tiền trong bao nhiêu ngày?", "expect_any": ["14"]},
    {"q": "Được làm việc từ xa mấy ngày một tuần?", "expect_any": ["2"]},
]


def run_eval() -> dict:
    store = build_store()
    results = []
    passed = 0
    for case in TEST_CASES:
        out = answer(case["q"], store)
        text = out["answer"].lower()
        ok = any(kw.lower() in text for kw in case["expect_any"])
        passed += 1 if ok else 0
        results.append(
            {
                "question": case["q"],
                "answer": out["answer"],
                "expect_any": case["expect_any"],
                "pass": ok,
            }
        )
    return {
        "total": len(TEST_CASES),
        "passed": passed,
        "score": round(passed / len(TEST_CASES), 2),
        "results": results,
    }
