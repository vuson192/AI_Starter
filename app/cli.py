"""
Điểm vào dòng lệnh. Ghép 3 khối RAG / Agent / Eval lại.

Cách dùng:
    python -m app.cli ask "câu hỏi"
    python -m app.cli agent "yêu cầu"
    python -m app.cli eval
"""
from __future__ import annotations

import json
import sys

# Windows console mặc định dùng codepage cp1252, không in được tiếng Việt và
# sẽ ném UnicodeEncodeError. Ép stdout/stderr sang UTF-8 để chạy ổn mọi nơi.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from .agent import run_agent
from .eval import run_eval
from .rag import answer, build_store


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("Dùng: python -m app.cli [ask|agent|eval] <nội dung>")
        return 1

    command = argv[0]

    if command == "eval":
        _print(run_eval())
        return 0

    if command in ("ask", "agent"):
        if len(argv) < 2:
            print(f"Thiếu nội dung cho lệnh '{command}'.")
            return 1
        query = " ".join(argv[1:])
        store = build_store()
        if command == "ask":
            _print(answer(query, store))
        else:
            _print(run_agent(query, store))
        return 0

    print(f"Lệnh không hợp lệ: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
