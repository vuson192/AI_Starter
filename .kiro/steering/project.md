# AI Starter — Project Context

## Tổng quan
Dự án học việc để hiểu công việc AI Engineer, ghép 3 khối cốt lõi:
- **RAG** (Retrieval-Augmented Generation): LLM trả lời dựa trên tài liệu riêng.
- **Agent**: LLM gọi công cụ (tool) để hành động, không chỉ trả lời.
- **Eval**: đo chất lượng câu trả lời một cách có hệ thống.

Chạy được ngay ở chế độ **mock** (không cần API key). Cắm OpenAI qua biến môi trường khi cần.

## Ngôn ngữ & môi trường
- Ngôn ngữ: **Python**.
- Luôn dùng môi trường ảo `.venv` (không cài thư viện vào Python hệ thống).
- Quản lý phụ thuộc qua `requirements.txt`.
- Hệ điều hành phát triển: Windows (dùng lệnh PowerShell/CMD phù hợp).

### Lệnh thường dùng (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app.cli ask "..."      # RAG
python -m app.cli agent "..."     # Agent dùng tool
python -m app.cli eval            # Chạy eval
```

### Bật LLM thật
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:LLM_PROVIDER   = "openai"    # mặc định "mock"
```

## Cấu trúc code (`app/`)
| File | Vai trò |
|------|---------|
| `app/llm.py` | Lớp trừu tượng gọi model: mock hoặc OpenAI. |
| `app/embeddings.py` | Biến chữ thành vector. |
| `app/vectorstore.py` | Lưu vector + tìm đoạn liên quan. |
| `app/rag.py` | Chunk tài liệu → tìm → nhét prompt → hỏi model. |
| `app/tools.py` | Các hàm agent được phép gọi. |
| `app/agent.py` | Vòng lặp: chọn tool → thực thi → trả lời. |
| `app/eval.py` | Bộ câu hỏi + đáp án kỳ vọng, chấm điểm tự động. |
| `app/cli.py` | Điểm vào dòng lệnh. |

Dữ liệu mẫu: `data/company_policy.md`.

## Quy ước làm việc
- Giữ luồng mock luôn chạy được, không phụ thuộc bắt buộc vào API key.
- Trả lời và tài liệu bằng tiếng Việt.
- Tôn trọng lớp trừu tượng `llm.py`: thêm provider mới thì mở rộng ở đó, không rải rác lời gọi model khắp nơi.
