# AI Starter — RAG + Agent + Eval (Python)

Một project học việc để hiểu **bản chất công việc AI Engineer**. Nó ghép 3 khối cốt lõi:

1. **RAG** (Retrieval-Augmented Generation) — cho LLM trả lời dựa trên tài liệu riêng của bạn.
2. **Agent** — cho LLM *hành động*: gọi công cụ (tool) để làm việc, không chỉ trả lời.
3. **Eval** — đo chất lượng câu trả lời một cách có hệ thống (thứ phân biệt người làm AI nghiêm túc với người nghịch cho vui).

Toàn bộ chạy được **ngay lập tức không cần API key** ở chế độ mock, để bạn thấy luồng hoạt động. Khi sẵn sàng, cắm OpenAI vào bằng 1 biến môi trường.

---

## Chạy nhanh (không cần API key)

```powershell
# 1. Tạo môi trường ảo và cài thư viện
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Hỏi đáp trên tài liệu mẫu (RAG) — chế độ mock
python -m app.cli ask "Chính sách nghỉ phép của công ty thế nào?"

# 3. Cho agent tự dùng công cụ
python -m app.cli agent "Tính giúp 15% của 2 triệu rồi cho biết chính sách bảo hành"

# 4. Chạy eval để đo chất lượng
python -m app.cli eval
```

## Bật LLM thật (OpenAI)

```powershell
$env:OPENAI_API_KEY = "sk-..."      # key của bạn
$env:LLM_PROVIDER   = "openai"       # mặc định là "mock"
python -m app.cli ask "Chính sách nghỉ phép thế nào?"
```

---

## Kiến trúc — đọc theo thứ tự này để hiểu nghề

| File | Khối kiến thức | Vai trò |
|------|----------------|---------|
| `app/llm.py` | Gọi model | Lớp trừu tượng: mock hoặc OpenAI. Phần "dev thuần". |
| `app/embeddings.py` | Embedding | Biến chữ thành vector để so sánh ngữ nghĩa. |
| `app/vectorstore.py` | Vector search | Lưu vector + tìm đoạn liên quan nhất. |
| `app/rag.py` | **RAG** | Ghép: chunk tài liệu → tìm → nhét vào prompt → hỏi model. |
| `app/tools.py` | Công cụ cho agent | Các hàm agent được phép gọi (máy tính, tra cứu). |
| `app/agent.py` | **Agent** | Vòng lặp: model quyết định gọi tool nào → thực thi → trả lời. |
| `app/eval.py` | **Eval** | Bộ câu hỏi + đáp án kỳ vọng, chấm điểm tự động. |
| `app/cli.py` | Giao diện | Điểm vào dòng lệnh cho 3 khối trên. |

Đọc `docs/LEARNING_PATH.md` để có lộ trình học kèm giải thích từng khối.
