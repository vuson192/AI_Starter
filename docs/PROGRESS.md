# Tiến độ học — AI Starter

File này ghi lại đã học tới đâu để lần sau tiếp tục không bị mất mạch.

---

## Cập nhật gần nhất: 2026-09-14

### Đã làm được trong buổi này
1. **Dựng môi trường**: cài Python 3.13.15, tạo `.venv`, cài dependencies (`numpy`, `openai`).
2. **Chạy thử cả 3 khối ở chế độ mock** — tất cả hoạt động:
   - `ask` (RAG): tìm đúng đoạn chính sách nghỉ phép.
   - `agent`: tự chọn tool `calculator`, tính 15% của 2.000.000 = 300.000.
   - `eval`: 4/4 pass, score 1.0.
3. **Sửa 1 bug thật**: project crash trên Windows với `UnicodeEncodeError` khi in tiếng Việt
   (console dùng codepage cp1252). Đã thêm đoạn ép `stdout`/`stderr` sang UTF-8 ở đầu
   `app/cli.py` — giờ chạy được ở mọi console, không cần set biến môi trường.

### Đã đọc / đã hiểu (lý thuyết)
Học theo thứ tự các khối, đã đọc tới:

- [x] **1. LLM & lớp trừu tượng** (`llm.py`)
  - `chat(messages)` bọc model sau interface chung để phần còn lại không cần biết dùng model nào.
  - `messages` gồm role `system` / `user` / `assistant`; `system` là chỗ "lập trình hành vi" model bằng ngôn ngữ tự nhiên.
- [x] **2. Embedding** (`embeddings.py`)
  - Biến chữ thành vector; hai đoạn gần nghĩa → vector gần nhau.
  - Đo độ giống nhau bằng cosine similarity (0→1).
- [x] **3. Vector store + Retrieval** (`vectorstore.py`)  ← **DỪNG Ở ĐÂY**
  - Lưu (text, vector); khi có câu hỏi thì embed câu hỏi rồi tìm đoạn gần nhất (`search`, `top_k`).
  - Bản trong project duyệt hết mọi doc (brute force).
  - Production dùng Pinecone / pgvector / Qdrant với thuật toán ANN (approximate nearest
    neighbor) để tìm nhanh trên hàng triệu vector.

### Lần sau đọc tiếp từ
- [ ] **4. RAG** (`rag.py`) — 4 bước: chunk → index → retrieve → generate.
- [ ] **5. Agent + Tools** (`agent.py`, `tools.py`) — model quyết định gọi tool, code thực thi.
- [ ] **6. Eval** (`eval.py`) — chấm điểm câu trả lời một cách có hệ thống.

---

## Lộ trình học (xếp theo độ khó tăng dần)

1. **Chunking & retrieval tốt hơn** — thêm overlap giữa các chunk, thử các `top_k` khác nhau, đo tác động lên eval.
2. **Agent nhiều bước (multi-step / ReAct)** — nâng `agent.py` từ 1 bước lên vòng lặp: model gọi nhiều tool liên tiếp, đọc kết quả rồi quyết định bước tiếp.
3. **Eval xịn hơn** — đo tỉ lệ bịa, đo retrieval hit-rate, tách điểm retrieval khỏi điểm generation.
4. **Prompt engineering có kỷ luật** — chỉnh `system` prompt, chạy eval trước/sau để thấy con số thay đổi.
5. **Cắm model thật** — chạy Ollama (local) hoặc OpenAI, so eval giữa mock và model thật.
6. **Quan sát & log (observability)** — ghi lại mỗi lần chạy: câu hỏi, đoạn lấy được, prompt, câu trả lời.

---

## Bài tập luyện (làm theo thứ tự, dùng eval làm thước đo)

- **Bài 1 (dễ, khởi động):** Thêm 3-4 test case mới vào `eval.py` cho các mục chính sách chưa được
  test (vd "chuyển tối đa mấy ngày phép", "giờ bắt đầu làm việc sớm nhất"). Quen vòng lặp thêm test → chạy → xem pass/fail.
- **Bài 2 (retrieval):** Thêm `overlap` vào `chunk_text`. Chạy eval trước và sau, giải thích score đổi hay không và tại sao.
- **Bài 3 (agent, bài lớn nhất):** Nâng `agent.py` xử lý được yêu cầu cần 2 tool:
  "Tính 15% của 2 triệu rồi cho biết chính sách bảo hành". Cần vòng lặp gọi nhiều tool rồi tổng hợp.
- **Bài 4 (eval nâng cao):** Thêm chỉ số "faithfulness" — kiểm tra câu trả lời có thông tin nằm ngoài ngữ cảnh không (dấu hiệu bịa).
- **Bài 5 (model thật):** Cài Ollama, kéo model nhỏ (vd `qwen3:8b`), chạy eval với `LLM_PROVIDER=ollama`, so với mock.

---

## Ghi chú môi trường (Windows)
- Máy dùng Python cài ở `%LOCALAPPDATA%\Programs\Python\Python313`. Nếu gõ `python` chưa nhận,
  mở terminal mới hoặc dùng `.\.venv\Scripts\python.exe`.
- Cách chạy chuẩn sau khi activate venv:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python -m app.cli ask "câu hỏi"
  python -m app.cli agent "yêu cầu"
  python -m app.cli eval
  ```
