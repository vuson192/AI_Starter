# Tiến độ học — AI Starter

> Ghi lại buổi học để lần sau mở ra là chiến tiếp được ngay.

## Đã hiểu (buổi 1)

### Nền tảng
- **Python** cài trên máy chỉ là interpreter + pip, KHÔNG kèm IDE. IDE (VS Code / Kiro / PyCharm) là phần mềm riêng.
- **Kiro dùng model của Anthropic ở sau, nhưng KHÁC Claude Code.** Trả tiền là trả cho công cụ/trải nghiệm, không phải cho model. Chọn công cụ hợp cách làm việc.

### Đọc code `app/llm.py`
- `self` = chính object đang gọi method; Python tự truyền vào.
- `chat()` là **dispatcher**: đọc env `LLM_PROVIDER` rồi gọi `self._chat_openai / _chat_ollama / _chat_mock`. Gọi qua `self.` vì là method cùng class.
- `OpenAI()` = class từ thư viện `openai` (cài qua pip). `ollama_host()` = hàm tự viết trong file, gọi không cần `self.`.
- Quy tắc phân biệt: `self.x()` = method của class | `x()` = hàm tự do/import | `X()` = tạo object từ class.

### messages & role
- `messages` = list các dict `{role, content}` = lịch sử hội thoại đưa cho model.
- 3 role: **system** (dev đặt luật + ngữ cảnh), **user** (câu hỏi), **assistant** (model trả lời).
- `system` là nơi "lập trình hành vi bằng ngôn ngữ tự nhiên" — chứa CẢ luật CẢ tài liệu tham khảo.

### Prompt engineering (system message)
- Rule trong `system` do **AI Engineer viết tay** (`rag.py`, `agent.py`).
- Rule lấy từ: (1) nghiệp vụ, (2) ràng buộc code phía sau cần, (3) điểm yếu model (hay bịa), (4) **kết quả eval**.
- Nguyên tắc: gán vai trò rõ, cụ thể không mơ hồ, ép định dạng output, đặt guardrail (không bịa), có đường lui (nói không biết), luật trước - dữ liệu sau.

### Eval (`app/eval.py`)
- Không dùng `assert x == 5` vì output model không cố định.
- Chấm bằng **keyword matching**: bộ `TEST_CASES` có sẵn (câu hỏi + `expect_any`), kiểm tra câu trả lời có chứa từ khóa đúng không → ra score.
- **Con người (AI Engineer) sinh bộ test** = "golden dataset": đọc tài liệu → đặt câu hỏi → ghi đáp án đúng. `eval.py` KHÔNG tự đọc tài liệu, chỉ cầm đáp án đóng băng.
- Hạn chế: đúng số nhưng sai ngữ cảnh vẫn pass; không đo chất lượng diễn đạt. Production dùng LLM chấm LLM, semantic similarity...

### Embedding (`app/embeddings.py`)
- Embedding = biến chữ thành **vector (dãy số)**. Câu gần nghĩa → vector gần nhau.
- Mock dùng **bag-of-words hashing**: mỗi từ hash vào 1 trong 256 ô, đếm lên. Chỉ giống khi TRÙNG TỪ (không hiểu nghĩa "ô tô" = "xe hơi").
- Embedding thật (OpenAI/Ollama) học từ dữ liệu khổng lồ → hiểu nghĩa thật.
- **Cosine similarity** = đo độ giống, ra số 0→1. Bản chất: `(số từ chung) / (√số từ câu A × √số từ câu B)`. Câu dài lan man bị "phạt" ở mẫu số.
- Đã chạy demo thấy: câu gần nghĩa điểm cao (0.75), câu khác chủ đề thấp (0.32), không liên quan = 0.

### Vector store (`app/vectorstore.py`)
- 2 việc: `add()` lưu (text + vector) vào `Doc`; `search()` embed câu hỏi → so cosine với mọi doc → lấy `top_k` gần nhất.
- Bản này là **list trong RAM** (mất khi tắt), chưa phải DB thật. Production: Pinecone/pgvector/Qdrant.
- Bản này **brute force** (so từng doc). Production dùng **ANN** (xấp xỉ, nhanh trên hàng triệu vector).
- Vector và text đi kèm nhau trong `Doc` — tìm thấy vector là có text luôn.

### RAG (`app/rag.py`) — 4 bước
1. **Chunk**: cắt tài liệu dài thành đoạn nhỏ.
2. **Index**: embed từng chunk, lưu store.
3. **Retrieve**: embed câu hỏi → tìm đoạn liên quan (`search`).
4. **Generate**: nhét đoạn tìm được vào `system` (ngữ cảnh) + câu hỏi vào `user` → model trả lời.
- **Quan trọng**: đoạn tìm được là **nguyên liệu đầu vào**, KHÔNG phải output. Model đọc nó rồi CHẮT ra câu trả lời gọn.
- Đưa cả tài liệu (system) VÀ câu hỏi (user) vào model — thiếu câu hỏi thì model không biết trả lời gì.
- Vector chỉ dùng ở bước Retrieve. Lúc Generate model chỉ thấy CHỮ, không thấy vector.
- NotebookLM của Google = RAG (có cả Enterprise API). Học nguyên lý này vẫn giá trị: nhiều DN cần tự xây (data nhạy cảm, tích hợp riêng, tránh vendor lock-in).

### Cách model hoạt động (đào sâu)
- Mạng nơ-ron = các **ma trận số (weights)** + phép **nhân ma trận** qua nhiều tầng. "Tham số" = từng con số trong ma trận.
- **Train** = chỉnh dần các con số cho tới khi đoán đúng. Học xong dữ liệu train bị vứt, kiến thức "ngấm" vào weights.
- **Chat trơn** = model NHỚ LẠI/tính tại chỗ từ weights (KHÔNG tra kho vector nào). Chỉ RAG mới có kho vector để tra.
- Model sinh **từng chữ một**: mỗi bước ra **bảng xác suất cho từ tiếp theo** → chọn từ → lặp lại. (Vì sao chữ hiện từ từ; vì sao hỏi lại ra khác — do temperature.)
- Model học bằng "đoán từ tiếp theo" nhưng để đoán đúng buộc phải học ngầm kiến thức/logic → trí thông minh **trồi lên (emergence)**.
- Model KHÔNG thật sự hiểu, tối ưu độ MƯỢT không phải độ THẬT → hay **bịa (hallucinate)**. Đây chính là lý do cần RAG.

### Agent + Tools (`app/agent.py`, `app/tools.py`)
- **Tool** = hàm Python thật + tên + mô tả (`desc` để model biết khi nào dùng).
- **Agent** = vòng lặp: đưa câu hỏi + danh sách tool → model trả **JSON quyết định** gọi tool nào + tham số → **code thực thi** → trả kết quả.
- Ranh giới sống còn: **model NGHĨ (chọn tool, điền tham số), code LÀM (thực thi).** Model chỉ đề xuất, không tự tay chạy.
- **An toàn**: calculator KHÔNG dùng `eval()` mà parse **AST + danh sách trắng phép toán**. Nguyên tắc: KHÔNG BAO GIỜ tin chuỗi model sinh ra.
- Bản này chỉ **1 bước**. Agent thật (ReAct, LangChain) lặp nhiều bước + xử lý lỗi, nhưng lõi giống hệt.
- **Khi nào dùng agent**: khi cần HÀNH ĐỘNG (không chỉ trả lời). Chuỗi nhiều bước phụ thuộc nhau là trường hợp mạnh nhất. Nhớ: RAG = "cho tôi biết", Agent = "làm giúp tôi".
- Đã chạy thật: "20% của 5000" → chọn `calculator` → ra 1000. "chính sách nghỉ phép" → chọn `knowledge_lookup` (nhưng lấy sai đoạn do embedding mock quá thô — minh họa vì sao cần embedding thật + eval).

## Bức tranh tổng
Bộ ba của project: **RAG (trả lời dựa tài liệu) — Agent (hành động qua tool) — Eval (đo chất lượng)**.
Luồng RAG: `chunk → embed → lưu store → (hỏi) embed câu hỏi → search top_k → nhét vào prompt → model gen`.

## Lệnh chạy (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m app.cli ask "câu hỏi"      # RAG
.\.venv\Scripts\python.exe -m app.cli agent "yêu cầu"     # Agent
.\.venv\Scripts\python.exe -m app.cli eval                # Eval
```

## Lần sau làm gì (TODO)
- [ ] Bật **model thật local bằng Ollama** (không cần API key) chạy lại để thấy embedding/retrieve tốt hơn mock.
- [ ] Chạy `eval` và thử sửa prompt trong `rag.py` xem score đổi thế nào.
- [ ] Thử thêm 1 case mới vào `TEST_CASES` (vd hỏi giờ làm việc — đáp án "8").
- [ ] Đọc kỹ `rag.py` phần `chunk_text` và `build_store` (chưa mở chi tiết).
- [ ] Tìm hiểu Transformer / attention (cơ chế bên trong model — mới chỉ chạm bề mặt).
