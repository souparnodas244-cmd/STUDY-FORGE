# StudyForge — AI Study Agent

## Stack
- **Backend**: Python + FastAPI
- **AI**: Google Gemini (gemini-2.5-flash)
- **Vector DB**: FAISS (in-memory RAG)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **Frontend**: Vanilla HTML

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

Notes:
- `faiss-cpu==1.8.0` is used with `numpy==1.26.4` for compatibility.
- Recommended Python for this project: `3.10` or `3.11`.

### 2. Set your Gemini API key
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend
Open `frontend/index.html` directly in your browser (no server needed).

---

## Windows Quick Start (PowerShell)

```powershell
cd C:\Users\Souparno\ai_code\agent\STUDYFORGE\backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY="your_gemini_api_key_here"
uvicorn main:app --reload --port 8000
```

Health check:
- Open `http://127.0.0.1:8000/health`
- Expected response:
  - `status: ok`
  - `model: gemini-2.5-flash`

Notes:
- A `404` on `/favicon.ico` is normal and does not indicate backend failure.

---

## API

### POST /analyze
```json
{ "syllabus": "your notes or syllabus text..." }
```
Returns: topic analysis + exam/viva/MCQ questions + structured study plan

### POST /evaluate
```json
{
  "question": "What is a binary search tree?",
  "user_answer": "A BST is a tree where left < root < right..."
}
```
Returns: score, correctness, missing points, improvement

### GET /health
Returns index status.

---

## Architecture

```
User Input (Syllabus)
       │
       ▼
  chunk_text()          ← splits into overlapping 400-char chunks
       │
       ▼
  SentenceTransformer   ← all-MiniLM-L6-v2 embeds locally
       │
       ▼
  FAISS IndexFlatIP     ← cosine similarity index (in-memory)
       │
  ┌────┴─────────────────────────────┐
  │                                  │
  ▼                                  ▼
retrieve("important topics")    retrieve("exam questions on X")
  │                                  │
  ▼                                  ▼
Gemini /analyze (SYSTEM prompt)  Gemini /questions (SYSTEM prompt)
  │                                  │
  ▼                                  ▼
topics JSON                     questions JSON
  │
  ▼
Gemini /plan → study_plan JSON
```

Every output is structured JSON — no raw text paragraphs.
