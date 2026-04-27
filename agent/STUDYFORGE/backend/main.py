"""
StudyForge AI Study Agent — Backend
Stack: FastAPI + FAISS + Sentence-Transformers + Google Gemini API
"""

import json, re, textwrap
from typing import Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4
GEMINI_MODEL  = "gemini-2.5-flash"

# Load API key from environment variable — set before starting uvicorn:
# PowerShell: $env:GEMINI_API_KEY="AIza...your_new_key"
import os
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Run: $env:GEMINI_API_KEY='your_key_here'")
genai.configure(api_key=GEMINI_KEY)

# ─── Globals ──────────────────────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
index    = None
chunks   = []

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="StudyForge API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzeRequest(BaseModel):
    syllabus: str

class EvaluateRequest(BaseModel):
    question: str
    user_answer: str
    context: Optional[str] = ""

# ── NEW: Performance Improvement Mode ─────────────────────────────────────────
class ImproveRequest(BaseModel):
    subject: str
    current_score: float
    target_score: float
    weak_topics: list[str]

# ─── RAG ──────────────────────────────────────────────────────────────────────
def chunk_text(text):
    words = text.split()
    result, i = [], 0
    while i < len(words):
        result.append(" ".join(words[i:i+CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return result or [text]

def build_faiss(text):
    global index, chunks
    chunks = chunk_text(text)
    vecs   = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    index  = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return len(chunks)

def retrieve(query, k=TOP_K):
    if index is None or not chunks: return ""
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    _, ids = index.search(q, min(k, len(chunks)))
    return "\n\n".join(chunks[i] for i in ids[0] if i != -1)

# ─── Gemini ───────────────────────────────────────────────────────────────────
def call_gemini(system, user):
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    return model.generate_content(user).text.strip()

def safe_json(text):
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
    if m: return json.loads(m.group(1))
    raise ValueError(f"No JSON found:\n{text[:300]}")

def parse_model_json(system: str, user: str, stage: str):
    try:
        return safe_json(call_gemini(system, user))
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=f"{stage}: model returned invalid JSON. {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"{stage}: model call failed. {str(e)}"
        ) from e

# ─── Prompts (original — untouched) ───────────────────────────────────────────
SYS_ANALYZE = """You are StudyForge academic AI. Extract important topics from syllabus.
Return ONLY valid JSON, no prose, no fences:
{"topics":[{"name":"...","difficulty":"easy|medium|hard","exam_probability":"low|medium|high","summary":"one sentence"}]}"""

SYS_QUESTIONS = """You are StudyForge exam question generator. Use provided context only.
Return ONLY valid JSON, no prose, no fences. Generate exactly 10 exam, 5 viva, 5 MCQs:
{"exam_questions":["Q1...","Q2...",...10 items],"viva_questions":["V1...",...5 items],"mcqs":[{"question":"...","options":["A)...","B)...","C)...","D)..."],"answer":"A"},...5 items]}"""

SYS_PLAN = """You are StudyForge study planner. Build structured plan from topic metadata.
Return ONLY valid JSON, no prose, no fences:
{"priority_topics":["t1","t2","t3"],"study_plan":[{"topic":"...","time":"X hours","focus":"..."}],"confidence":0.85,"tips":["tip1","tip2","tip3"]}"""

SYS_EVAL = """You are StudyForge answer evaluator. Evaluate student answer vs question and context.
Return ONLY valid JSON, no prose, no fences:
{"score":7,"max_score":10,"correctness":"correct|partial|incorrect","covered_points":["..."],"missing_points":["..."],"improvement":"...","model_answer_snippet":"..."}"""

# ── NEW PROMPTS ────────────────────────────────────────────────────────────────
SYS_IMPROVE = """You are StudyForge Performance Coach. Analyse a student's score gap and weak topics.
Return ONLY valid JSON, no prose, no markdown fences. Be concise and actionable.
Rules for daily_plan method field:
- Practice-heavy topics (numericals, algorithms, problems): assign "Pomodoro" as method
- Theory-heavy topics (concepts, definitions, explanations): assign "SQ3R" as method
Output schema:
{
  "gap_analysis": [
    {"weakness_type": "conceptual_gap|lack_of_practice|retention_issue", "topic": "...", "reason": "one line"}
  ],
  "improvement_strategy": ["action step 1", "action step 2"],
  "daily_plan": [
    {"day": 1, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 2, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 3, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 4, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 5, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 6, "focus": "topic name", "method": "Pomodoro|SQ3R"},
    {"day": 7, "focus": "topic name", "method": "Pomodoro|SQ3R"}
  ],
  "expected_score_range": "XX-YY",
  "confidence": 0.0,
  "predicted_score_range": "XX-YY",
  "justification": "one to two line explanation",
  "study_techniques": {
    "pomodoro": {
      "work": "25 min",
      "break": "5 min",
      "usage": "use for practice-heavy topics and numericals"
    },
    "sq3r": {
      "steps": ["Survey", "Question", "Read", "Recite", "Review"],
      "usage": "use for theory-heavy topics and concepts"
    }
  }
}"""

SYS_RECALL = """You are StudyForge Active Recall Generator. Generate exam-oriented questions for weak topics.
Avoid generic textbook phrasing. Test deep understanding.
Return ONLY valid JSON, no prose, no markdown fences:
{
  "active_recall": {
    "short_questions": ["q1","q2","q3","q4","q5"],
    "conceptual_questions": ["q1","q2","q3","q4","q5"],
    "numerical_questions": ["q1","q2","q3"]
  }
}
If the subject has no numerical component, set numerical_questions to []."""

# ── NEW: YouTube Search Link Builder (no API needed) ──────────────────────────
def build_youtube_links(subject: str, weak_topics: list[str]) -> list[dict]:
    import urllib.parse
    videos = []
    for topic in weak_topics:
        query = f"{subject} {topic} explained exam"
        url   = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
        videos.append({"topic": topic, "search_query": query, "url": url})
    return videos

# ─── Endpoints (original — untouched) ─────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.syllabus.strip():
        raise HTTPException(400, "Syllabus required.")

    n = build_faiss(req.syllabus)

    ctx1 = retrieve("important topics difficulty exam probability")
    topics_data = parse_model_json(
        SYS_ANALYZE,
        f"CONTEXT:\n{ctx1}\n\nFULL SYLLABUS:\n{req.syllabus[:3000]}",
        "Topic analysis"
    )

    names = ", ".join(t["name"] for t in topics_data.get("topics", [])[:6])
    ctx2  = retrieve(f"exam viva questions {names}")
    questions_data = parse_model_json(
        SYS_QUESTIONS,
        f"TOPICS: {names}\n\nCONTEXT:\n{ctx2}",
        "Question generation"
    )

    plan_data = parse_model_json(
        SYS_PLAN,
        f"TOPICS:\n{json.dumps(topics_data['topics'], indent=2)}",
        "Study plan generation"
    )

    return {"status":"success","chunks_indexed":n,
            "topic_analysis":topics_data,"questions":questions_data,"study_plan":plan_data}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    ctx = retrieve(req.question) if index is not None else req.context
    eval_data = parse_model_json(
        SYS_EVAL,
        f"QUESTION: {req.question}\n\nSTUDENT ANSWER: {req.user_answer}\n\nCONTEXT:\n{ctx or 'None.'}",
        "Answer evaluation"
    )
    return {"status":"success","evaluation":eval_data}


# ── NEW ENDPOINT ───────────────────────────────────────────────────────────────
@app.post("/improve")
async def improve(req: ImproveRequest):
    if not req.weak_topics:
        raise HTTPException(400, "At least one weak topic required.")
    if req.target_score <= req.current_score:
        raise HTTPException(400, "target_score must be greater than current_score.")

    gap        = req.target_score - req.current_score
    topics_str = ", ".join(req.weak_topics)

    # Core improvement analysis
    improve_data = parse_model_json(
        SYS_IMPROVE,
        (
            f"SUBJECT: {req.subject}\n"
            f"CURRENT SCORE: {req.current_score}\n"
            f"TARGET SCORE: {req.target_score}\n"
            f"SCORE GAP: {gap:.1f} points\n"
            f"WEAK TOPICS: {topics_str}"
        ),
        "Improvement analysis"
    )

    # Active recall questions
    recall_data = parse_model_json(
        SYS_RECALL,
        f"SUBJECT: {req.subject}\nWEAK TOPICS: {topics_str}",
        "Active recall generation"
    )

    # YouTube links (pure Python, zero external API)
    videos = build_youtube_links(req.subject, req.weak_topics)

    return {
        "status":                "success",
        "subject":               req.subject,
        "score_gap":             gap,
        "gap_analysis":          improve_data.get("gap_analysis", []),
        "improvement_strategy":  improve_data.get("improvement_strategy", []),
        "daily_plan":            improve_data.get("daily_plan", []),
        "expected_score_range":  improve_data.get("expected_score_range", ""),
        "confidence":            improve_data.get("confidence", 0.0),
        "predicted_score_range": improve_data.get("predicted_score_range", ""),
        "justification":         improve_data.get("justification", ""),
        "active_recall":         recall_data.get("active_recall", {}),
        "videos":                videos,
        "study_techniques":      improve_data.get("study_techniques", {}),
    }


@app.get("/health")
def health():
    return {"status":"ok","model":GEMINI_MODEL,"chunks_indexed":len(chunks)}
