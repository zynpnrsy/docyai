DOCY AI — PDF-Based Multi-Agent Learning Assistant
====================================================

Live demo (HuggingFace Spaces):
https://huggingface.co/spaces/zeyneppinarsoy/docyai/tree/main


OVERVIEW
--------
An AI-powered PDF assistant built as a multi-agent system. Users upload any PDF
and can chat with it, generate flashcards, take MCQ quizzes, and monitor answer
quality — all grounded strictly in the uploaded document.


MULTI-AGENT ARCHITECTURE
-------------------------
All agents are defined in agents.py and orchestrated by CoordinatorAgent.

  CoordinatorAgent          — routes every request through the full pipeline
    ├── RetrieverAgent      — ChromaDB vector search (embedding similarity)
    ├── AnswerAgent         — Groq LLM, answers grounded only in PDF context
    ├── StudyAgent          — flashcard and MCQ quiz generation (structured JSON)
    ├── RiskMonitorAgent    — unsafe content, hallucination, and bias detection
    └── EvaluationAgent     — per-query metrics (confidence, latency, fallback rate)

Pipeline per chat query:
  PDF → Chunk → Embed → ChromaDB → Retrieve → LLM Answer
      → Confidence Score → Risk Assessment → Evaluation Record


FEATURES
--------
  Chat Assistant     Ask questions; get PDF-grounded answers with confidence score
                     and colour-coded risk panel (low / medium / high)
  Flashcard Study    5 AI-generated flashcards from document content
  MCQ Quiz           5 or 10 multiple-choice questions; topic-focused retrieval
  Evaluation Dashboard  Session metrics: avg confidence, latency, fallback rate,
                        risk distribution (refreshed on demand)
  Feedback System    1–5 star rating saved to Google Sheets


MONITORING & RISK MANAGEMENT
------------------------------
RiskMonitorAgent runs three checks on every answer:

  1. Unsafe content   — keyword detection (violence, exploitation, radicalization…)
  2. Hallucination    — heuristic: answer/context vocabulary overlap < 35% → flag
  3. Bias language    — generalising phrases ("all people", "inherently", etc.)

Risk level (low / medium / high) and any flags are shown inline in the UI.
All agent activity is logged to docai_agent.log (INFO level).


EVALUATION
-----------
Intrinsic:
  Confidence score = cosine similarity (question ↔ retrieved chunks)
                   + LLM self-evaluation (0–100), averaged.
  Shown per query; accumulated across the session.

Extrinsic:
  Fallback rate     — % of queries the PDF did not contain an answer for
  Answer latency    — ms from question to LLM response
  Retrieval latency — ms for ChromaDB vector search
  Risk distribution — count of low / medium / high queries

All metrics visible in the Evaluation Dashboard tab.


TECH STACK
----------
  Multi-agent framework : custom (agents.py)
  LLM                   : Groq API — llama-3.1-8b-instant
  Embeddings            : Sentence Transformers — all-MiniLM-L6-v2
  Vector Database       : ChromaDB (EphemeralClient, in-memory)
  UI                    : Gradio (Blocks)
  Feedback storage      : Google Sheets API (gspread)


SETUP
-----
1. Install dependencies:
     pip install -r requirements.txt

2. Create a .env file with:
     GROQ_API_KEY=<your Groq API key>
     GOOGLE_CREDENTIALS=<full JSON string of GCP service account>
     GOOGLE_SHEET_NAME=DocAI Feedback

   Note: GOOGLE_CREDENTIALS is the entire credentials.json content as a
   single-line string, NOT a file path.

3. Create a Google Sheet named "DocAI Feedback" and share it with the
   service account email from your credentials JSON.

4. Run:
     python app.py
   App starts at http://0.0.0.0:7860
