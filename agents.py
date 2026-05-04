"""
agents.py – Multi-agent framework for DocAI.

Agent hierarchy:
  CoordinatorAgent          orchestrates the full pipeline for every request
    ├── RetrieverAgent      ChromaDB vector search
    ├── AnswerAgent         LLM-grounded question answering
    ├── StudyAgent          flashcard and MCQ quiz generation
    ├── RiskMonitorAgent    hallucination, unsafe-content, and bias detection
    └── EvaluationAgent     per-query metrics accumulation and reporting
"""

import json
import logging
import re
import time

logging.basicConfig(
    filename="docai_agent.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(f"docai.{name}")

    def _log(self, msg: str, level: str = "info"):
        getattr(self._logger, level)(msg)


# ---------------------------------------------------------------------------
# RetrieverAgent
# ---------------------------------------------------------------------------

class RetrieverAgent(BaseAgent):
    """Retrieves semantically relevant chunks from a ChromaDB collection."""

    def __init__(self, embedding_model, chroma_client):
        super().__init__("RetrieverAgent")
        self.embedding_model = embedding_model
        self.chroma_client = chroma_client

    def retrieve(self, query: str, collection_name: str, k: int = 3) -> dict:
        start = time.time()
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            if count == 0:
                return {"chunks": [], "distances": [], "latency_ms": 0.0}
            query_emb = self.embedding_model.encode([query])
            results = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=min(k, count),
            )
            chunks = results["documents"][0]
            distances = results["distances"][0]
        except Exception as exc:
            self._log(f"Retrieval failed: {exc}", "error")
            chunks, distances = [], []

        latency_ms = (time.time() - start) * 1000
        self._log(f"Retrieved {len(chunks)} chunks in {latency_ms:.1f}ms")
        return {"chunks": chunks, "distances": distances, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# AnswerAgent
# ---------------------------------------------------------------------------

class AnswerAgent(BaseAgent):
    """Generates context-grounded answers using an LLM."""

    def __init__(self, groq_client, model_name: str):
        super().__init__("AnswerAgent")
        self.client = groq_client
        self.model_name = model_name

    def answer(self, context: str, question: str) -> dict:
        start = time.time()
        prompt = (
            "You are a strict academic assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, respond exactly with:\n"
            "This PDF does not contain this information.\n"
            "Do not use outside knowledge.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer_text = response.choices[0].message.content.strip()
        latency_ms = (time.time() - start) * 1000
        self._log(f"Answer generated in {latency_ms:.1f}ms")
        return {"answer": answer_text, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# StudyAgent
# ---------------------------------------------------------------------------

class StudyAgent(BaseAgent):
    """Generates flashcards and MCQ quizzes from document context."""

    def __init__(self, groq_client, model_name: str):
        super().__init__("StudyAgent")
        self.client = groq_client
        self.model_name = model_name

    def generate_flashcards(self, context: str) -> list:
        prompt = (
            "You are an expert educator. Based ONLY on the provided context, "
            "generate 5 high-quality flashcards.\n"
            "Each flashcard must have a 'question' and an 'answer'.\n\n"
            'Format EXACTLY as JSON: [{"question": "...", "answer": "..."}, ...]\n\n'
            f"Context:\n{context}\n\nJSON Output:"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        try:
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            cards = json.loads(content)
            if isinstance(cards, dict) and "flashcards" in cards:
                return cards["flashcards"]
            return cards
        except Exception as exc:
            self._log(f"Flashcard parse error: {exc}", "error")
            return [{"question": "Error", "answer": "Could not generate cards. Please try again."}]

    def generate_quiz(self, context: str, num_questions: int = 5, topic: str = None) -> list:
        topic_instruction = (
            f"Focus the questions specifically on: {topic}"
            if topic else
            "Cover the most important concepts from the context"
        )
        prompt = (
            f"You are an expert quiz creator. Based ONLY on the context, generate exactly "
            f"{num_questions} multiple-choice questions.\n{topic_instruction}\n\n"
            "Rules:\n- Each question MUST have exactly 4 options\n"
            "- Exactly ONE option must be correct\n"
            "- Provide a brief explanation referencing the context for each correct answer.\n\n"
            'Format EXACTLY as: {"questions": [{"question":"...","options":["...","...","...","..."],'
            '"correct_index":0,"explanation":"..."}]}\n\n'
            f"Context:\n{context}\n\nJSON Output:"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        try:
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            data = json.loads(content)
            questions = data.get("questions", data if isinstance(data, list) else [])
            valid = [
                q for q in questions
                if isinstance(q, dict)
                and all(k in q for k in ["question", "options", "correct_index", "explanation"])
                and isinstance(q["options"], list) and len(q["options"]) == 4
                and isinstance(q["correct_index"], int) and 0 <= q["correct_index"] <= 3
                and str(q["question"]).strip()
                and all(str(o).strip() for o in q["options"])
            ]
            return valid or [{
                "question": "Could not generate valid questions. Try again.",
                "options": ["Retry", "Retry", "Retry", "Retry"],
                "correct_index": 0,
                "explanation": "Validation failed.",
            }]
        except Exception as exc:
            self._log(f"Quiz parse error: {exc}", "error")
            return [{
                "question": "Error generating quiz",
                "options": ["A", "B", "C", "D"],
                "correct_index": 0,
                "explanation": str(exc),
            }]


# ---------------------------------------------------------------------------
# RiskMonitorAgent
# ---------------------------------------------------------------------------

class RiskMonitorAgent(BaseAgent):
    """
    Monitors agent outputs for three risk categories:
      1. Unsafe content  – keyword-based detection of harmful topics
      2. Hallucination   – heuristic: answer vocabulary poorly overlaps context
      3. Bias language   – absolute/generalising phrasing
    """

    _UNSAFE_KEYWORDS = [
        "kill", "murder", "suicide", "bomb", "weapon", "hack", "exploit",
        "radicalize", "extremist", "terrorist", "violence", "self-harm",
        "how to make", "instructions for", "step by step to",
    ]
    _BIAS_PHRASES = [
        "all people", "every person", "always true", "inherently",
        "by nature", "they all", "these people", "you people",
    ]

    def __init__(self):
        super().__init__("RiskMonitorAgent")

    def assess(self, question: str, answer: str, context: str) -> dict:
        flags = []
        combined_lower = (question + " " + answer).lower()

        # 1. Unsafe content check
        unsafe_hits = [kw for kw in self._UNSAFE_KEYWORDS if kw in combined_lower]
        if unsafe_hits:
            flags.append(f"Unsafe keywords detected: {unsafe_hits}")

        # 2. Bias language check
        bias_hits = [p for p in self._BIAS_PHRASES if p in combined_lower]
        if bias_hits:
            flags.append(f"Potential bias language: {bias_hits}")

        # 3. Hallucination heuristic (skip for fallback responses)
        is_fallback = "this pdf does not contain this information" in answer.lower()
        hallucination_risk = "low"
        if not is_fallback and context:
            answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
            context_words = set(re.findall(r"\b\w+\b", context.lower()))
            stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "it", "that", "this"}
            answer_content = answer_words - stopwords
            if answer_content:
                overlap = len(answer_content & context_words) / len(answer_content)
                if overlap < 0.35:
                    hallucination_risk = "medium"
                    flags.append(f"Low answer-context overlap ({round(overlap * 100)}%)")

        risk_level = "high" if unsafe_hits else ("medium" if flags else "low")
        self._log(f"Risk={risk_level} | flags={flags}")

        return {
            "risk_level": risk_level,
            "unsafe": bool(unsafe_hits),
            "hallucination_risk": hallucination_risk,
            "bias_flags": bias_hits,
            "flags": flags,
            "is_fallback_response": is_fallback,
        }


# ---------------------------------------------------------------------------
# EvaluationAgent
# ---------------------------------------------------------------------------

class EvaluationAgent(BaseAgent):
    """
    Accumulates per-query metrics and exposes a structured evaluation report.

    Tracks (intrinsic):  confidence score distribution
    Tracks (extrinsic):  fallback rate, latency, risk distribution
    """

    def __init__(self):
        super().__init__("EvaluationAgent")
        self._records: list = []

    def record(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list,
        confidence: int,
        answer_latency_ms: float,
        retrieval_latency_ms: float,
        risk: dict,
    ):
        self._records.append({
            "question_len": len(question.split()),
            "answer_len": len(answer.split()),
            "num_chunks": len(retrieved_chunks),
            "confidence": confidence,
            "answer_latency_ms": round(answer_latency_ms, 1),
            "retrieval_latency_ms": round(retrieval_latency_ms, 1),
            "risk_level": risk.get("risk_level", "unknown"),
            "is_fallback": risk.get("is_fallback_response", False),
        })
        self._log(f"Recorded interaction #{len(self._records)}")

    def get_report(self) -> dict:
        if not self._records:
            return {}
        n = len(self._records)
        confidences = [r["confidence"] for r in self._records]
        a_lat = [r["answer_latency_ms"] for r in self._records]
        r_lat = [r["retrieval_latency_ms"] for r in self._records]
        fallback_count = sum(1 for r in self._records if r["is_fallback"])
        risk_dist: dict = {}
        for r in self._records:
            risk_dist[r["risk_level"]] = risk_dist.get(r["risk_level"], 0) + 1

        return {
            "total_queries": n,
            "avg_confidence": round(sum(confidences) / n, 1),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "avg_answer_latency_ms": round(sum(a_lat) / n, 1),
            "avg_retrieval_latency_ms": round(sum(r_lat) / n, 1),
            "fallback_count": fallback_count,
            "fallback_rate_pct": round(fallback_count / n * 100, 1),
            "risk_distribution": risk_dist,
        }


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------

class CoordinatorAgent(BaseAgent):
    """
    Top-level orchestrator. Routes every user request through the full pipeline:
      Retrieve → Answer → Confidence → Risk → Evaluate → Return

    Also delegates study tasks (flashcards, quiz) to StudyAgent.
    """

    def __init__(
        self,
        retriever: RetrieverAgent,
        answer_agent: AnswerAgent,
        study_agent: StudyAgent,
        risk_monitor: RiskMonitorAgent,
        evaluator: EvaluationAgent,
        confidence_fn,
    ):
        super().__init__("CoordinatorAgent")
        self.retriever = retriever
        self.answer_agent = answer_agent
        self.study_agent = study_agent
        self.risk_monitor = risk_monitor
        self.evaluator = evaluator
        self.confidence_fn = confidence_fn

    def process_query(self, question: str, collection_name: str) -> dict:
        self._log(f"Processing query: '{question[:70]}'")

        # Step 1 – Retrieve
        retrieval = self.retriever.retrieve(question, collection_name)
        chunks = retrieval["chunks"]
        context = "\n\n".join(chunks)

        # Step 2 – Answer
        answer_result = self.answer_agent.answer(context, question)
        answer = answer_result["answer"]

        # Step 3 – Confidence (embedding + LLM self-eval)
        confidence = self.confidence_fn(
            question,
            chunks,
            self.retriever.embedding_model,
            answer=answer,
            client=self.answer_agent.client,
            model_name=self.answer_agent.model_name,
        )

        # Step 4 – Risk assessment
        risk = self.risk_monitor.assess(question, answer, context)

        # Step 5 – Record for evaluation
        self.evaluator.record(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            confidence=confidence["score"],
            answer_latency_ms=answer_result["latency_ms"],
            retrieval_latency_ms=retrieval["latency_ms"],
            risk=risk,
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "risk": risk,
            "retrieval_latency_ms": retrieval["latency_ms"],
            "answer_latency_ms": answer_result["latency_ms"],
        }

    def get_evaluation_report(self) -> dict:
        return self.evaluator.get_report()
