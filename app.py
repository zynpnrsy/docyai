import os
import uuid
import gradio as gr
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from agents import (
    RetrieverAgent,
    AnswerAgent,
    StudyAgent,
    RiskMonitorAgent,
    EvaluationAgent,
    CoordinatorAgent,
)
from confidence_scorer import calculate_confidence
from feedback import save_feedback
from flashcard_ui import get_flashcard_css, render_flashcard_html, render_empty_card
from quiz_ui import (
    get_quiz_css,
    render_quiz_header_html,
    render_quiz_feedback_html,
    render_quiz_result_html,
    render_empty_quiz,
)

load_dotenv()

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text


def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Shared infrastructure (module-level singletons)
# ---------------------------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.EphemeralClient()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Agent instantiation
# ---------------------------------------------------------------------------

retriever_agent = RetrieverAgent(embedding_model, chroma_client)
answer_agent    = AnswerAgent(groq_client, MODEL_NAME)
study_agent     = StudyAgent(groq_client, MODEL_NAME)
risk_monitor    = RiskMonitorAgent()
evaluator       = EvaluationAgent()

coordinator = CoordinatorAgent(
    retriever=retriever_agent,
    answer_agent=answer_agent,
    study_agent=study_agent,
    risk_monitor=risk_monitor,
    evaluator=evaluator,
    confidence_fn=calculate_confidence,
)

# ---------------------------------------------------------------------------
# Risk HTML helper
# ---------------------------------------------------------------------------

_RISK_COLORS = {"low": "#4ade80", "medium": "#facc15", "high": "#f87171"}

def render_risk_html(risk: dict) -> str:
    level = risk.get("risk_level", "low")
    color = _RISK_COLORS.get(level, "#94a3b8")
    flags = risk.get("flags", [])
    fallback = risk.get("is_fallback_response", False)

    flag_html = ""
    if flags:
        items = "".join(f"<li style='margin:4px 0;color:#cbd5e1'>{f}</li>" for f in flags)
        flag_html = f"<ul style='margin:8px 0 0 0;padding-left:18px'>{items}</ul>"

    fallback_note = (
        "<p style='margin:6px 0 0 0;font-size:12px;color:#64748b'>"
        "ℹ️ Model answered with fallback (topic not in PDF)</p>"
        if fallback else ""
    )

    return f"""
<div style='background:#1e293b;border:1px solid {color}44;border-radius:12px;padding:14px 18px;margin-top:8px'>
  <div style='display:flex;align-items:center;gap:10px'>
    <span style='font-size:11px;font-family:monospace;letter-spacing:.1em;text-transform:uppercase;color:#64748b'>Risk Level</span>
    <span style='font-weight:700;font-size:14px;color:{color}'>{level.upper()}</span>
  </div>
  {flag_html}{fallback_note}
</div>"""


# ---------------------------------------------------------------------------
# Evaluation report helper
# ---------------------------------------------------------------------------

def render_eval_report(report: dict) -> str:
    if not report:
        return "No queries have been recorded yet. Ask a question first."

    risk = report.get("risk_distribution", {})
    risk_str = "  ".join(f"{k.upper()}: {v}" for k, v in risk.items()) or "—"

    return (
        f"**Total queries recorded:** {report['total_queries']}\n\n"
        f"**Confidence score** — Avg: {report['avg_confidence']}/100 "
        f"| Min: {report['min_confidence']} | Max: {report['max_confidence']}\n\n"
        f"**Avg answer latency:** {report['avg_answer_latency_ms']} ms\n\n"
        f"**Avg retrieval latency:** {report['avg_retrieval_latency_ms']} ms\n\n"
        f"**Fallback rate:** {report['fallback_rate_pct']}% "
        f"({report['fallback_count']} / {report['total_queries']} queries not in PDF)\n\n"
        f"**Risk distribution:** {risk_str}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=get_flashcard_css() + get_quiz_css()) as demo:

    gr.Markdown("## 📄 Welcome to DOCY AI")
    gr.Markdown("### Upload your PDF to Chat or Study with Flashcards.")

    with gr.Row():
        pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
        status = gr.Markdown("Please upload a PDF to start.")

    # State stores (chunks, chromadb_collection_name)
    state      = gr.State()
    flash_state = gr.State({"cards": [], "index": 0, "flipped": False})
    quiz_state  = gr.State({"questions": [], "index": 0, "answers": {}, "finished": False})

    with gr.Tabs():

        # ── Chat Assistant ──────────────────────────────────────────────────
        with gr.Tab("🔍 Chat Assistant"):
            question           = gr.Textbox(label="Ask a question", lines=3)
            answer             = gr.Textbox(label="Answer", lines=8)
            confidence_display = gr.Textbox(label="Confidence Score", interactive=False, lines=1)
            risk_display       = gr.HTML(label="Risk Assessment")

            with gr.Row():
                run_btn   = gr.Button("▶️ Run", variant="primary")
                stop_btn  = gr.Button("⛔ Stop Thinking")
                clear_btn = gr.Button("🆕 New Question")

            gr.Markdown("---")
            gr.Markdown("#### ⭐ Rate this answer")
            with gr.Row():
                rating = gr.Radio(
                    choices=[("⭐ 1", 1), ("⭐⭐ 2", 2), ("⭐⭐⭐ 3", 3),
                             ("⭐⭐⭐⭐ 4", 4), ("⭐⭐⭐⭐⭐ 5", 5)],
                    label="How helpful was this answer?",
                    type="value",
                    interactive=True,
                )
                submit_rating_btn = gr.Button("Submit Rating", variant="secondary")
            rating_status = gr.Markdown("")

        # ── Flashcard Study ─────────────────────────────────────────────────
        with gr.Tab("🗂️ Flashcard Study"):
            gen_btn      = gr.Button("✨ Generate Study Cards", variant="secondary")
            card_display = gr.HTML(value=render_empty_card())
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous")
                flip_btn = gr.Button("🔄 Flip / Show Answer")
                next_btn = gr.Button("Next ➡️")

        # ── MCQ Quiz ────────────────────────────────────────────────────────
        with gr.Tab("📝 MCQ Quiz"):
            with gr.Row():
                quiz_topic = gr.Textbox(
                    label="Topic (optional)",
                    placeholder="e.g., 'RAG architecture' or leave empty",
                    scale=3,
                )
                quiz_num = gr.Radio(choices=[5, 10], value=5, label="Number of questions", scale=1)

            quiz_gen_btn = gr.Button("🎯 Generate Quiz", variant="primary")
            quiz_header  = gr.HTML(value=render_empty_quiz())

            opt_a_btn = gr.Button("A", variant="secondary", interactive=False, visible=False)
            opt_b_btn = gr.Button("B", variant="secondary", interactive=False, visible=False)
            opt_c_btn = gr.Button("C", variant="secondary", interactive=False, visible=False)
            opt_d_btn = gr.Button("D", variant="secondary", interactive=False, visible=False)

            quiz_feedback = gr.HTML(value="")
            quiz_result   = gr.HTML(value="")
            quiz_next_btn    = gr.Button("Next Question ➡️", variant="primary")
            quiz_restart_btn = gr.Button("🔄 New Quiz")

        # ── Evaluation Dashboard ────────────────────────────────────────────
        with gr.Tab("📊 Evaluation Dashboard"):
            gr.Markdown(
                "Tracks intrinsic metrics (confidence score) and extrinsic metrics "
                "(fallback rate, latency, risk) across all queries in this session."
            )
            refresh_btn  = gr.Button("🔄 Refresh Metrics", variant="secondary")
            eval_display = gr.Markdown("No queries recorded yet.")

    # ── PDF processing ──────────────────────────────────────────────────────

    def process_pdf(file):
        text   = load_pdf(file.name)
        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks)

        # Each upload gets a fresh isolated ChromaDB collection
        collection_name = f"docai_{uuid.uuid4().hex[:8]}"
        collection = chroma_client.create_collection(collection_name)
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[str(i) for i in range(len(chunks))],
        )
        return (chunks, collection_name), f"✅ PDF indexed in ChromaDB ({len(chunks)} chunks)"

    pdf_file.upload(process_pdf, inputs=pdf_file, outputs=[state, status])

    # ── Chat logic ──────────────────────────────────────────────────────────

    def ask_question(question_val, state_val):
        if state_val is None:
            return "Please upload a PDF first.", "", render_risk_html({"risk_level": "low", "flags": []})

        _, collection_name = state_val
        result = coordinator.process_query(question_val, collection_name)

        conf = result["confidence"]
        conf_text = f"Confidence: {conf['score']}/100"
        if conf["llm_score"] is not None:
            conf_text += f"  (Embedding: {conf['embedding_score']}/100 | LLM: {conf['llm_score']}/100)"

        return result["answer"], conf_text, render_risk_html(result["risk"])

    def handle_submit_rating(question_val, answer_val, rating_val):
        if not answer_val:
            return "Please ask a question first."
        if rating_val is None:
            return "Please select a star rating before submitting."
        _, msg = save_feedback(question_val, answer_val, int(rating_val))
        return msg

    def clear_chat():
        return "", "", "", "", None, ""

    run_event = run_btn.click(
        ask_question,
        inputs=[question, state],
        outputs=[answer, confidence_display, risk_display],
    )
    stop_btn.click(None, None, None, cancels=[run_event])
    clear_btn.click(clear_chat, outputs=[question, answer, confidence_display, risk_display, rating, rating_status])
    submit_rating_btn.click(
        handle_submit_rating, inputs=[question, answer, rating], outputs=[rating_status]
    )

    # ── Flashcard logic ─────────────────────────────────────────────────────

    def handle_gen(state_val):
        if state_val is None:
            return {"cards": [], "index": 0, "flipped": False}, render_empty_card()
        chunks, _ = state_val
        context   = "\n\n".join(chunks[:5])
        cards     = coordinator.study_agent.generate_flashcards(context)
        html = render_flashcard_html(cards[0]["question"], cards[0]["answer"], 1, len(cards), False)
        return {"cards": cards, "index": 0, "flipped": False}, html

    def handle_next(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_idx   = (f_state["index"] + 1) % len(cards)
        new_state = {"cards": cards, "index": new_idx, "flipped": False}
        return new_state, render_flashcard_html(
            cards[new_idx]["question"], cards[new_idx]["answer"], new_idx + 1, len(cards), False
        )

    def handle_prev(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_idx   = (f_state["index"] - 1) % len(cards)
        new_state = {"cards": cards, "index": new_idx, "flipped": False}
        return new_state, render_flashcard_html(
            cards[new_idx]["question"], cards[new_idx]["answer"], new_idx + 1, len(cards), False
        )

    def handle_flip(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_flipped = not f_state["flipped"]
        idx         = f_state["index"]
        new_state   = {**f_state, "flipped": new_flipped}
        return new_state, render_flashcard_html(
            cards[idx]["question"], cards[idx]["answer"], idx + 1, len(cards), new_flipped
        )

    gen_btn.click(handle_gen,  inputs=[state],       outputs=[flash_state, card_display])
    next_btn.click(handle_next, inputs=[flash_state], outputs=[flash_state, card_display])
    prev_btn.click(handle_prev, inputs=[flash_state], outputs=[flash_state, card_display])
    flip_btn.click(handle_flip, inputs=[flash_state], outputs=[flash_state, card_display])

    # ── Quiz logic ──────────────────────────────────────────────────────────

    LETTERS = ["A", "B", "C", "D"]

    def _opt_label(letter, text, marker=""):
        prefix = f"{marker} " if marker else ""
        return f"{prefix}{letter} — {text}"

    def _quiz_outputs(q_state, header, feedback, result, vis, inter, la, lb, lc, ld, next_vis, restart_vis):
        return (
            q_state,
            gr.update(value=header),
            gr.update(value=feedback),
            gr.update(value=result),
            gr.update(value=la, visible=vis, interactive=inter),
            gr.update(value=lb, visible=vis, interactive=inter),
            gr.update(value=lc, visible=vis, interactive=inter),
            gr.update(value=ld, visible=vis, interactive=inter),
            gr.update(visible=next_vis),
            gr.update(visible=restart_vis),
        )

    def _init_labels(opts):
        return [_opt_label(LETTERS[i], opts[i]) for i in range(4)]

    def _marked_labels(opts, sel, cor):
        labels = []
        for i in range(4):
            marker = "✓" if i == cor else ("✗" if i == sel else "")
            labels.append(_opt_label(LETTERS[i], opts[i], marker))
        return labels

    def handle_quiz_gen(p_state, topic, num):
        empty = {"questions": [], "index": 0, "answers": {}, "finished": False}
        if p_state is None:
            return _quiz_outputs(empty,
                "<div class='qz-scene'><div class='qz-empty'>⚠️ Please upload a PDF first.</div></div>",
                "", "", False, False, "A", "B", "C", "D", False, False)

        chunks, collection_name = p_state
        topic_clean = (topic or "").strip()
        if topic_clean:
            retrieval = coordinator.retriever.retrieve(topic_clean, collection_name, k=5)
            context   = "\n\n".join(retrieval["chunks"])
        else:
            context = "\n\n".join(chunks[:5])

        questions = coordinator.study_agent.generate_quiz(
            context, num_questions=int(num), topic=topic_clean or None
        )
        new_state = {"questions": questions, "index": 0, "answers": {}, "finished": False}
        first  = questions[0]
        labels = _init_labels(first["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(first["question"], 1, len(questions)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False,
        )

    def handle_answer(q_state, selected_idx):
        questions = q_state.get("questions", [])
        if not questions or q_state.get("finished"):
            return (q_state,) + tuple(gr.update() for _ in range(9))

        idx        = q_state["index"]
        new_answers = {**q_state["answers"], idx: selected_idx}
        new_state  = {**q_state, "answers": new_answers}
        q          = questions[idx]
        labels     = _marked_labels(q["options"], selected_idx, q["correct_index"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(q["question"], idx + 1, len(questions)),
            render_quiz_feedback_html(selected_idx, q["correct_index"], q["options"], q.get("explanation")),
            "",
            True, False,
            labels[0], labels[1], labels[2], labels[3],
            True, False,
        )

    def handle_quiz_next(q_state):
        questions = q_state.get("questions", [])
        if not questions:
            return _quiz_outputs(q_state, render_empty_quiz(), "", "", False, False,
                                 "A", "B", "C", "D", False, False)

        next_idx = q_state["index"] + 1
        if next_idx >= len(questions):
            score   = sum(1 for i, q in enumerate(questions)
                         if q_state["answers"].get(i) == q["correct_index"])
            results = [
                {
                    "question": q["question"], "options": q["options"],
                    "correct_index": q["correct_index"],
                    "selected_index": q_state["answers"].get(i),
                    "explanation": q.get("explanation", ""),
                    "is_correct": q_state["answers"].get(i) == q["correct_index"],
                }
                for i, q in enumerate(questions)
            ]
            return _quiz_outputs(
                {**q_state, "finished": True},
                "", "", render_quiz_result_html(score, len(questions), results),
                False, False, "A", "B", "C", "D", False, True,
            )

        new_state = {**q_state, "index": next_idx}
        q = questions[next_idx]
        labels = _init_labels(q["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(q["question"], next_idx + 1, len(questions)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False,
        )

    def handle_quiz_restart():
        return _quiz_outputs(
            {"questions": [], "index": 0, "answers": {}, "finished": False},
            render_empty_quiz(), "", "",
            False, False, "A", "B", "C", "D", False, False,
        )

    quiz_outputs_list = [
        quiz_state, quiz_header, quiz_feedback, quiz_result,
        opt_a_btn, opt_b_btn, opt_c_btn, opt_d_btn,
        quiz_next_btn, quiz_restart_btn,
    ]

    quiz_gen_btn.click(handle_quiz_gen, inputs=[state, quiz_topic, quiz_num], outputs=quiz_outputs_list)
    opt_a_btn.click(lambda q: handle_answer(q, 0), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_b_btn.click(lambda q: handle_answer(q, 1), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_c_btn.click(lambda q: handle_answer(q, 2), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_d_btn.click(lambda q: handle_answer(q, 3), inputs=[quiz_state], outputs=quiz_outputs_list)
    quiz_next_btn.click(handle_quiz_next, inputs=[quiz_state], outputs=quiz_outputs_list)
    quiz_restart_btn.click(handle_quiz_restart, inputs=None, outputs=quiz_outputs_list)

    # ── Evaluation dashboard ────────────────────────────────────────────────

    def refresh_eval():
        return render_eval_report(coordinator.get_evaluation_report())

    refresh_btn.click(refresh_eval, outputs=[eval_display])

demo.launch(server_name="0.0.0.0", server_port=7860)
