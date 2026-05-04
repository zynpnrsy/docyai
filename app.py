import os
import random
import time
import uuid
import gradio as gr
import chromadb
import pandas as pd

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
from flashcard_ui import (
    get_flashcard_css, render_flashcard_html, render_empty_card,
    render_flashcard_summary,
)
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
    """Return [(page_number, text), ...] — 1-based page numbers."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(pages, chunk_size=800, overlap=200):
    """
    Chunk per-page so each chunk carries the page number it came from.
    Returns ([chunk_text, ...], [page_num, ...]) parallel lists.
    """
    chunks, page_nums = [], []
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start: start + chunk_size]))
            page_nums.append(page_num)
            start += chunk_size - overlap
            if start >= len(words):
                break
    return chunks, page_nums


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


def _to_messages(tuples_history: list) -> list:
    """Convert internal [(user, assistant), ...] history to gr.Chatbot's messages format."""
    msgs = []
    for u, a in tuples_history or []:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs


def render_sources_html(sources: list) -> str:
    """Render retrieved chunks as collapsible source citations."""
    if not sources:
        return ""

    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    items = []
    for i, s in enumerate(sources, start=1):
        page = s.get("page")
        page_label = f"Page {page}" if page is not None else "Unknown page"
        excerpt = (s.get("chunk") or "").strip().replace("\n", " ")
        if len(excerpt) > 320:
            excerpt = excerpt[:320] + "…"
        items.append(
            f"<details style='margin:6px 0;background:#0f172a;border:1px solid #334155;"
            f"border-radius:8px;padding:8px 12px'>"
            f"<summary style='cursor:pointer;color:#cbd5e1;font-size:13px'>"
            f"<span style='color:#60a5fa;font-weight:600'>[{i}]</span> "
            f"<span style='color:#94a3b8'>{page_label}</span></summary>"
            f"<p style='margin:8px 0 0 0;color:#cbd5e1;font-size:12px;line-height:1.55'>"
            f"{_esc(excerpt)}</p></details>"
        )

    return (
        "<div style='margin-top:8px'>"
        "<p style='font-size:11px;letter-spacing:.1em;text-transform:uppercase;"
        "color:#64748b;font-family:monospace;margin:0 0 6px 0'>"
        f"Sources ({len(sources)})</p>"
        + "".join(items) +
        "</div>"
    )

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

def _summarise_records(records: list) -> dict:
    """Compute the same shape as EvaluationAgent.get_report() over an arbitrary list."""
    if not records:
        return {}
    n = len(records)
    confidences = [r["confidence"] for r in records]
    a_lat = [r["answer_latency_ms"] for r in records]
    r_lat = [r["retrieval_latency_ms"] for r in records]
    fallback_count = sum(1 for r in records if r["is_fallback"])
    risk_dist: dict = {}
    for r in records:
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


def render_eval_report(report: dict) -> str:
    if not report:
        return "No records yet for this filter."

    risk = report.get("risk_distribution", {})
    risk_str = "  ".join(f"{k.upper()}: {v}" for k, v in risk.items()) or "—"

    return (
        f"**Total records:** {report['total_queries']}\n\n"
        f"**Confidence / accuracy** — Avg: {report['avg_confidence']}/100 "
        f"| Min: {report['min_confidence']} | Max: {report['max_confidence']}\n\n"
        f"**Avg answer latency:** {report['avg_answer_latency_ms']} ms\n\n"
        f"**Avg retrieval latency:** {report['avg_retrieval_latency_ms']} ms\n\n"
        f"**Fallback rate:** {report['fallback_rate_pct']}% "
        f"({report['fallback_count']} / {report['total_queries']})\n\n"
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
    flash_state = gr.State({
        "cards": [], "index": 0, "flipped": False,
        "known": [], "unknown": [], "finished": False,
    })
    quiz_state  = gr.State({
        "questions": [], "index": 0, "answers": {}, "finished": False,
        "question_start": None, "durations": {}, "order": [],
    })
    chat_history = gr.State([])  # [(user_msg, assistant_msg), ...]

    with gr.Tabs():

        # ── Chat Assistant ──────────────────────────────────────────────────
        with gr.Tab("🔍 Chat Assistant"):
            chatbot            = gr.Chatbot(label="Conversation", height=420)
            question           = gr.Textbox(label="Ask a question", lines=2,
                                            placeholder="Type a question and press Run…")
            confidence_display = gr.Textbox(label="Confidence Score (latest answer)",
                                            interactive=False, lines=1)
            risk_display       = gr.HTML(label="Risk Assessment")
            sources_display    = gr.HTML(label="Sources")

            with gr.Row():
                run_btn        = gr.Button("▶️ Run", variant="primary")
                regenerate_btn = gr.Button("🔄 Regenerate")
                stop_btn       = gr.Button("⛔ Stop Thinking")
                clear_btn      = gr.Button("🆕 New Question")

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
            with gr.Row():
                flash_num = gr.Radio(choices=[5, 10], value=5,
                                     label="Number of cards", scale=1)
                gen_btn   = gr.Button("✨ Generate Study Cards",
                                      variant="secondary", scale=2)
            card_display = gr.HTML(value=render_empty_card())
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous", elem_id="fc-prev")
                flip_btn = gr.Button("🔄 Flip / Show Answer", elem_id="fc-flip")
                next_btn = gr.Button("Next ➡️", elem_id="fc-next")
            with gr.Row():
                knew_btn   = gr.Button("✓ Knew it", variant="primary")
                unknew_btn = gr.Button("✗ Didn't know")
            with gr.Row():
                shuffle_btn = gr.Button("🔀 Shuffle")
                summary_btn = gr.Button("📊 Show Summary")
                review_restart_btn = gr.Button("🔁 Restart Review")

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
            quiz_skip_btn = gr.Button("⏭ Skip (review later)", visible=False)
            quiz_next_btn = gr.Button("Next Question ➡️", variant="primary")
            with gr.Row():
                quiz_retry_wrong_btn = gr.Button("🎯 Retry Wrong Answers",
                                                  variant="primary", visible=False)
                quiz_restart_btn     = gr.Button("🔄 New Quiz")

        # ── Evaluation Dashboard ────────────────────────────────────────────
        with gr.Tab("📊 Evaluation Dashboard"):
            gr.Markdown(
                "Tracks intrinsic metrics (confidence score) and extrinsic metrics "
                "(fallback rate, latency, risk) across all queries in this session."
            )
            with gr.Row():
                source_filter = gr.Radio(
                    choices=["All", "Chat", "Quiz", "Flashcard"],
                    value="All", label="Source filter", scale=3,
                )
                refresh_btn  = gr.Button("🔄 Refresh Metrics",
                                          variant="secondary", scale=1)
            eval_display = gr.Markdown("No queries recorded yet.")
            with gr.Row():
                conf_plot = gr.LinePlot(
                    x="query", y="confidence",
                    title="Confidence trend", height=260,
                )
                latency_plot = gr.LinePlot(
                    x="query", y="latency_ms", color="kind",
                    title="Latency (answer + retrieval)", height=260,
                )
            risk_plot = gr.BarPlot(
                x="risk", y="count", title="Risk distribution", height=240,
            )
            gr.Markdown("#### Recent queries")
            recent_table = gr.Dataframe(
                headers=["#", "Question", "Confidence", "Risk", "Answer ms", "Fallback"],
                datatype=["number", "str", "number", "str", "number", "str"],
                interactive=False,
                wrap=True,
            )

    # ── PDF processing ──────────────────────────────────────────────────────

    def process_pdf(file):
        pages = load_pdf(file.name)
        chunks, page_nums = chunk_text(pages)
        embeddings = embedding_model.encode(chunks)

        # Each upload gets a fresh isolated ChromaDB collection
        collection_name = f"docai_{uuid.uuid4().hex[:8]}"
        collection = chroma_client.create_collection(collection_name)
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"page": p} for p in page_nums],
            ids=[str(i) for i in range(len(chunks))],
        )
        return (chunks, collection_name), f"✅ PDF indexed in ChromaDB ({len(chunks)} chunks, {len(pages)} pages)"

    pdf_file.upload(process_pdf, inputs=pdf_file, outputs=[state, status])

    # ── Chat logic ──────────────────────────────────────────────────────────

    def _stream_answer(q, state_val, history, temperature):
        if state_val is None:
            new_hist = history + [(q, "Please upload a PDF first.")]
            yield (
                new_hist, _to_messages(new_hist), "",
                "",
                render_risk_html({"risk_level": "low", "flags": []}),
                "",
            )
            return

        _, collection_name = state_val
        for event in coordinator.process_query_stream(
            q, collection_name, history=history, temperature=temperature
        ):
            if event["type"] == "partial":
                streaming_hist = history + [(q, event["answer"])]
                yield (
                    history,                          # don't commit to chat_history yet
                    _to_messages(streaming_hist),     # display only
                    "",                               # clear input
                    "Generating…",
                    "",
                    "",
                )
            else:  # final
                conf = event["confidence"]
                conf_text = f"Confidence: {conf['score']}/100"
                if conf["llm_score"] is not None:
                    conf_text += (
                        f"  (Embedding: {conf['embedding_score']}/100 | "
                        f"LLM: {conf['llm_score']}/100)"
                    )
                new_hist = history + [(q, event["answer"])]
                yield (
                    new_hist,
                    _to_messages(new_hist),
                    "",
                    conf_text,
                    render_risk_html(event["risk"]),
                    render_sources_html(event.get("sources", [])),
                )

    def ask_question(question_val, state_val, history):
        history = history or []
        q = (question_val or "").strip()
        if not q:
            yield history, _to_messages(history), "", "", "", ""
            return
        yield from _stream_answer(q, state_val, history, temperature=0.0)

    def regenerate_answer(state_val, history):
        history = history or []
        if not history:
            yield history, _to_messages(history), "", "", "", ""
            return
        # Pop the last turn, re-ask the same question with higher temperature
        last_q, _ = history[-1]
        prior = history[:-1]
        yield from _stream_answer(last_q, state_val, prior, temperature=0.5)

    def handle_submit_rating(history, rating_val):
        if not history:
            return "Please ask a question first."
        if rating_val is None:
            return "Please select a star rating before submitting."
        last_q, last_a = history[-1]
        _, msg = save_feedback(last_q, last_a, int(rating_val))
        return msg

    def clear_chat():
        return [], [], "", "", "", "", None, ""

    chat_outputs = [chat_history, chatbot, question,
                    confidence_display, risk_display, sources_display]

    run_event = run_btn.click(
        ask_question,
        inputs=[question, state, chat_history],
        outputs=chat_outputs,
    )
    regen_event = regenerate_btn.click(
        regenerate_answer,
        inputs=[state, chat_history],
        outputs=chat_outputs,
    )
    stop_btn.click(None, None, None, cancels=[run_event, regen_event])

    # Auto-refresh the evaluation dashboard whenever a chat query completes.
    # Wired AFTER the eval components are defined; see end of Blocks.
    clear_btn.click(
        clear_chat,
        outputs=[chat_history, chatbot, question,
                 confidence_display, risk_display, sources_display,
                 rating, rating_status],
    )
    submit_rating_btn.click(
        handle_submit_rating, inputs=[chat_history, rating], outputs=[rating_status]
    )

    # ── Flashcard logic ─────────────────────────────────────────────────────

    def _empty_flash():
        return {
            "cards": [], "index": 0, "flipped": False,
            "known": [], "unknown": [], "finished": False,
        }

    def _render_card(f_state):
        cards = f_state["cards"]
        if f_state.get("finished"):
            return render_flashcard_summary(cards, f_state["known"], f_state["unknown"])
        if not cards:
            return render_empty_card()
        idx = f_state["index"]
        return render_flashcard_html(
            cards[idx]["question"], cards[idx]["answer"],
            idx + 1, len(cards), f_state.get("flipped", False),
        )

    def handle_gen(state_val, num):
        if state_val is None:
            empty = _empty_flash()
            return empty, _render_card(empty)
        chunks, _ = state_val
        context   = "\n\n".join(chunks[:5])
        cards     = coordinator.study_agent.generate_flashcards(context, num_cards=int(num))
        new_state = {
            "cards": cards, "index": 0, "flipped": False,
            "known": [], "unknown": [], "finished": False,
        }
        return new_state, _render_card(new_state)

    def handle_next(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, _render_card(f_state)
        new_idx = (f_state["index"] + 1) % len(cards)
        new_state = {**f_state, "index": new_idx, "flipped": False, "finished": False}
        return new_state, _render_card(new_state)

    def handle_prev(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, _render_card(f_state)
        new_idx = (f_state["index"] - 1) % len(cards)
        new_state = {**f_state, "index": new_idx, "flipped": False, "finished": False}
        return new_state, _render_card(new_state)

    def handle_flip(f_state):
        cards = f_state["cards"]
        if not cards or f_state.get("finished"):
            return f_state, _render_card(f_state)
        new_state = {**f_state, "flipped": not f_state["flipped"]}
        return new_state, _render_card(new_state)

    def _mark_card(f_state, knew: bool):
        cards = f_state["cards"]
        if not cards or f_state.get("finished"):
            return f_state, _render_card(f_state)
        idx = f_state["index"]
        # Remove from both lists first (in case user is changing their mind), then add.
        known   = [i for i in f_state["known"]   if i != idx]
        unknown = [i for i in f_state["unknown"] if i != idx]
        (known if knew else unknown).append(idx)

        is_last = idx == len(cards) - 1
        if is_last:
            coordinator.evaluator.record_flashcard_session(len(known), len(cards))
            new_state = {**f_state, "known": known, "unknown": unknown,
                         "flipped": False, "finished": True}
        else:
            new_state = {**f_state, "known": known, "unknown": unknown,
                         "index": idx + 1, "flipped": False}
        return new_state, _render_card(new_state)

    def handle_knew(f_state):
        return _mark_card(f_state, knew=True)

    def handle_didnt_know(f_state):
        return _mark_card(f_state, knew=False)

    def handle_show_summary(f_state):
        if not f_state["cards"]:
            return f_state, _render_card(f_state)
        if not f_state.get("finished"):
            coordinator.evaluator.record_flashcard_session(
                len(f_state["known"]), len(f_state["cards"])
            )
        new_state = {**f_state, "finished": True, "flipped": False}
        return new_state, _render_card(new_state)

    def handle_restart_review(f_state):
        if not f_state["cards"]:
            return f_state, _render_card(f_state)
        new_state = {**f_state, "index": 0, "flipped": False,
                     "known": [], "unknown": [], "finished": False}
        return new_state, _render_card(new_state)

    def handle_shuffle(f_state):
        if not f_state["cards"]:
            return f_state, _render_card(f_state)
        # Shuffling reorders cards, so previous known/unknown indices are no longer
        # meaningful — reset review progress.
        shuffled = f_state["cards"][:]
        random.shuffle(shuffled)
        new_state = {**f_state, "cards": shuffled, "index": 0, "flipped": False,
                     "known": [], "unknown": [], "finished": False}
        return new_state, _render_card(new_state)

    flash_outputs = [flash_state, card_display]
    gen_btn.click(handle_gen,             inputs=[state, flash_num], outputs=flash_outputs)
    next_btn.click(handle_next,           inputs=[flash_state], outputs=flash_outputs)
    prev_btn.click(handle_prev,           inputs=[flash_state], outputs=flash_outputs)
    flip_btn.click(handle_flip,           inputs=[flash_state], outputs=flash_outputs)
    knew_btn.click(handle_knew,           inputs=[flash_state], outputs=flash_outputs)
    unknew_btn.click(handle_didnt_know,   inputs=[flash_state], outputs=flash_outputs)
    summary_btn.click(handle_show_summary, inputs=[flash_state], outputs=flash_outputs)
    review_restart_btn.click(handle_restart_review, inputs=[flash_state], outputs=flash_outputs)
    shuffle_btn.click(handle_shuffle, inputs=[flash_state], outputs=flash_outputs)

    # ── Quiz logic ──────────────────────────────────────────────────────────

    LETTERS = ["A", "B", "C", "D"]

    def _opt_label(letter, text, marker=""):
        prefix = f"{marker} " if marker else ""
        return f"{prefix}{letter} — {text}"

    def _quiz_outputs(q_state, header, feedback, result, vis, inter, la, lb, lc, ld,
                      next_vis, restart_vis, retry_vis=False, skip_vis=False):
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
            gr.update(visible=retry_vis),
            gr.update(visible=skip_vis),
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
        empty = {
            "questions": [], "index": 0, "answers": {}, "finished": False,
            "question_start": None, "durations": {}, "order": [],
        }
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
        order = list(range(len(questions)))
        new_state = {
            "questions": questions, "index": 0, "answers": {}, "finished": False,
            "question_start": time.time(), "durations": {}, "order": order,
        }
        first  = questions[order[0]]
        labels = _init_labels(first["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(first["question"], 1, len(order)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False, skip_vis=True,
        )

    def handle_answer(q_state, selected_idx):
        questions = q_state.get("questions", [])
        order     = q_state.get("order", [])
        if not questions or not order or q_state.get("finished"):
            return (q_state,) + tuple(gr.update() for _ in range(11))

        cursor   = q_state["index"]
        q_idx    = order[cursor]                       # real question index
        # Record time spent thinking on this question (only on first selection)
        new_durations = dict(q_state.get("durations") or {})
        if q_idx not in new_durations:
            qs = q_state.get("question_start")
            if qs is not None:
                new_durations[q_idx] = round(time.time() - qs, 1)

        new_answers = {**q_state["answers"], q_idx: selected_idx}
        new_state  = {**q_state, "answers": new_answers, "durations": new_durations}
        q          = questions[q_idx]
        labels     = _marked_labels(q["options"], selected_idx, q["correct_index"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(q["question"], cursor + 1, len(order)),
            render_quiz_feedback_html(selected_idx, q["correct_index"], q["options"], q.get("explanation")),
            "",
            True, False,
            labels[0], labels[1], labels[2], labels[3],
            True, False, skip_vis=False,
        )

    def handle_quiz_next(q_state):
        questions = q_state.get("questions", [])
        order     = q_state.get("order", [])
        if not questions or not order:
            return _quiz_outputs(q_state, render_empty_quiz(), "", "", False, False,
                                 "A", "B", "C", "D", False, False)

        next_cursor = q_state["index"] + 1
        if next_cursor >= len(order):
            answers = q_state.get("answers", {})
            score   = sum(1 for i, q in enumerate(questions)
                         if answers.get(i) == q["correct_index"])
            durations = q_state.get("durations") or {}
            results = [
                {
                    "question": q["question"], "options": q["options"],
                    "correct_index": q["correct_index"],
                    "selected_index": answers.get(i),
                    "explanation": q.get("explanation", ""),
                    "is_correct": answers.get(i) == q["correct_index"],
                    "time_seconds": durations.get(i),
                }
                for i, q in enumerate(questions)
            ]
            total_seconds = round(sum(v for v in durations.values() if v), 1)
            has_wrong = any(not r["is_correct"] for r in results)
            coordinator.evaluator.record_quiz_session(score, len(questions), total_seconds)
            return _quiz_outputs(
                {**q_state, "finished": True},
                "", "", render_quiz_result_html(score, len(questions), results,
                                                total_seconds=total_seconds),
                False, False, "A", "B", "C", "D", False, True,
                retry_vis=has_wrong, skip_vis=False,
            )

        new_state = {**q_state, "index": next_cursor, "question_start": time.time()}
        q_idx = order[next_cursor]
        q = questions[q_idx]
        # If this question was answered earlier (e.g., we're cycling back to a
        # previously skipped one), skip is meaningless; show feedback again.
        already_answered = q_idx in q_state.get("answers", {})
        labels = _init_labels(q["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(q["question"], next_cursor + 1, len(order)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False, skip_vis=not already_answered,
        )

    def handle_quiz_restart():
        return _quiz_outputs(
            {
                "questions": [], "index": 0, "answers": {}, "finished": False,
                "question_start": None, "durations": {}, "order": [],
            },
            render_empty_quiz(), "", "",
            False, False, "A", "B", "C", "D", False, False,
        )

    def handle_retry_wrong(q_state):
        questions = q_state.get("questions", [])
        answers   = q_state.get("answers", {})
        wrong_qs = [q for i, q in enumerate(questions)
                    if answers.get(i) != q["correct_index"]]
        if not wrong_qs:
            return _quiz_outputs(q_state, "", "", "", False, False,
                                 "A", "B", "C", "D", False, True)
        order = list(range(len(wrong_qs)))
        new_state = {
            "questions": wrong_qs, "index": 0, "answers": {}, "finished": False,
            "question_start": time.time(), "durations": {}, "order": order,
        }
        first  = wrong_qs[0]
        labels = _init_labels(first["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(first["question"], 1, len(wrong_qs)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False, retry_vis=False, skip_vis=True,
        )

    def handle_skip(q_state):
        questions = q_state.get("questions", [])
        order     = list(q_state.get("order") or [])
        cursor    = q_state.get("index", 0)
        if not questions or not order or q_state.get("finished"):
            return (q_state,) + tuple(gr.update() for _ in range(11))

        # Already on the only remaining question — skipping would loop forever.
        if cursor >= len(order) - 1:
            return (q_state,) + tuple(gr.update() for _ in range(11))

        # Move the current question to the end of the queue. Cursor stays put
        # and now points at the next question.
        skipped = order.pop(cursor)
        order.append(skipped)

        new_state = {**q_state, "order": order, "question_start": time.time()}
        q_idx  = order[cursor]
        q      = questions[q_idx]
        already_answered = q_idx in q_state.get("answers", {})
        labels = _init_labels(q["options"])
        return _quiz_outputs(
            new_state,
            render_quiz_header_html(q["question"], cursor + 1, len(order)),
            "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False, skip_vis=not already_answered,
        )

    quiz_outputs_list = [
        quiz_state, quiz_header, quiz_feedback, quiz_result,
        opt_a_btn, opt_b_btn, opt_c_btn, opt_d_btn,
        quiz_next_btn, quiz_restart_btn, quiz_retry_wrong_btn, quiz_skip_btn,
    ]

    quiz_gen_btn.click(handle_quiz_gen, inputs=[state, quiz_topic, quiz_num], outputs=quiz_outputs_list)
    opt_a_btn.click(lambda q: handle_answer(q, 0), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_b_btn.click(lambda q: handle_answer(q, 1), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_c_btn.click(lambda q: handle_answer(q, 2), inputs=[quiz_state], outputs=quiz_outputs_list)
    opt_d_btn.click(lambda q: handle_answer(q, 3), inputs=[quiz_state], outputs=quiz_outputs_list)
    quiz_next_btn.click(handle_quiz_next, inputs=[quiz_state], outputs=quiz_outputs_list)
    quiz_restart_btn.click(handle_quiz_restart, inputs=None, outputs=quiz_outputs_list)
    quiz_retry_wrong_btn.click(handle_retry_wrong, inputs=[quiz_state], outputs=quiz_outputs_list)
    quiz_skip_btn.click(handle_skip, inputs=[quiz_state], outputs=quiz_outputs_list)

    # ── Evaluation dashboard ────────────────────────────────────────────────

    def _filter_records(filter_val: str):
        records = coordinator.evaluator.get_records()
        if not filter_val or filter_val == "All":
            return records
        target = filter_val.lower()
        return [r for r in records if r.get("source", "chat") == target]

    def _build_eval_dataframes(records):
        if not records:
            empty_line = pd.DataFrame({"query": [], "confidence": [], "latency_ms": [], "kind": []})
            empty_bar  = pd.DataFrame({"risk": [], "count": []})
            return empty_line, empty_line, empty_bar

        conf_df = pd.DataFrame({
            "query":      list(range(1, len(records) + 1)),
            "confidence": [r["confidence"] for r in records],
        })
        latency_df = pd.DataFrame({
            "query":      list(range(1, len(records) + 1)) * 2,
            "latency_ms": [r["answer_latency_ms"]    for r in records]
                          + [r["retrieval_latency_ms"] for r in records],
            "kind":       ["answer"]    * len(records)
                          + ["retrieval"] * len(records),
        })
        risk_counts = {}
        for r in records:
            level = r.get("risk_level", "unknown")
            risk_counts[level] = risk_counts.get(level, 0) + 1
        risk_df = pd.DataFrame({
            "risk":  list(risk_counts.keys()),
            "count": list(risk_counts.values()),
        })
        return conf_df, latency_df, risk_df

    def _build_recent_table(records):
        rows = []
        for i, r in enumerate(records, start=1):
            rows.append([
                i,
                f"[{r.get('source', 'chat')}] {r.get('question', '')}",
                r.get("confidence", 0),
                r.get("risk_level", "—"),
                r.get("answer_latency_ms", 0),
                "yes" if r.get("is_fallback") else "no",
            ])
        rows.reverse()  # most recent first
        return rows

    def refresh_eval(filter_val="All"):
        records = _filter_records(filter_val)
        conf_df, latency_df, risk_df = _build_eval_dataframes(records)
        return (
            render_eval_report(_summarise_records(records)),
            conf_df, latency_df, risk_df,
            _build_recent_table(records),
        )

    eval_outputs = [eval_display, conf_plot, latency_plot, risk_plot, recent_table]
    refresh_btn.click(refresh_eval, inputs=[source_filter], outputs=eval_outputs)
    source_filter.change(refresh_eval, inputs=[source_filter], outputs=eval_outputs)

    # Auto-refresh dashboard after each chat query / regeneration.
    run_event.then(refresh_eval, inputs=[source_filter], outputs=eval_outputs)
    regen_event.then(refresh_eval, inputs=[source_filter], outputs=eval_outputs)

    # ── Flashcard keyboard shortcuts (Space = flip, ←/→ = prev/next) ────────
    gr.HTML("""
<script>
(function() {
    if (window.__fcKeysBound) return;
    window.__fcKeysBound = true;

    function findButton(elemId) {
        // Gradio sometimes wraps buttons; check both the elem itself and descendants.
        const root = document.getElementById(elemId);
        if (!root) return null;
        return root.tagName === 'BUTTON' ? root : root.querySelector('button');
    }

    document.addEventListener('keydown', function(e) {
        // Don't intercept when typing in an input field.
        const ae = document.activeElement;
        const tag = (ae && ae.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || (ae && ae.isContentEditable)) return;

        // Only fire when a flashcard is currently visible on screen.
        const scene = document.querySelector('.fc-scene');
        if (!scene || !scene.offsetParent) return;

        let id = null;
        if (e.code === 'Space')           id = 'fc-flip';
        else if (e.code === 'ArrowLeft')  id = 'fc-prev';
        else if (e.code === 'ArrowRight') id = 'fc-next';
        if (!id) return;

        e.preventDefault();
        const btn = findButton(id);
        if (btn) btn.click();
    });
})();
</script>
""")

demo.launch(server_name="0.0.0.0", server_port=7860)
