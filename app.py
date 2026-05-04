import os
import numpy as np
import faiss
import gradio as gr
import json

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from flashcard_ui import get_flashcard_css, render_flashcard_html, render_empty_card
from confidence_scorer import calculate_confidence
from feedback import save_feedback
from quiz_ui import (
    get_quiz_css,
    render_quiz_header_html,
    render_quiz_feedback_html,
    render_quiz_question_html,
    render_quiz_result_html,
    render_empty_quiz,
)

load_dotenv()

# PDF Loader
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text


#Chunking
def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index


# Retrieval
def retrieve(query, chunks, index, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


# Groq Client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

#prompt
def ask_llm(context, question):
    prompt = f"""
You are a strict academic assistant.
Answer ONLY using the provided context.
If the answer is not explicitly stated in the context,
respond exactly with:
This PDF does not contain this information.
Do not use outside knowledge.
Context:
{context}
Question:
{question}
Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", #model type
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def generate_flashcards(context):
    prompt = f"""
    You are an expert educator. Based ONLY on the provided context, generate 5 high-quality flashcards.
    Each flashcard must have a 'question' and an 'answer'.
    
    Format your response EXACTLY as a JSON list of objects like this:
    [
      {{"question": "What is...?", "answer": "It is..."}},
      {{"question": "Explain...?", "answer": "..."}}
    ]
    
    Context:
    {context}
    
    JSON Output:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # Slight creativity for better card variety
        response_format={"type": "json_object"} if "llama-3.1" in "llama-3.1-8b-instant" else None
    )
    
    try:
        # Some models might wrap JSON in markdown blocks, let's be safe
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        
        cards = json.loads(content)
        # If the model returns a single object with a list, extract the list
        if isinstance(cards, dict) and "flashcards" in cards:
            return cards["flashcards"]
        return cards
    except Exception as e:
        print(f"Error parsing flashcards: {e}")
        return [{"question": "Error", "answer": "Could not generate cards. Please try again."}]


def generate_mcq_quiz(context, num_questions=5, topic=None):
    """Generate `num_questions` MCQs grounded in `context`. Returns a validated list."""
    topic_instruction = (
        f"Focus the questions specifically on: {topic}"
        if topic else
        "Cover the most important concepts from the context"
    )

    prompt = f"""
You are an expert quiz creator. Based ONLY on the provided context, generate
exactly {num_questions} multiple-choice questions.

{topic_instruction}

Rules:
- Each question MUST have exactly 4 options
- Exactly ONE option must be correct
- Use ONLY information from the context. Do NOT invent facts.
- Provide a brief explanation referencing the context for each correct answer.

Format your response EXACTLY as a JSON object:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}

Context:
{context}

JSON Output:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"}
    )

    try:
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        data = json.loads(content)
        questions = data.get("questions", data if isinstance(data, list) else [])

        # Output validation: discard malformed questions
        valid = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            if not all(k in q for k in ["question", "options", "correct_index", "explanation"]):
                continue
            if not isinstance(q["options"], list) or len(q["options"]) != 4:
                continue
            if not isinstance(q["correct_index"], int) or not 0 <= q["correct_index"] <= 3:
                continue
            if not str(q["question"]).strip() or any(not str(o).strip() for o in q["options"]):
                continue
            valid.append(q)

        if valid:
            return valid

        return [{
            "question": "Could not generate valid questions. Try again.",
            "options": ["Retry", "Retry", "Retry", "Retry"],
            "correct_index": 0,
            "explanation": "Validation failed."
        }]
    except Exception as e:
        print(f"Quiz generation error: {e}")
        return [{
            "question": "Error generating quiz",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
            "explanation": str(e)
        }]


# Gradio App
with gr.Blocks(css=get_flashcard_css() + get_quiz_css()) as demo:

    gr.Markdown("## 📄 Welcome to DOCY AI ")
    gr.Markdown("### Upload your PDF to Chat or Study with Flashcards.")

    with gr.Row():
        pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
        status = gr.Markdown("Please upload a PDF to start.")

    state = gr.State()
    flash_state = gr.State({"cards": [], "index": 0, "flipped": False})
    quiz_state = gr.State({"questions": [], "index": 0, "answers": {}, "finished": False})

    with gr.Tabs() as tabs:
        with gr.Tab("🔍 Chat Assistant"):
            question = gr.Textbox(label="Ask a question", lines=3)
            answer = gr.Textbox(label="Answer", lines=8)
            confidence_display = gr.Textbox(label="Confidence Score", interactive=False, lines=1)

            with gr.Row():
                run_btn = gr.Button("▶️ Run", variant="primary")
                stop_btn = gr.Button("⛔ Stop Thinking")
                clear_btn = gr.Button("🆕 New Question")

            gr.Markdown("---")
            gr.Markdown("#### ⭐ Rate this answer")
            with gr.Row():
                rating = gr.Radio(
                    choices=[("⭐ 1", 1), ("⭐⭐ 2", 2), ("⭐⭐⭐ 3", 3), ("⭐⭐⭐⭐ 4", 4), ("⭐⭐⭐⭐⭐ 5", 5)],
                    label="How helpful was this answer?",
                    type="value",
                    interactive=True,
                )
                submit_rating_btn = gr.Button("Submit Rating", variant="secondary")
            rating_status = gr.Markdown("")

        with gr.Tab("🗂️ Flashcard Study"):
            with gr.Column(visible=True) as flashcard_col:
                gen_btn = gr.Button("✨ Generate Study Cards", variant="secondary")

                card_display = gr.HTML(value=render_empty_card())

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    flip_btn = gr.Button("🔄 Flip / Show Answer")
                    next_btn = gr.Button("Next ➡️")

        with gr.Tab("📝 MCQ Quiz"):
            with gr.Row():
                quiz_topic = gr.Textbox(
                    label="Topic (optional)",
                    placeholder="e.g., 'RAG architecture' or leave empty",
                    scale=3
                )
                quiz_num = gr.Radio(
                    choices=[5, 10], value=5,
                    label="Number of questions", scale=1
                )

            quiz_gen_btn = gr.Button("🎯 Generate Quiz", variant="primary")

            quiz_header = gr.HTML(value=render_empty_quiz())

            opt_a_btn = gr.Button("A", variant="secondary", interactive=False, visible=False)
            opt_b_btn = gr.Button("B", variant="secondary", interactive=False, visible=False)
            opt_c_btn = gr.Button("C", variant="secondary", interactive=False, visible=False)
            opt_d_btn = gr.Button("D", variant="secondary", interactive=False, visible=False)

            quiz_feedback = gr.HTML(value="")
            quiz_result = gr.HTML(value="")

            quiz_next_btn = gr.Button("Next Question ➡️", variant="primary")
            quiz_restart_btn = gr.Button("🔄 New Quiz")

    # ---------------- PDF Processing ----------------
    def process_pdf(file):
        text = load_pdf(file.name)
        chunks = chunk_text(text)
        index = create_vector_store(chunks)
        return (chunks, index), "✅ PDF indexed successfully!"

    pdf_file.upload(
        process_pdf,
        inputs=pdf_file,
        outputs=[state, status]
    )

    # ---------------- Question Logic ----------------
    def ask_question(question, state):
        if state is None:
            return "Please upload a PDF first.", ""
        chunks, index = state
        retrieved_chunks = retrieve(question, chunks, index)
        context = "\n\n".join(retrieved_chunks)
        answer_text = ask_llm(context, question)
        
        result = calculate_confidence(
            question, retrieved_chunks, embedding_model,
            answer=answer_text, client=client, model_name="llama-3.1-8b-instant"
        )
        score = result["score"]
        
        confidence_text = f"Confidence Score: {score}/100"
        if result["llm_score"] is not None:
            confidence_text += f"  (Embedding: {result['embedding_score']}/100 | LLM: {result['llm_score']}/100)"
            
        return answer_text, confidence_text

    def handle_submit_rating(question_val, answer_val, rating_val):
        if not answer_val:
            return "Please ask a question first."
        if rating_val is None:
            return "Please select a star rating before submitting."
        _, msg = save_feedback(question_val, answer_val, int(rating_val))
        return msg

    run_event = run_btn.click(ask_question, inputs=[question, state], outputs=[answer, confidence_display])
    stop_btn.click(None, None, None, cancels=[run_event])
    clear_btn.click(lambda: ("", "", "", None, ""), outputs=[question, answer, confidence_display, rating, rating_status])
    submit_rating_btn.click(handle_submit_rating, inputs=[question, answer, rating], outputs=[rating_status])

    # ---------------- Flashcard Logic ----------------
    def start_flashcards(state):
        if state is None:
            return [], 0, False, render_empty_card()
        chunks, _ = state
        # Use first 5 chunks for general context
        context = "\n\n".join(chunks[:5])
        cards = generate_flashcards(context)
        html = render_flashcard_html(
            cards[0]["question"], cards[0]["answer"], 1, len(cards), False
        )
        return cards, 0, False, html

    def handle_gen(state):
        cards, idx, flipped, html = start_flashcards(state)
        return {"cards": cards, "index": idx, "flipped": flipped}, html

    def handle_next(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_idx = (f_state["index"] + 1) % len(cards)
        new_state = {"cards": cards, "index": new_idx, "flipped": False}
        html = render_flashcard_html(
            cards[new_idx]["question"], cards[new_idx]["answer"],
            new_idx + 1, len(cards), False
        )
        return new_state, html

    def handle_prev(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_idx = (f_state["index"] - 1) % len(cards)
        new_state = {"cards": cards, "index": new_idx, "flipped": False}
        html = render_flashcard_html(
            cards[new_idx]["question"], cards[new_idx]["answer"],
            new_idx + 1, len(cards), False
        )
        return new_state, html

    def handle_flip(f_state):
        cards = f_state["cards"]
        if not cards:
            return f_state, render_empty_card()
        new_flipped = not f_state["flipped"]
        new_state = f_state.copy()
        new_state["flipped"] = new_flipped
        idx = f_state["index"]
        html = render_flashcard_html(
            cards[idx]["question"], cards[idx]["answer"],
            idx + 1, len(cards), new_flipped
        )
        return new_state, html

    gen_btn.click(handle_gen, inputs=[state], outputs=[flash_state, card_display])
    next_btn.click(handle_next, inputs=[flash_state], outputs=[flash_state, card_display])
    prev_btn.click(handle_prev, inputs=[flash_state], outputs=[flash_state, card_display])
    flip_btn.click(handle_flip, inputs=[flash_state], outputs=[flash_state, card_display])

    # ---------------- Quiz Logic ----------------
    LETTERS = ["A", "B", "C", "D"]

    def _option_label(letter, text, marker=""):
        """Build a button label like 'A — text' or '✓ A — text'."""
        prefix = f"{marker} " if marker else ""
        return f"{prefix}{letter} — {text}"

    def _quiz_outputs(
        q_state,
        header_html,
        feedback_html,
        result_html,
        opts_visible,
        opts_interactive,
        a_label,
        b_label,
        c_label,
        d_label,
        next_visible,
        restart_visible,
    ):
        return (
            q_state,
            gr.update(value=header_html),
            gr.update(value=feedback_html),
            gr.update(value=result_html),
            gr.update(value=a_label, visible=opts_visible, interactive=opts_interactive),
            gr.update(value=b_label, visible=opts_visible, interactive=opts_interactive),
            gr.update(value=c_label, visible=opts_visible, interactive=opts_interactive),
            gr.update(value=d_label, visible=opts_visible, interactive=opts_interactive),
            gr.update(visible=next_visible),
            gr.update(visible=restart_visible),
        )

    def _initial_labels(options):
        return [_option_label(LETTERS[i], options[i]) for i in range(4)]

    def _marked_labels(options, selected_idx, correct_idx):
        labels = []
        for i in range(4):
            if i == correct_idx:
                marker = "✓"
            elif i == selected_idx:
                marker = "✗"
            else:
                marker = ""
            labels.append(_option_label(LETTERS[i], options[i], marker))
        return labels

    def handle_quiz_gen(p_state, topic, num):
        empty_state = {"questions": [], "index": 0, "answers": {}, "finished": False}

        if p_state is None:
            return _quiz_outputs(
                empty_state,
                "<div class='qz-scene'><div class='qz-empty'>⚠️ Please upload a PDF first.</div></div>",
                "",
                "",
                False, False,
                "A", "B", "C", "D",
                False, False,
            )

        chunks, index = p_state
        topic_clean = (topic or "").strip()
        if topic_clean:
            retrieved = retrieve(topic_clean, chunks, index, k=5)
            context = "\n\n".join(retrieved)
        else:
            context = "\n\n".join(chunks[:5])

        questions = generate_mcq_quiz(
            context,
            num_questions=int(num),
            topic=topic_clean if topic_clean else None,
        )

        new_state = {
            "questions": questions,
            "index": 0,
            "answers": {},
            "finished": False,
        }
        first = questions[0]
        header = render_quiz_header_html(first["question"], 1, len(questions))
        labels = _initial_labels(first["options"])
        return _quiz_outputs(
            new_state,
            header, "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False,
        )

    def handle_answer(q_state, selected_idx):
        questions = q_state.get("questions", [])
        if not questions or q_state.get("finished"):
            # No active quiz — keep everything as-is (no-op-ish)
            return (
                q_state,
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
            )

        idx = q_state["index"]
        new_answers = dict(q_state["answers"])
        new_answers[idx] = selected_idx
        new_state = {**q_state, "answers": new_answers}

        q = questions[idx]
        header = render_quiz_header_html(q["question"], idx + 1, len(questions))
        feedback = render_quiz_feedback_html(
            selected_idx=selected_idx,
            correct_idx=q["correct_index"],
            options=q["options"],
            explanation=q.get("explanation"),
        )
        labels = _marked_labels(q["options"], selected_idx, q["correct_index"])
        return _quiz_outputs(
            new_state,
            header, feedback, "",
            True, False,
            labels[0], labels[1], labels[2], labels[3],
            True, False,   # next_visible=True, restart_visible=False
        )

    def handle_quiz_next(q_state):
        questions = q_state.get("questions", [])
        if not questions:
            return _quiz_outputs(
                q_state,
                render_empty_quiz(), "", "",
                False, False,
                "A", "B", "C", "D",
                False, False,
            )

        next_idx = q_state["index"] + 1
        if next_idx >= len(questions):
            score = sum(
                1 for i, q in enumerate(questions)
                if q_state["answers"].get(i) == q["correct_index"]
            )
            results = []
            for i, q in enumerate(questions):
                sel = q_state["answers"].get(i)
                results.append({
                    "question": q["question"],
                    "options": q["options"],
                    "correct_index": q["correct_index"],
                    "selected_index": sel,
                    "explanation": q.get("explanation", ""),
                    "is_correct": sel == q["correct_index"],
                })
            result_html = render_quiz_result_html(score, len(questions), results)
            new_state = {**q_state, "finished": True}
            return _quiz_outputs(
                new_state,
                "", "", result_html,
                False, False,
                "A", "B", "C", "D",
                False, True,
            )

        new_state = {**q_state, "index": next_idx}
        q = questions[next_idx]
        header = render_quiz_header_html(q["question"], next_idx + 1, len(questions))
        labels = _initial_labels(q["options"])
        return _quiz_outputs(
            new_state,
            header, "", "",
            True, True,
            labels[0], labels[1], labels[2], labels[3],
            False, False,
        )

    def handle_quiz_restart():
        return _quiz_outputs(
            {"questions": [], "index": 0, "answers": {}, "finished": False},
            render_empty_quiz(), "", "",
            False, False,
            "A", "B", "C", "D",
            False, False,
        )

    quiz_outputs_list = [
        quiz_state,
        quiz_header, quiz_feedback, quiz_result,
        opt_a_btn, opt_b_btn, opt_c_btn, opt_d_btn,
        quiz_next_btn, quiz_restart_btn,
    ]

    quiz_gen_btn.click(
        handle_quiz_gen,
        inputs=[state, quiz_topic, quiz_num],
        outputs=quiz_outputs_list,
    )
    opt_a_btn.click(
        lambda q: handle_answer(q, 0),
        inputs=[quiz_state], outputs=quiz_outputs_list,
    )
    opt_b_btn.click(
        lambda q: handle_answer(q, 1),
        inputs=[quiz_state], outputs=quiz_outputs_list,
    )
    opt_c_btn.click(
        lambda q: handle_answer(q, 2),
        inputs=[quiz_state], outputs=quiz_outputs_list,
    )
    opt_d_btn.click(
        lambda q: handle_answer(q, 3),
        inputs=[quiz_state], outputs=quiz_outputs_list,
    )
    quiz_next_btn.click(
        handle_quiz_next,
        inputs=[quiz_state], outputs=quiz_outputs_list,
    )
    quiz_restart_btn.click(
        handle_quiz_restart,
        inputs=None, outputs=quiz_outputs_list,
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
