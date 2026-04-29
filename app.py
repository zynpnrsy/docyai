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



# Gradio App
with gr.Blocks(css=get_flashcard_css()) as demo:

    gr.Markdown("## 📄 Welcome to DOCY AI ")
    gr.Markdown("### Upload your PDF to Chat or Study with Flashcards.")

    with gr.Row():
        pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
        status = gr.Markdown("Please upload a PDF to start.")

    state = gr.State()
    flash_state = gr.State({"cards": [], "index": 0, "flipped": False})

    with gr.Tabs() as tabs:
        with gr.Tab("🔍 Chat Assistant"):
            question = gr.Textbox(label="Ask a question", lines=3)
            answer = gr.Textbox(label="Answer", lines=8)
            confidence_display = gr.Textbox(label="Confidence Score", interactive=False, lines=1)

            with gr.Row():
                run_btn = gr.Button("▶️ Run", variant="primary")
                stop_btn = gr.Button("⛔ Stop Thinking")
                clear_btn = gr.Button("🆕 New Question")

        with gr.Tab("🗂️ Flashcard Study"):
            with gr.Column(visible=True) as flashcard_col:
                gen_btn = gr.Button("✨ Generate Study Cards", variant="secondary")

                card_display = gr.HTML(value=render_empty_card())

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    flip_btn = gr.Button("🔄 Flip / Show Answer")
                    next_btn = gr.Button("Next ➡️")

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

    run_event = run_btn.click(ask_question, inputs=[question, state], outputs=[answer, confidence_display])
    stop_btn.click(None, None, None, cancels=[run_event])
    clear_btn.click(lambda: ("", "", ""), outputs=[question, answer, confidence_display])

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

demo.launch()
