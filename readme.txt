
https://huggingface.co/spaces/zeyneppinarsoy/docyai/tree/main


An AI-powered PDF assistant that transforms static documents into an interactive learning experience using Retrieval-Augmented Generation (RAG).
The system allows users to query PDF documents, generate flashcards, create multiple-choice quizzes, and evaluate answer reliability — all based strictly on the uploaded content.

Features
 Ask questions based on PDF content
 Generate flashcards from key information
 Create MCQ quizzes from selected topics
 Confidence score for each response
 User rating system with feedback storage (Google Sheets)


 Tech Stack
LLM: Groq API / Ollama
Embedding: Sentence Transformers
Vector DB: FAISS
UI: Gradio
Storage: Google Sheets API

Pipeline
PDF → Chunking → Embedding → FAISS → Retrieval → LLM → Output + Evaluation
