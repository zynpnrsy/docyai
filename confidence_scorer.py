"""
confidence_scorer.py
Computes a confidence / reliability score for RAG-generated answers.

Two-phase approach:
  A) Embedding Similarity  -- cosine similarity between the user question
     and the retrieved document chunks.
  B) LLM Self-Evaluation   -- asks the LLM to rate how well its own answer
     is supported by the provided context.

The public entry point is `calculate_confidence()`.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Phase A: Embedding-based confidence
# ---------------------------------------------------------------------------

def compute_embedding_confidence(question, retrieved_chunks, embedding_model):
    """
    Compute a 0-100 confidence score based on cosine similarity between
    the question embedding and each retrieved chunk embedding.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_chunks : list[str]
        The top-k text chunks returned by the retriever.
    embedding_model : SentenceTransformer
        The same model instance used elsewhere in the pipeline.

    Returns
    -------
    int
        Score in the range [0, 100].
    """
    if not retrieved_chunks:
        return 0

    # Encode question and chunks independently (retrieve() is not touched)
    question_emb = embedding_model.encode([question])          # shape (1, dim)
    chunk_embs = embedding_model.encode(retrieved_chunks)      # shape (k, dim)

    # Cosine similarities: shape (1, k) -> flatten to (k,)
    similarities = cosine_similarity(question_emb, chunk_embs)[0]

    # Mean similarity scaled to 0-100
    mean_sim = float(np.mean(similarities))

    # Cosine similarity for normalised sentence-transformer vectors is
    # typically in [0.0, 1.0]; clamp just in case.
    score = int(round(max(0.0, min(1.0, mean_sim)) * 100))
    return score


# ---------------------------------------------------------------------------
# Phase B: LLM self-evaluation
# ---------------------------------------------------------------------------

def compute_llm_confidence(context, answer, client, model_name):
    """
    Ask the LLM to self-evaluate how well its answer is supported by the
    given context.  Returns a 0-100 integer, or None on failure.

    Parameters
    ----------
    context : str
        The concatenated retrieved chunks that were fed to the LLM.
    answer : str
        The LLM-generated answer.
    client : groq.Groq
        Initialised Groq client (uses the existing env-var API key).
    model_name : str
        Model identifier, e.g. "llama-3.1-8b-instant".

    Returns
    -------
    int | None
        Score in [0, 100], or None if parsing fails.
    """
    eval_prompt = (
        "You are an objective evaluator. Given the context and the answer below, "
        "rate how well the answer is supported by the context. "
        "Return ONLY a single integer between 0 and 100. "
        "Do not include any other text.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score:"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Extract the first integer found in the response
        match = re.search(r"\d+", raw)
        if match:
            value = int(match.group())
            return max(0, min(100, value))
        return None
    except Exception as exc:
        print(f"[confidence_scorer] LLM self-eval failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_confidence(
    question,
    retrieved_chunks,
    embedding_model,
    answer=None,
    client=None,
    model_name=None,
):
    """
    Compute an overall confidence score for a RAG answer.

    Parameters
    ----------
    question : str
        The user's original question.
    retrieved_chunks : list[str]
        Text chunks returned by the retriever.
    embedding_model : SentenceTransformer
        Shared embedding model instance.
    answer : str | None
        The LLM-generated answer (needed for Phase B).
    client : groq.Groq | None
        Groq client instance (needed for Phase B).
    model_name : str | None
        LLM model name (needed for Phase B).

    Returns
    -------
    dict
        {
            "score": int,            # final combined score 0-100
            "embedding_score": int,  # Phase A score
            "llm_score": int | None  # Phase B score (None if skipped)
        }
    """
    # Phase A -- always runs
    embedding_score = compute_embedding_confidence(
        question, retrieved_chunks, embedding_model
    )

    # Phase B -- runs only when answer + client are available
    llm_score = None
    if answer and client and model_name:
        context = "\n\n".join(retrieved_chunks)
        llm_score = compute_llm_confidence(context, answer, client, model_name)

    # Combine scores
    if llm_score is not None:
        final_score = int(round((embedding_score + llm_score) / 2))
    else:
        final_score = embedding_score

    return {
        "score": final_score,
        "embedding_score": embedding_score,
        "llm_score": llm_score,
    }
