"""
quiz_ui.py
Provides CSS and HTML rendering helpers for the MCQ Quiz component.
"""

from typing import List, Optional, Dict, Any


def get_quiz_css() -> str:
    """Return the CSS string for the MCQ quiz component."""
    return """
/* ── Quiz scene wrapper ────────────────────────────────── */
.qz-scene {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 28px 16px;
    background: transparent;
}

/* ── Main quiz card ────────────────────────────────────── */
.qz-card {
    width: 640px;
    max-width: 100%;
    background: #1e293b;
    border: 1px solid rgba(99,120,160,0.25);
    border-radius: 18px;
    padding: 28px 32px 32px;
    box-sizing: border-box;
    filter: drop-shadow(0 8px 32px rgba(0,0,0,0.45));
    transition: border-color 0.3s;
}

.qz-card:hover {
    border-color: rgba(99,179,237,0.45);
}

/* ── Header row ────────────────────────────────────────── */
.qz-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.qz-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0;
}

.qz-badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #94a3b8;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
}

/* ── Question text ─────────────────────────────────────── */
.qz-question {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.55;
    margin: 0 0 22px 0;
}

/* ── Options list ──────────────────────────────────────── */
.qz-options {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.qz-option {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #e2e8f0;
    transition: all 0.25s ease;
}

.qz-letter {
    flex: 0 0 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: rgba(255,255,255,0.06);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    color: #94a3b8;
    letter-spacing: 0.04em;
}

.qz-option-text {
    flex: 1;
    line-height: 1.45;
}

.qz-mark {
    flex: 0 0 auto;
    font-size: 16px;
    font-weight: 700;
    opacity: 0;
}

/* Correct answer highlight (green) */
.qz-option-correct {
    background: rgba(34,211,122,0.12);
    border-color: rgba(74,222,128,0.55);
    color: #d1fae5;
}
.qz-option-correct .qz-letter {
    background: rgba(74,222,128,0.25);
    color: #4ade80;
}
.qz-option-correct .qz-mark {
    color: #4ade80;
    opacity: 1;
}

/* Wrong selected answer (red) */
.qz-option-wrong {
    background: rgba(239,68,68,0.12);
    border-color: rgba(248,113,113,0.55);
    color: #fee2e2;
}
.qz-option-wrong .qz-letter {
    background: rgba(248,113,113,0.25);
    color: #f87171;
}
.qz-option-wrong .qz-mark {
    color: #f87171;
    opacity: 1;
}

/* ── Explanation box ───────────────────────────────────── */
.qz-explanation {
    margin-top: 22px;
    padding: 16px 18px;
    background: linear-gradient(135deg, #162032 0%, #1e293b 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 12px;
}

.qz-exp-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22d3ee;
    margin: 0 0 8px 0;
    opacity: 0.85;
}

.qz-exp-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
    margin: 0;
}

/* ── Result screen ─────────────────────────────────────── */
.qz-result-card {
    width: 720px;
    max-width: 100%;
    background: #1e293b;
    border: 1px solid rgba(99,120,160,0.25);
    border-radius: 18px;
    padding: 32px;
    box-sizing: border-box;
    filter: drop-shadow(0 8px 32px rgba(0,0,0,0.45));
}

.qz-final-score {
    text-align: center;
    padding: 18px 0 28px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 22px;
}

.qz-score-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0 0 6px 0;
}

.qz-score-num {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 700;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}

.qz-score-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #94a3b8;
    margin: 4px 0 0 0;
    letter-spacing: 0.04em;
}

.qz-summary {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.qz-summary-item {
    display: flex;
    gap: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
}

.qz-summary-item.correct {
    border-left: 3px solid #4ade80;
}
.qz-summary-item.wrong {
    border-left: 3px solid #f87171;
}

.qz-summary-icon {
    flex: 0 0 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 14px;
}
.qz-summary-item.correct .qz-summary-icon {
    background: rgba(74,222,128,0.18);
    color: #4ade80;
}
.qz-summary-item.wrong .qz-summary-icon {
    background: rgba(248,113,113,0.18);
    color: #f87171;
}

.qz-summary-content {
    flex: 1;
}

.qz-summary-q {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0 0 6px 0;
    line-height: 1.45;
}

.qz-summary-detail {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #94a3b8;
    margin: 2px 0;
    line-height: 1.55;
}

.qz-summary-detail .ok { color: #4ade80; }
.qz-summary-detail .ko { color: #f87171; }

.qz-summary-exp {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #cbd5e1;
    margin: 8px 0 0 0;
    padding-top: 8px;
    border-top: 1px dashed rgba(255,255,255,0.08);
    line-height: 1.55;
    font-style: italic;
}

/* ── Empty state ───────────────────────────────────────── */
.qz-empty {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #64748b;
    text-align: center;
    padding: 60px 24px;
}
"""


def _esc(s: str) -> str:
    """Escape minimal HTML entities to prevent injection."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_quiz_header_html(question: str, index: int, total: int) -> str:
    """
    Render only the question header (question text + progress badge).

    Used when option choices are real Gradio buttons rendered separately
    below this header.
    """
    badge = f"{index} / {total}"
    pct = round(((index - 1) / total) * 100) if total else 0
    return f"""
<div class="qz-scene">
  <div class="qz-card">
    <div class="qz-header">
      <p class="qz-label">Question</p>
      <span class="qz-badge">{badge}</span>
    </div>
    <div style="height:6px;background:#0f172a;border-radius:999px;overflow:hidden;margin-bottom:18px">
      <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#4ade80,#22d3ee);
                  transition:width 0.4s ease"></div>
    </div>
    <p class="qz-question">{_esc(question)}</p>
  </div>
</div>
"""


def render_quiz_feedback_html(
    selected_idx: Optional[int],
    correct_idx: int,
    options: List[str],
    explanation: Optional[str] = None,
) -> str:
    """
    Render the feedback panel shown after the user picks an option.

    Returns an empty string when `selected_idx` is None (no answer yet),
    so the same component can be cleared by binding its value to "".
    """
    if selected_idx is None:
        return ""

    letters = ["A", "B", "C", "D"]
    is_correct = selected_idx == correct_idx

    if is_correct:
        verdict_cls = "qz-option-correct"
        verdict_icon = "✓"
        verdict_label = "Correct!"
    else:
        verdict_cls = "qz-option-wrong"
        verdict_icon = "✗"
        verdict_label = "Incorrect"

    sel_text = _esc(options[selected_idx])
    cor_text = _esc(options[correct_idx])

    your_line = (
        f"<p class='qz-summary-detail'>Your answer: "
        f"<span class='{'ok' if is_correct else 'ko'}'>"
        f"{letters[selected_idx]}. {sel_text}</span></p>"
    )
    correct_line = (
        f"<p class='qz-summary-detail'>Correct: "
        f"<span class='ok'>{letters[correct_idx]}. {cor_text}</span></p>"
    )

    explanation_html = ""
    if explanation:
        explanation_html = f"""
    <div class="qz-explanation">
      <p class="qz-exp-label">Explanation</p>
      <p class="qz-exp-text">{_esc(explanation)}</p>
    </div>"""

    return f"""
<div class="qz-scene">
  <div class="qz-card" style="padding: 20px 24px;">
    <div class="qz-option {verdict_cls}" style="margin-bottom: 12px;">
      <span class="qz-letter">{verdict_icon}</span>
      <span class="qz-option-text"><strong>{verdict_label}</strong></span>
    </div>
    {your_line}
    {correct_line}{explanation_html}
  </div>
</div>
"""


def render_quiz_question_html(
    question: str,
    options: List[str],
    index: int,
    total: int,
    selected_idx: Optional[int] = None,
    correct_idx: Optional[int] = None,
    explanation: Optional[str] = None,
) -> str:
    """
    Render a single quiz question card.

    Parameters
    ----------
    question     : str            – Question text.
    options      : List[str]      – Exactly 4 option strings.
    index        : int            – 1-based question number.
    total        : int            – Total number of questions.
    selected_idx : Optional[int]  – Index (0-3) the user picked, or None if unanswered.
    correct_idx  : Optional[int]  – Correct option index, used when selected_idx is set.
    explanation  : Optional[str]  – Explanation text shown after answering.
    """
    badge = f"{index} / {total}"
    answered = selected_idx is not None and correct_idx is not None
    letters = ["A", "B", "C", "D"]

    option_html_parts = []
    for i, opt in enumerate(options):
        cls = "qz-option"
        mark = ""
        if answered:
            if i == correct_idx:
                cls += " qz-option-correct"
                mark = "✓"
            elif i == selected_idx:
                cls += " qz-option-wrong"
                mark = "✗"
        option_html_parts.append(
            f"""
      <div class="{cls}">
        <span class="qz-letter">{letters[i]}</span>
        <span class="qz-option-text">{_esc(opt)}</span>
        <span class="qz-mark">{mark}</span>
      </div>"""
        )
    options_html = "".join(option_html_parts)

    explanation_html = ""
    if answered and explanation:
        explanation_html = f"""
    <div class="qz-explanation">
      <p class="qz-exp-label">Explanation</p>
      <p class="qz-exp-text">{_esc(explanation)}</p>
    </div>"""

    return f"""
<div class="qz-scene">
  <div class="qz-card">
    <div class="qz-header">
      <p class="qz-label">Question</p>
      <span class="qz-badge">{badge}</span>
    </div>
    <p class="qz-question">{_esc(question)}</p>
    <div class="qz-options">{options_html}
    </div>{explanation_html}
  </div>
</div>
"""


def _format_seconds(s: float) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    sec = int(round(s - m * 60))
    return f"{m}m {sec}s"


def render_quiz_result_html(
    score: int,
    total: int,
    questions_with_results: List[Dict[str, Any]],
    total_seconds: Optional[float] = None,
) -> str:
    """
    Render the final result screen.

    Parameters
    ----------
    score                  : int  – Number of correct answers.
    total                  : int  – Total number of questions.
    questions_with_results : list – Per-question dicts with keys:
        question, options, correct_index, selected_index, explanation, is_correct.
    """
    pct = round((score / total) * 100) if total else 0
    letters = ["A", "B", "C", "D"]

    items_html_parts = []
    for i, r in enumerate(questions_with_results):
        is_correct = r.get("is_correct", False)
        wrap_cls = "correct" if is_correct else "wrong"
        icon = "✓" if is_correct else "✗"
        sel_idx = r.get("selected_index")
        cor_idx = r["correct_index"]

        if sel_idx is None:
            sel_label = "<span class='ko'>Skipped</span>"
        else:
            sel_text = _esc(r["options"][sel_idx])
            sel_class = "ok" if is_correct else "ko"
            sel_label = (
                f"<span class='{sel_class}'>{letters[sel_idx]}. {sel_text}</span>"
            )

        cor_text = _esc(r["options"][cor_idx])
        cor_label = f"<span class='ok'>{letters[cor_idx]}. {cor_text}</span>"

        exp_html = ""
        if r.get("explanation"):
            exp_html = (
                f"<p class='qz-summary-exp'>{_esc(r['explanation'])}</p>"
            )

        time_html = ""
        if r.get("time_seconds") is not None:
            time_html = (
                f"<p class='qz-summary-detail' style='color:#64748b'>"
                f"⏱ {_format_seconds(r['time_seconds'])}</p>"
            )

        items_html_parts.append(
            f"""
      <div class="qz-summary-item {wrap_cls}">
        <span class="qz-summary-icon">{icon}</span>
        <div class="qz-summary-content">
          <p class="qz-summary-q">Q{i+1}. {_esc(r['question'])}</p>
          <p class="qz-summary-detail">Your answer: {sel_label}</p>
          <p class="qz-summary-detail">Correct: {cor_label}</p>
          {time_html}
          {exp_html}
        </div>
      </div>"""
        )
    items_html = "".join(items_html_parts)

    time_summary = ""
    if total_seconds is not None and total_seconds > 0:
        avg = total_seconds / total if total else 0
        time_summary = (
            f"<p class='qz-score-pct'>⏱ {_format_seconds(total_seconds)} total · "
            f"{_format_seconds(avg)} avg per question</p>"
        )

    return f"""
<div class="qz-scene">
  <div class="qz-result-card">
    <div class="qz-final-score">
      <p class="qz-score-label">Final Score</p>
      <p class="qz-score-num">{score} / {total}</p>
      <p class="qz-score-pct">{pct}% correct</p>
      {time_summary}
    </div>
    <div class="qz-summary">{items_html}
    </div>
  </div>
</div>
"""


def render_empty_quiz() -> str:
    """Return placeholder HTML when no quiz has been generated yet."""
    return """
<div class="qz-scene">
  <div class="qz-empty">
    🎯 Click <strong>Generate Quiz</strong> to test your knowledge!
  </div>
</div>
"""
