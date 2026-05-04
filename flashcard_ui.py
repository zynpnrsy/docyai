"""
flashcard_ui.py
Provides CSS and HTML rendering helpers for the animated flashcard component.
"""


def get_flashcard_css() -> str:
    """Return the CSS string for the 3D-flip flashcard component."""
    return """
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Flashcard scene wrapper ───────────────────────────── */
.fc-scene {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 32px 16px;
    background: transparent;
}

/* ── Perspective container ─────────────────────────────── */
.fc-card {
    width: 500px;
    height: 280px;
    perspective: 1200px;
    cursor: pointer;
}

/* ── Rotating inner card ───────────────────────────────── */
.fc-inner {
    position: relative;
    width: 100%;
    height: 100%;
    transform-style: preserve-3d;
    -webkit-transform-style: preserve-3d;
    transition: transform 0.65s cubic-bezier(0.4, 0.2, 0.2, 1);
    border-radius: 18px;
    /* NOTE: do not set `filter` / `clip-path` / `mask` here — they flatten the
       3D context and break the flip, leaving the front face mirrored. Shadow
       is applied per-face below instead. */
}

/* Flip state: back face visible */
.fc-inner.flipped {
    transform: rotateY(180deg);
}

/* ── Shared face styles ────────────────────────────────── */
.fc-face {
    position: absolute;
    inset: 0;
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 28px 36px;
    box-sizing: border-box;
    background: #1e293b;
    border: 1px solid rgba(99,120,160,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.55);
    transition: border-color 0.3s;
}

/* Hover glow on the whole card */
.fc-card:hover .fc-face {
    border-color: rgba(99,179,237,0.55);
    box-shadow: 0 0 24px rgba(99,179,237,0.18) inset;
}

/* ── Front face ────────────────────────────────────────── */
.fc-front {
    /* front stays as-is (no extra transform) */
}

/* badge: "2 / 5" */
.fc-badge {
    position: absolute;
    top: 14px;
    right: 18px;
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

/* label above question */
.fc-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
}

/* question text */
.fc-question {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
    text-align: center;
    line-height: 1.55;
    margin: 0;
}

/* hint text at bottom of front */
.fc-hint {
    position: absolute;
    bottom: 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #334155;
    letter-spacing: 0.05em;
}

/* ── Back face ─────────────────────────────────────────── */
.fc-back {
    transform: rotateY(180deg);
    background: linear-gradient(135deg, #162032 0%, #1e293b 100%);
}

/* answer label */
.fc-answer-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22d37a;
    margin-bottom: 12px;
    opacity: 0.8;
}

/* answer text */
.fc-answer {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 400;
    color: #4ade80;
    text-align: center;
    line-height: 1.6;
    margin: 0;
}

/* thin accent line on back */
.fc-divider {
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    border-radius: 999px;
    margin-bottom: 16px;
}

/* ── Empty / loading state ─────────────────────────────── */
.fc-empty {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #64748b;
    text-align: center;
    padding: 60px 24px;
}
"""


def render_flashcard_html(
    question: str,
    answer: str,
    index: int,
    total: int,
    flipped: bool,
) -> str:
    """
    Return a fully self-contained HTML string for a single flashcard.

    Parameters
    ----------
    question : str   – Question text shown on the front face.
    answer   : str   – Answer text shown on the back face.
    index    : int   – 1-based card number (e.g. 2).
    total    : int   – Total number of cards (e.g. 5).
    flipped  : bool  – True → back face visible; False → front face visible.
    """
    flip_class = "flipped" if flipped else ""
    badge = f"{index} / {total}"

    # Escape minimal HTML entities to prevent injection
    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    q_safe = _esc(question)
    a_safe = _esc(answer)

    pct = round((index / total) * 100) if total else 0
    return f"""
<div class="fc-scene">
  <div style="display:flex;flex-direction:column;align-items:center;gap:14px">
    <div class="fc-card">
      <div class="fc-inner {flip_class}">

        <!-- FRONT FACE -->
        <div class="fc-face fc-front">
          <span class="fc-badge">{badge}</span>
          <p class="fc-label">Question</p>
          <p class="fc-question">{q_safe}</p>
          <span class="fc-hint">click flip to reveal answer</span>
        </div>

        <!-- BACK FACE -->
        <div class="fc-face fc-back">
          <span class="fc-badge">{badge}</span>
          <p class="fc-answer-label">Answer</p>
          <div class="fc-divider"></div>
          <p class="fc-answer">{a_safe}</p>
        </div>

      </div>
    </div>
    <div style="width:500px">
      <div style="height:6px;background:#0f172a;border-radius:999px;overflow:hidden">
        <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#4ade80,#22d3ee);
                    transition:width 0.4s ease"></div>
      </div>
      <p style="margin:6px 0 0 0;font-family:'DM Sans',sans-serif;font-size:11px;
                color:#64748b;letter-spacing:.08em;text-align:center">
        Card {index} of {total}
      </p>
    </div>
  </div>
</div>
"""


def render_empty_card() -> str:
    """Return placeholder HTML when no cards have been generated yet."""
    return """
<div class="fc-scene">
  <div class="fc-empty">
    ✨ Click <strong>Generate Study Cards</strong> to get started!
  </div>
</div>
"""


def render_flashcard_summary(cards, known_idx, unknown_idx) -> str:
    """Render a study summary after the deck has been reviewed."""
    total = len(cards)
    n_known = len(known_idx)
    n_unknown = len(unknown_idx)
    pct = round(n_known / total * 100) if total else 0

    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    weak_html = ""
    if unknown_idx:
        items = "".join(
            f"<li style='margin:6px 0;color:#cbd5e1'>"
            f"<span style='color:#f87171;font-weight:600'>•</span> "
            f"{_esc(cards[i]['question'])}</li>"
            for i in unknown_idx
        )
        weak_html = (
            "<p style='font-family:DM Sans,sans-serif;font-size:11px;letter-spacing:.12em;"
            "text-transform:uppercase;color:#f87171;margin:18px 0 8px 0'>Review these</p>"
            f"<ul style='margin:0;padding-left:18px'>{items}</ul>"
        )

    return f"""
<div class="fc-scene">
  <div style="width:520px;background:#1e293b;border:1px solid rgba(99,120,160,0.25);
              border-radius:18px;padding:28px 32px;box-shadow:0 8px 32px rgba(0,0,0,0.55);
              font-family:'DM Sans',sans-serif;color:#f1f5f9">
    <p style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;margin:0 0 6px 0'>
      Study Summary</p>
    <p style='margin:0;font-size:14px;color:#94a3b8'>You knew <strong style='color:#4ade80'>{n_known}</strong>
       of <strong>{total}</strong> ({pct}%)</p>
    <div style='margin-top:14px;height:8px;background:#0f172a;border-radius:999px;overflow:hidden'>
      <div style='height:100%;width:{pct}%;background:linear-gradient(90deg,#4ade80,#22d3ee)'></div>
    </div>
    {weak_html}
  </div>
</div>
"""
