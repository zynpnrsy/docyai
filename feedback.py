"""
feedback.py
Handles saving user ratings.

Storage strategy:
  - Try Google Sheets when credentials are available
    (GOOGLE_CREDENTIALS env var with raw JSON, or a credentials.json file).
  - Fall back to a local CSV (feedback_local.csv) when Sheets is not configured
    or fails. Ratings are never lost just because credentials are missing.
"""

import csv
import json
import os
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

LOCAL_CSV = "feedback_local.csv"

_sheet = None
_sheets_unavailable_reason: str | None = None  # cached so we don't retry credentials each call


def _credentials_available() -> bool:
    if os.environ.get("GOOGLE_CREDENTIALS"):
        return True
    creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    return os.path.exists(creds_file)


def _get_sheet():
    global _sheet
    if _sheet is not None:
        return _sheet

    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "DocAI Feedback")
    creds_raw = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds_raw:
        creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
        with open(creds_file, "r", encoding="utf-8") as f:
            creds_raw = f.read()

    creds_info = json.loads(creds_raw)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    _sheet = client.open(sheet_name).sheet1
    if _sheet.row_count == 0 or not _sheet.row_values(1):
        _sheet.append_row(["Question", "Answer", "Rating", "Timestamp"])
    return _sheet


def _save_local(question: str, answer: str, rating: int, timestamp: str) -> None:
    new_file = not os.path.exists(LOCAL_CSV)
    with open(LOCAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Question", "Answer", "Rating", "Timestamp"])
        writer.writerow([question, answer, int(rating), timestamp])


def save_feedback(question: str, answer: str, rating: int) -> tuple[bool, str]:
    """
    Persist a rating. Returns (success, user-facing message).
    """
    global _sheets_unavailable_reason

    if not question or not answer:
        return False, "No question/answer to rate yet."
    if rating not in range(1, 6):
        return False, "Rating must be between 1 and 5."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Try Google Sheets only when credentials look configured.
    if _credentials_available() and _sheets_unavailable_reason is None:
        try:
            sheet = _get_sheet()
            sheet.append_row([question, answer, int(rating), timestamp])
            return True, f"✅ Rating ({rating}⭐) saved to Google Sheets!"
        except Exception as e:
            # Cache the failure so we don't retry every click.
            _sheets_unavailable_reason = str(e)
            print(f"[feedback] Google Sheets unavailable: {e} — falling back to local CSV.")

    # Fallback: local CSV.
    try:
        _save_local(question, answer, rating, timestamp)
        if _sheets_unavailable_reason:
            return True, f"✅ Rating ({rating}⭐) saved locally (Sheets unavailable)."
        return True, f"✅ Rating ({rating}⭐) saved locally to {LOCAL_CSV}."
    except Exception as e:
        return False, f"❌ Could not save rating: {e}"
