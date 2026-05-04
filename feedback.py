"""
feedback.py
Handles saving user ratings to Google Sheets.

Setup:
  1. Create a Google Cloud service account and download credentials JSON.
  2. Create a Google Sheet and share it with the service account email.
  3. Set env vars:
       GOOGLE_CREDENTIALS_FILE  (default: credentials.json)
       GOOGLE_SHEET_NAME        (default: DocAI Feedback)
"""

import json
import os
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

_sheet = None


def _get_sheet():
    global _sheet
    if _sheet is not None:
        return _sheet

    creds_raw = os.environ.get("GOOGLE_CREDENTIALS")
    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "DocAI Feedback")

    if not creds_raw:
        raise ValueError("GOOGLE_CREDENTIALS environment variable is not set.")

    creds_info = json.loads(creds_raw)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    _sheet = client.open(sheet_name).sheet1

    # Add header row if the sheet is empty
    if _sheet.row_count == 0 or not _sheet.row_values(1):
        _sheet.append_row(["Question", "Answer", "Rating", "Timestamp"])

    return _sheet


def save_feedback(question: str, answer: str, rating: int) -> tuple[bool, str]:
    """
    Append a feedback row to Google Sheets.

    Returns
    -------
    (success: bool, message: str)
    """
    if not question or not answer:
        return False, "No question/answer to rate yet."
    if rating not in range(1, 6):
        return False, "Rating must be between 1 and 5."

    try:
        sheet = _get_sheet()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([question, answer, int(rating), timestamp])
        return True, f"✅ Rating ({rating}⭐) saved to Google Sheets!"
    except FileNotFoundError as e:
        return False, f"❌ {e}"
    except Exception as e:
        print(f"[feedback] Failed to save rating: {e}")
        return False, f"❌ Could not save rating: {e}"
