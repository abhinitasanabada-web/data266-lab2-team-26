"""Strict local checker matching the Part 1 submission contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


COLUMNS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]
STATUSES = {"answered", "not_found", "conflict", "false_premise", "session_override"}


def validate(path: str | Path, questions: str | Path | None = None) -> None:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS: raise ValueError(f"columns must be exactly {COLUMNS}")
    if frame["id"].duplicated().any(): raise ValueError("duplicate ids")
    if not frame["status"].isin(STATUSES).all(): raise ValueError("invalid status")
    if (frame["answer"].str.split().str.len() > 120).any(): raise ValueError("an answer exceeds 120 words")
    for column in ["figure_references", "citations"]:
        if frame[column].eq("").any() or frame[column].isna().any(): raise ValueError(f"empty {column}; use NONE")
    if questions:
        expected = pd.read_csv(questions, dtype=str, keep_default_na=False)
        if frame["id"].tolist() != expected["id"].tolist(): raise ValueError("row ids/order differs from Questions.csv")
    print(f"valid: {len(frame)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("submission"); parser.add_argument("--questions", default="")
    args = parser.parse_args(); validate(args.submission, args.questions or None)
