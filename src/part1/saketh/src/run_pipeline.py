"""Kaggle-compatible Part 1 runner.

The script preserves the input row order, creates one isolated conversation per
conversation_id, writes exactly the required submission columns, and keeps
timing/index diagnostics in separate JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rag import GroundedAnswerer, HybridRetriever, enrich_figure_descriptions, load_records, new_conversation, records_to_json


SEED = 266
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REQUIRED_INPUT_COLUMNS = {"id", "conversation_id", "turn", "question_id", "question"}
SUBMISSION_COLUMNS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return "unknown"


def _clean_csv_list(values: list[str]) -> str:
    return ",".join(dict.fromkeys(value for value in values if value)) or "NONE"


def _format_row(row: pd.Series, answer: dict) -> dict[str, str]:
    return {
        "id": str(row["id"]),
        "conversation_id": str(row["conversation_id"]),
        "question_id": str(row["question_id"]),
        "answer": " ".join(str(answer.get("answer", "")).split())[:1200],
        "figure_references": _clean_csv_list(answer.get("figure_references", [])),
        "citations": _clean_csv_list(answer.get("citations", [])),
        "status": str(answer.get("status", "not_found")),
    }


def _safe_failure(row: pd.Series, error: Exception) -> dict[str, str]:
    print(f"[warning] failed question {row['question_id']}: {error}")
    return _format_row(row, {
        "answer": "The system could not safely produce a grounded answer from the supplied documents.",
        "status": "not_found",
        "citations": [],
        "figure_references": [],
    })


def run(data_dir: str | Path, out_dir: str | Path, questions_path: str | Path | None = None) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = Path(questions_path) if questions_path else Path(os.environ.get("QUESTIONS_PATH", data_dir / "Questions.csv"))
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_INPUT_COLUMNS - set(questions.columns)
    if missing:
        raise ValueError(f"Questions.csv is missing columns: {sorted(missing)}")

    records = enrich_figure_descriptions(load_records(data_dir, out_dir / "figures"), os.environ.get("VISION_MODEL_NAME") or None)
    if not records:
        raise FileNotFoundError("No PDF records found. Copy the three Part 1 PDFs into part1/data.")
    records_to_json(records, out_dir / "records.json")

    embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    dense_weight = float(os.environ.get("DENSE_WEIGHT", "0.68"))
    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
    answerer = GroundedAnswerer(os.environ.get("ANSWER_MODEL_NAME") or None)
    retriever = HybridRetriever(records, model_name=embedding_model, batch_size=batch_size, dense_weight=dense_weight)
    vector_store = retriever.persist_vector_db(out_dir / "chroma")

    output_rows: list[dict[str, str]] = []
    timings: list[float] = []
    for conversation_id, turns in questions.groupby("conversation_id", sort=False):
        chat = new_conversation(retriever, answerer)
        turns = turns.copy()
        turns["_turn_number"] = pd.to_numeric(turns["turn"], errors="coerce").fillna(0).astype(int)
        for _, row in turns.sort_values("_turn_number", kind="stable").iterrows():
            started = time.perf_counter()
            try:
                answer = chat.answer(str(row["question"]))
                output_rows.append(_format_row(row, answer))
            except Exception as error:  # A single malformed request must not kill the Kaggle run.
                output_rows.append(_safe_failure(row, error))
            timings.append(time.perf_counter() - started)

    submission = pd.DataFrame(output_rows, columns=SUBMISSION_COLUMNS)
    # The assignment requires the original Questions.csv order, not group order.
    submission = submission.set_index("id").reindex(questions["id"].tolist()).reset_index()
    if submission["id"].isna().any() or len(submission) != len(questions):
        raise RuntimeError("Submission row count/order does not match Questions.csv")
    submission.to_csv(out_dir / "submission.csv", index=False)

    metrics = {
        "status": "computed_from_real_data",
        "commit": _git_commit(),
        "seed": SEED,
        "n_questions": int(len(submission)),
        "records_indexed": int(len(records)),
        "median_seconds_per_question": float(np.median(timings)) if timings else 0.0,
        "embedding_model": embedding_model,
        "answer_model": os.environ.get("ANSWER_MODEL_NAME", "deterministic_fallback"),
        "answer_model_loaded": bool(answerer.model.model is not None),
        "answer_model_dtype": answerer.model.dtype_name,
        "answer_model_generated_answers": int(answerer.model.generated_count),
        "answer_model_fallback_answers": int(answerer.fallback_count),
        "answer_model_invalid_json_answers": int(answerer.invalid_json_count),
        "answer_model_error": answerer.model.load_error,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "submission_columns": list(submission.columns),
        "vector_store": vector_store,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("DATA_DIR", "part1/data"))
    parser.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "part1/outputs"))
    parser.add_argument("--questions", default=os.environ.get("QUESTIONS_PATH", ""))
    args = parser.parse_args()
    metrics = run(args.data, args.out, args.questions or None)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
