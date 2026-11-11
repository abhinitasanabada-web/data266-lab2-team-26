"""Notebook-friendly evaluation for the Part 1 golden set.

The evaluator consumes a committed ``golden_set.json``.  It never invents
labels or metrics when that file is absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rag import GroundedAnswerer, HybridRetriever, new_conversation, safe_records


NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?%?")


def _numbers(text: str) -> list[float]:
    values = []
    for token in NUMBER_RE.findall(text):
        try:
            values.append(float(token.rstrip("%").replace(",", "")))
        except ValueError:
            pass
    return values


def _answer_correct(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    if answer.get("status") != expected.get("status"):
        return False
    expected_numbers = _numbers(str(expected.get("answer", "")))
    actual_numbers = _numbers(str(answer.get("answer", "")))
    if expected_numbers:
        if len(actual_numbers) < len(expected_numbers):
            return False
        for target in expected_numbers:
            if not any(abs(value - target) <= max(0.01 * max(abs(target), 1.0), 0.01) for value in actual_numbers):
                return False
    accepted = expected.get("accepted_phrases", [])
    if accepted and not any(str(phrase).lower() in str(answer.get("answer", "")).lower() for phrase in accepted):
        return False
    return True


def evaluate(golden_path: str | Path, retriever: HybridRetriever, answerer: GroundedAnswerer) -> dict[str, Any]:
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    if not isinstance(golden, list) or len(golden) != 25:
        raise ValueError("golden_set.json must be a list of exactly 25 original questions")
    answers = []
    for item in golden:
        conversation = new_conversation(retriever, answerer)
        for prior in item.get("conversation_prefix", []):
            conversation.answer(prior)
        question = str(item["question"])
        answer = conversation.answer(question)
        hits = retriever.search(question, k=5)
        trusted = safe_records(hits)
        expected_citations = set(item.get("citations", []))
        predicted_citations = set(answer.get("citations", []))
        citation_accuracy = bool(predicted_citations) and predicted_citations.issubset(expected_citations)
        target_pages = expected_citations | set(item.get("figure_references", []))
        recall = any(hit.record.citation in target_pages or hit.record.label in target_pages for hit in hits)
        answers.append({
            "question": question,
            "correct": _answer_correct(answer, item),
            "recall_at_5": recall,
            "citation_accuracy": citation_accuracy,
            "answer": answer,
            "trusted_hits": len(trusted),
        })
    result = {
        "n": len(answers),
        "correct": int(sum(row["correct"] for row in answers)),
        "answer_accuracy": float(np.mean([row["correct"] for row in answers])),
        "recall_at_5": float(np.mean([row["recall_at_5"] for row in answers])),
        "citation_accuracy": float(np.mean([row["citation_accuracy"] for row in answers])),
        "answers": answers,
        "status": "computed_from_committed_golden_set",
    }
    return result


def human_summary(path: str | Path) -> dict[str, Any]:
    ratings = pd.read_csv(path)
    required = {"question_id", "member_1", "member_2"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Human ratings missing columns: {sorted(missing)}")
    values = ratings[["member_1", "member_2"]].apply(pd.to_numeric, errors="coerce")
    return {
        "average_member_1": float(values["member_1"].mean()),
        "average_member_2": float(values["member_2"].mean()),
        "average": float(values.stack().mean()),
        "agreement_within_one": float((values["member_1"].sub(values["member_2"]).abs() <= 1).mean()),
        "n": int(len(values)),
    }
