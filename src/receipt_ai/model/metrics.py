from __future__ import annotations

from collections import Counter
from typing import Any

import evaluate


def compute_token_classification_metrics(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    seqeval_metric = evaluate.load("seqeval")
    seqeval_result = seqeval_metric.compute(predictions=pred_sequences, references=gold_sequences)

    total = 0
    correct = 0
    label_tp: Counter[str] = Counter()
    label_fp: Counter[str] = Counter()
    label_fn: Counter[str] = Counter()

    for gold_row, pred_row in zip(gold_sequences, pred_sequences):
        for gold, pred in zip(gold_row, pred_row):
            total += 1
            if gold == pred:
                correct += 1
            if pred == gold and pred != "O":
                label_tp[pred] += 1
            elif pred != gold:
                if pred != "O":
                    label_fp[pred] += 1
                if gold != "O":
                    label_fn[gold] += 1

    per_label: dict[str, dict[str, float]] = {}
    all_labels = sorted(set(label_tp) | set(label_fp) | set(label_fn))
    for label in all_labels:
        tp = float(label_tp[label])
        fp = float(label_fp[label])
        fn = float(label_fn[label])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "token_accuracy": float(correct / total) if total > 0 else 0.0,
        "precision": float(seqeval_result.get("overall_precision", 0.0)),
        "recall": float(seqeval_result.get("overall_recall", 0.0)),
        "f1": float(seqeval_result.get("overall_f1", 0.0)),
        "overall_accuracy": float(seqeval_result.get("overall_accuracy", 0.0)),
        "per_label": per_label,
    }
