from __future__ import annotations

from typing import Iterable

# Fixed semantic BIO schema for receipt key information extraction.
SEMANTIC_LABELS: list[str] = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(SEMANTIC_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

FIELD_ALIASES: dict[str, str] = {
    "COMPANY": "merchant",
    "DATE": "date",
    "ADDRESS": "address",
    "TOTAL": "total",
}

RECEIPT_ENTITY_TYPES: set[str] = set(FIELD_ALIASES.keys())


def normalize_label_name(label: str) -> str:
    """Normalize a model label to uppercase BIO form."""
    value = (label or "O").strip().upper()
    return value or "O"


def split_bio(label: str) -> tuple[str, str]:
    """Split BIO label into (prefix, entity_type)."""
    value = normalize_label_name(label)
    if value == "O":
        return "O", "O"
    if value.startswith("B-"):
        return "B", value[2:]
    if value.startswith("I-"):
        return "I", value[2:]
    return "B", value


def is_generic_label(label: str) -> bool:
    value = normalize_label_name(label)
    return value.startswith("LABEL_")


def has_semantic_receipt_labels(id2label: dict[int, str] | dict[str, str] | None) -> bool:
    """
    Check whether a checkpoint has semantic receipt labels instead of generic LABEL_N labels.
    """
    if not id2label:
        return False

    labels = {normalize_label_name(str(v)) for v in id2label.values()}
    if not labels:
        return False
    if any(is_generic_label(label) for label in labels):
        return False

    required = {"B-COMPANY", "B-DATE", "B-ADDRESS", "B-TOTAL"}
    return bool(required.intersection(labels))


def sanitize_labels(labels: Iterable[str]) -> list[str]:
    """Force unknown labels to 'O' to keep training/inference robust."""
    out: list[str] = []
    for label in labels:
        normalized = normalize_label_name(label)
        out.append(normalized if normalized in LABEL2ID else "O")
    return out
