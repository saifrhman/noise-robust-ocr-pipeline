from __future__ import annotations

from typing import Iterable


ENTITY_NAMES: list[str] = [
    "VENDOR_NAME",
    "ADDRESS",
    "REG_NO",
    "INVOICE_TYPE",
    "BILL_NO",
    "ORDER_NO",
    "TABLE_NO",
    "DATE",
    "TIME",
    "CASHIER",
    "ITEM_NAME",
    "ITEM_QTY",
    "ITEM_PRICE",
    "SUBTOTAL",
    "TAX",
    "TOTAL",
    "PAYMENT_METHOD",
]

SEMANTIC_BIO_LABELS: list[str] = ["O"] + [f"{prefix}-{entity}" for entity in ENTITY_NAMES for prefix in ("B", "I")]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(SEMANTIC_BIO_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

# Support decoding from legacy checkpoints already in this repo.
LEGACY_ENTITY_ALIASES: dict[str, str] = {
    "COMPANY": "VENDOR_NAME",
    "VENDOR": "VENDOR_NAME",
    "MERCHANT": "VENDOR_NAME",
    "ADDRESS": "ADDRESS",
    "REG_NO": "REG_NO",
    "REG": "REG_NO",
    "INVOICE_TYPE": "INVOICE_TYPE",
    "BILL_NO": "BILL_NO",
    "ORDER_NO": "ORDER_NO",
    "TABLE_NO": "TABLE_NO",
    "DATE": "DATE",
    "TIME": "TIME",
    "CASHIER": "CASHIER",
    "ITEM_NAME": "ITEM_NAME",
    "ITEM_QTY": "ITEM_QTY",
    "ITEM_PRICE": "ITEM_PRICE",
    "SUBTOTAL": "SUBTOTAL",
    "TAX": "TAX",
    "TOTAL": "TOTAL",
    "PAYMENT_METHOD": "PAYMENT_METHOD",
}

FIELD_TO_RESULT_KEY: dict[str, str] = {
    "VENDOR_NAME": "vendor.name",
    "ADDRESS": "vendor.address",
    "REG_NO": "vendor.registration_number",
    "INVOICE_TYPE": "invoice.invoice_type",
    "BILL_NO": "invoice.bill_number",
    "ORDER_NO": "invoice.order_number",
    "TABLE_NO": "invoice.table_number",
    "DATE": "invoice.date",
    "TIME": "invoice.time",
    "CASHIER": "invoice.cashier",
    "SUBTOTAL": "totals.subtotal",
    "TAX": "totals.tax",
    "TOTAL": "totals.total",
    "PAYMENT_METHOD": "payment.method",
}


def normalize_model_label(label: str) -> str:
    value = (label or "O").strip().upper()
    if not value or value == "O":
        return "O"
    if value.startswith("LABEL_"):
        return "O"

    prefix = "B"
    entity = value
    if value.startswith("B-"):
        prefix = "B"
        entity = value[2:]
    elif value.startswith("I-"):
        prefix = "I"
        entity = value[2:]

    entity = LEGACY_ENTITY_ALIASES.get(entity, entity)
    candidate = f"{prefix}-{entity}"
    return candidate if candidate in LABEL2ID else "O"


def split_bio(label: str) -> tuple[str, str]:
    normalized = normalize_model_label(label)
    if normalized == "O":
        return "O", "O"
    return normalized[:1], normalized[2:]


def sanitize_labels(labels: Iterable[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        normalized = normalize_model_label(label)
        out.append(normalized if normalized in LABEL2ID else "O")
    return out


def has_semantic_receipt_labels(id2label: dict[int, str] | dict[str, str] | None) -> bool:
    if not id2label:
        return False
    normalized = {normalize_model_label(str(v)) for v in id2label.values()}
    normalized.discard("O")
    return bool(normalized)
