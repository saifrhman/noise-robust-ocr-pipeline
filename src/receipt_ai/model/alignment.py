from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from src.receipt_ai.parsing.rules_parser import RuleBasedReceiptParser
from src.receipt_ai.schemas import OCRLine, OCRToken, ReceiptItem, ReceiptSample
from .labels import split_bio


@dataclass(slots=True)
class WeakLabelingResult:
    labels: list[str]
    assumptions: list[str]
    target_values: dict[str, str]


def build_weak_bio_labels(sample: ReceiptSample) -> WeakLabelingResult:
    """
    Build practical weak BIO labels for training from SROIE annotations + parser output.

    Assumptions:
    - SROIE provides gold `company`, `address`, `date`, `total` only.
    - Other fields are pseudo-labeled from the rule parser.
    - Item fields are line-aware pseudo labels derived from parser output.
    """
    parser = RuleBasedReceiptParser()
    parsed = parser.parse(sample.ocr_lines, source_image=sample.image_path, mode="weak_labeling")

    labels = ["O"] * len(sample.ocr_tokens)
    assumptions: list[str] = [
        "Ground truth token labels are not provided directly in SROIE.",
        "company/address/date/total use dataset annotations when available.",
        "remaining fields are pseudo-labeled from the rule parser.",
        "item labels are assigned with line-aware heuristics.",
    ]

    target_values: dict[str, str] = {}
    if sample.ground_truth is not None:
        if sample.ground_truth.vendor.name:
            target_values["VENDOR_NAME"] = sample.ground_truth.vendor.name
        if sample.ground_truth.vendor.address:
            target_values["ADDRESS"] = sample.ground_truth.vendor.address
        if sample.ground_truth.invoice.date:
            target_values["DATE"] = sample.ground_truth.invoice.date
        if sample.ground_truth.totals.total > 0:
            target_values["TOTAL"] = f"{sample.ground_truth.totals.total:.2f}"

    # Pseudo-label missing fields from parser output.
    if parsed.vendor.registration_number:
        target_values.setdefault("REG_NO", parsed.vendor.registration_number)
    if parsed.invoice.invoice_type:
        target_values.setdefault("INVOICE_TYPE", parsed.invoice.invoice_type)
    if parsed.invoice.bill_number:
        target_values.setdefault("BILL_NO", parsed.invoice.bill_number)
    if parsed.invoice.order_number:
        target_values.setdefault("ORDER_NO", parsed.invoice.order_number)
    if parsed.invoice.table_number:
        target_values.setdefault("TABLE_NO", parsed.invoice.table_number)
    if parsed.invoice.time:
        target_values.setdefault("TIME", parsed.invoice.time)
    if parsed.invoice.cashier:
        target_values.setdefault("CASHIER", parsed.invoice.cashier)
    if parsed.totals.subtotal > 0:
        target_values.setdefault("SUBTOTAL", f"{parsed.totals.subtotal:.2f}")
    if parsed.totals.tax > 0:
        target_values.setdefault("TAX", f"{parsed.totals.tax:.2f}")
    if parsed.payment.method:
        target_values.setdefault("PAYMENT_METHOD", parsed.payment.method)

    _label_scalar_fields(sample.ocr_tokens, labels, target_values)
    _label_item_fields(sample.ocr_lines, sample.ocr_tokens, labels, parsed.items)

    return WeakLabelingResult(labels=labels, assumptions=assumptions, target_values=target_values)


def _normalize_token(text: str) -> str:
    value = (text or "").upper()
    value = value.replace("0", "O")
    value = re.sub(r"[^A-Z0-9]+", "", value)
    return value


def _phrase_to_tokens(text: str) -> list[str]:
    return [tok for tok in (_normalize_token(part) for part in (text or "").split()) if tok]


def _can_write_span(labels: list[str], start: int, end: int) -> bool:
    return all(label == "O" for label in labels[start:end])


def _write_span(labels: list[str], start: int, end: int, entity: str) -> None:
    if start >= end:
        return
    labels[start] = f"B-{entity}"
    for idx in range(start + 1, end):
        labels[idx] = f"I-{entity}"


def _find_spans(tokens: list[OCRToken], phrase_tokens: list[str]) -> list[tuple[int, int]]:
    if not phrase_tokens:
        return []
    token_norms = [_normalize_token(token.text) for token in tokens]
    spans: list[tuple[int, int]] = []
    window = len(phrase_tokens)
    for idx in range(0, len(token_norms) - window + 1):
        if token_norms[idx : idx + window] == phrase_tokens:
            spans.append((idx, idx + window))
    return spans


def _label_scalar_fields(tokens: list[OCRToken], labels: list[str], target_values: dict[str, str]) -> None:
    for entity, value in target_values.items():
        phrase_tokens = _phrase_to_tokens(value)
        if not phrase_tokens:
            continue
        for start, end in _find_spans(tokens, phrase_tokens):
            if _can_write_span(labels, start, end):
                _write_span(labels, start, end, entity)
                break


def _line_token_indexes(tokens: list[OCRToken], line_id: int) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if token.line_id == line_id]


def _format_float_candidates(value: float) -> set[str]:
    if value <= 0:
        return set()
    base = f"{value:.2f}"
    alt = base.rstrip("0").rstrip(".")
    return {base, alt, base.replace(".", ""), base.replace(".", ",")}


def _label_item_fields(lines: list[OCRLine], tokens: list[OCRToken], labels: list[str], items: Iterable[ReceiptItem]) -> None:
    item_list = list(items)
    if not item_list:
        return

    for item in item_list:
        item_name_tokens = _phrase_to_tokens(item.name)
        if not item_name_tokens:
            continue

        # Prefer exact line containment for item names.
        matched_line_ids: list[int] = []
        for line in lines:
            line_norm = _phrase_to_tokens(line.text)
            if not line_norm:
                continue
            joined = " ".join(line_norm)
            if " ".join(item_name_tokens) in joined:
                matched_line_ids.append(line.line_id)

        if not matched_line_ids:
            for start, end in _find_spans(tokens, item_name_tokens):
                if _can_write_span(labels, start, end):
                    _write_span(labels, start, end, "ITEM_NAME")
                    break
        else:
            for line_id in matched_line_ids:
                idxs = _line_token_indexes(tokens, line_id)
                if not idxs:
                    continue
                token_norms = [_normalize_token(tokens[idx].text) for idx in idxs]
                name_norm = item_name_tokens
                window = len(name_norm)
                for rel in range(0, len(token_norms) - window + 1):
                    if token_norms[rel : rel + window] == name_norm:
                        start = idxs[rel]
                        end = idxs[rel + window - 1] + 1
                        if _can_write_span(labels, start, end):
                            _write_span(labels, start, end, "ITEM_NAME")
                            break

        # Label quantity and price on nearby lines if exact text matches.
        qty_candidates = _format_float_candidates(item.quantity)
        price_candidates = _format_float_candidates(item.line_total) | _format_float_candidates(item.unit_price)
        for token_idx, token in enumerate(tokens):
            if labels[token_idx] != "O":
                continue
            norm = _normalize_token(token.text)
            if norm in {_normalize_token(candidate) for candidate in qty_candidates}:
                labels[token_idx] = "B-ITEM_QTY"
            elif norm in {_normalize_token(candidate) for candidate in price_candidates}:
                labels[token_idx] = "B-ITEM_PRICE"
