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
    target_sources: dict[str, str]
    item_summary: dict[str, int]


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
    target_sources: dict[str, str] = {}

    def add_target(entity: str, value: str, source: str) -> None:
        text = (value or "").strip()
        if not text:
            return
        if entity in target_values:
            return
        target_values[entity] = text
        target_sources[entity] = source

    if sample.ground_truth is not None:
        add_target("VENDOR_NAME", sample.ground_truth.vendor.name, "gold_sroie")
        add_target("ADDRESS", sample.ground_truth.vendor.address, "gold_sroie")
        add_target("DATE", sample.ground_truth.invoice.date, "gold_sroie")
        if sample.ground_truth.totals.total > 0:
            add_target("TOTAL", f"{sample.ground_truth.totals.total:.2f}", "gold_sroie")

    # Pseudo-label missing fields from parser output.
    add_target("REG_NO", parsed.vendor.registration_number, "pseudo_rules")
    add_target("INVOICE_TYPE", parsed.invoice.invoice_type, "pseudo_rules")
    add_target("BILL_NO", parsed.invoice.bill_number, "pseudo_rules")
    add_target("ORDER_NO", parsed.invoice.order_number, "pseudo_rules")
    add_target("TABLE_NO", parsed.invoice.table_number, "pseudo_rules")
    add_target("TIME", parsed.invoice.time, "pseudo_rules")
    add_target("CASHIER", parsed.invoice.cashier, "pseudo_rules")
    if parsed.totals.subtotal > 0:
        add_target("SUBTOTAL", f"{parsed.totals.subtotal:.2f}", "pseudo_rules")
    if parsed.totals.tax > 0:
        add_target("TAX", f"{parsed.totals.tax:.2f}", "pseudo_rules")
    add_target("PAYMENT_METHOD", parsed.payment.method, "pseudo_rules")

    line_to_indexes = _line_to_indexes(sample.ocr_tokens)
    _label_scalar_fields(sample.ocr_lines, sample.ocr_tokens, labels, target_values, line_to_indexes)
    item_summary = _label_item_fields(sample.ocr_lines, sample.ocr_tokens, labels, parsed.items, line_to_indexes)
    _enforce_valid_bio(labels)

    return WeakLabelingResult(
        labels=labels,
        assumptions=assumptions,
        target_values=target_values,
        target_sources=target_sources,
        item_summary=item_summary,
    )


def _normalize_token(text: str) -> str:
    value = (text or "").upper()
    value = re.sub(r"[^A-Z0-9]+", "", value)
    return value


def _token_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    left_norm = left.replace("0", "O").replace("1", "I")
    right_norm = right.replace("0", "O").replace("1", "I")
    return left_norm == right_norm


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
        candidate = token_norms[idx : idx + window]
        if all(_token_equivalent(a, b) for a, b in zip(candidate, phrase_tokens)):
            spans.append((idx, idx + window))
    return spans


def _line_to_indexes(tokens: list[OCRToken]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for idx, token in enumerate(tokens):
        if token.line_id is None:
            continue
        mapping.setdefault(int(token.line_id), []).append(idx)
    return mapping


def _label_scalar_fields(
    lines: list[OCRLine],
    tokens: list[OCRToken],
    labels: list[str],
    target_values: dict[str, str],
    line_to_indexes: dict[int, list[int]],
) -> None:
    line_biased_entities = {"BILL_NO", "ORDER_NO", "TABLE_NO", "TIME", "CASHIER", "SUBTOTAL", "TAX", "PAYMENT_METHOD"}
    for entity, value in target_values.items():
        phrase_tokens = _phrase_to_tokens(value)
        if not phrase_tokens:
            continue

        # Prefer line-local exact/fuzzy alignment for header-like and totals-like fields.
        if entity in line_biased_entities:
            if _label_from_line_context(lines, tokens, labels, phrase_tokens, entity, line_to_indexes):
                continue

        for start, end in _find_spans(tokens, phrase_tokens):
            if _can_write_span(labels, start, end):
                _write_span(labels, start, end, entity)
                break


def _label_from_line_context(
    lines: list[OCRLine],
    tokens: list[OCRToken],
    labels: list[str],
    phrase_tokens: list[str],
    entity: str,
    line_to_indexes: dict[int, list[int]],
) -> bool:
    for line in lines:
        idxs = line_to_indexes.get(line.line_id, [])
        if not idxs:
            continue
        line_tokens = [_normalize_token(tokens[idx].text) for idx in idxs]
        window = len(phrase_tokens)
        for rel in range(0, len(line_tokens) - window + 1):
            candidate = line_tokens[rel : rel + window]
            if not all(_token_equivalent(a, b) for a, b in zip(candidate, phrase_tokens)):
                continue
            start = idxs[rel]
            end = idxs[rel + window - 1] + 1
            if _can_write_span(labels, start, end):
                _write_span(labels, start, end, entity)
                return True
    return False


def _format_float_candidates(value: float) -> set[str]:
    if value <= 0:
        return set()
    base = f"{value:.2f}"
    alt = base.rstrip("0").rstrip(".")
    return {base, alt, base.replace(".", ""), base.replace(".", ",")}


def _label_item_fields(
    lines: list[OCRLine],
    tokens: list[OCRToken],
    labels: list[str],
    items: Iterable[ReceiptItem],
    line_to_indexes: dict[int, list[int]],
) -> dict[str, int]:
    item_list = list(items)
    if not item_list:
        return {"items_detected": 0, "items_labeled": 0, "items_missing_price": 0, "name_spans": 0, "price_spans": 0}

    labeled_items = 0
    missing_price = 0
    name_spans = 0
    price_spans = 0

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

        item_name_written = False
        if not matched_line_ids:
            for start, end in _find_spans(tokens, item_name_tokens):
                if _can_write_span(labels, start, end):
                    _write_span(labels, start, end, "ITEM_NAME")
                    name_spans += 1
                    item_name_written = True
                    break
        else:
            for line_id in matched_line_ids:
                idxs = line_to_indexes.get(line_id, [])
                if not idxs:
                    continue
                token_norms = [_normalize_token(tokens[idx].text) for idx in idxs]
                name_norm = item_name_tokens
                window = len(name_norm)
                for rel in range(0, len(token_norms) - window + 1):
                    candidate = token_norms[rel : rel + window]
                    if all(_token_equivalent(a, b) for a, b in zip(candidate, name_norm)):
                        start = idxs[rel]
                        end = idxs[rel + window - 1] + 1
                        if _can_write_span(labels, start, end):
                            _write_span(labels, start, end, "ITEM_NAME")
                            name_spans += 1
                            item_name_written = True
                            break

        # Label quantity and price with line-awareness and nearest-number fallback.
        candidate_lines = matched_line_ids or [line.line_id for line in lines]
        qty_candidates = _format_float_candidates(item.quantity)
        price_candidates = _format_float_candidates(item.line_total) | _format_float_candidates(item.unit_price)
        _label_numeric_entity(tokens, labels, candidate_lines, line_to_indexes, qty_candidates, "ITEM_QTY")
        price_written = _label_numeric_entity(tokens, labels, candidate_lines, line_to_indexes, price_candidates, "ITEM_PRICE")
        if price_written:
            price_spans += 1
        else:
            missing_price += 1
        if item_name_written:
            labeled_items += 1

    return {
        "items_detected": len(item_list),
        "items_labeled": labeled_items,
        "items_missing_price": missing_price,
        "name_spans": name_spans,
        "price_spans": price_spans,
    }


def _label_numeric_entity(
    tokens: list[OCRToken],
    labels: list[str],
    candidate_lines: list[int],
    line_to_indexes: dict[int, list[int]],
    candidates: set[str],
    entity: str,
) -> bool:
    if not candidates:
        return False
    candidate_norms = {_normalize_token(candidate) for candidate in candidates if candidate}
    for line_id in candidate_lines:
        idxs = line_to_indexes.get(line_id, [])
        for idx in idxs:
            if labels[idx] != "O":
                continue
            norm = _normalize_token(tokens[idx].text)
            if norm in candidate_norms:
                labels[idx] = f"B-{entity}"
                return True
    return False


def _enforce_valid_bio(labels: list[str]) -> None:
    prev_entity = ""
    prev_prefix = "O"
    for idx, label in enumerate(labels):
        if label == "O":
            prev_prefix = "O"
            prev_entity = ""
            continue
        prefix, entity = split_bio(label)
        if prefix == "O":
            labels[idx] = "O"
            prev_prefix = "O"
            prev_entity = ""
            continue
        if prefix == "I" and (prev_prefix == "O" or prev_entity != entity):
            labels[idx] = f"B-{entity}"
            prefix = "B"
        prev_prefix = prefix
        prev_entity = entity
