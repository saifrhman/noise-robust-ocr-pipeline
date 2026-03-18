from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from src.receipt_ai.parsing.rules_parser import RuleBasedReceiptParser
from src.receipt_ai.schemas import OCRLine, OCRToken, ReceiptItem, ReceiptSample
from .labels import CRITICAL_ENTITY_NAMES, split_bio


@dataclass(slots=True)
class WeakLabelingResult:
    labels: list[str]
    label_confidences: list[float]
    assumptions: list[str]
    target_values: dict[str, str]
    target_sources: dict[str, str]
    item_summary: dict[str, int]
    filtering_summary: dict[str, int]
    sample_quality: dict[str, float | int | bool | str]
    drop_training_sample: bool = False
    drop_reason: str = ""


@dataclass(slots=True)
class SpanProposal:
    start: int
    end: int
    entity: str
    confidence: float
    source: str
    reason: str


FIELD_BASE_CONFIDENCE: dict[str, float] = {
    "gold_sroie": 1.0,
    "pseudo_rules": 0.72,
}

ENTITY_CONFIDENCE_ADJUSTMENT: dict[str, float] = {
    "VENDOR_NAME": 0.12,
    "ADDRESS": 0.08,
    "DATE": 0.12,
    "TOTAL": 0.14,
    "SUBTOTAL": -0.02,
    "TAX": -0.04,
    "ITEM_NAME": -0.10,
    "ITEM_QTY": -0.06,
    "ITEM_PRICE": -0.02,
}

NOISY_SAMPLE_MAX_LABELED_RATIO = 0.45
NOISY_SAMPLE_MIN_LABELED_RATIO = 0.01
MIN_PSEUDO_SPAN_CONFIDENCE = 0.45


def build_weak_bio_labels(sample: ReceiptSample) -> WeakLabelingResult:
    """
    Build weak BIO labels with filtering and confidence weighting.

    The returned supervision is still compatible with current training code:
    - labels carry the discrete BIO targets
    - label_confidences carry per-token supervision strength
    - drop_training_sample flags severely noisy examples
    """
    parser = RuleBasedReceiptParser()
    parsed = parser.parse(sample.ocr_lines, source_image=sample.image_path, mode="weak_labeling")

    labels = ["O"] * len(sample.ocr_tokens)
    label_confidences = [1.0] * len(sample.ocr_tokens)
    assumptions: list[str] = [
        "Ground truth token labels are not provided directly in SROIE.",
        "company/address/date/total use dataset annotations when available.",
        "remaining fields are pseudo-labeled from the rule parser.",
        "low-confidence or contradictory pseudo spans are filtered before training.",
    ]

    target_values: dict[str, str] = {}
    target_sources: dict[str, str] = {}
    filtering_summary: dict[str, int] = {
        "spans_written": 0,
        "spans_filtered_low_confidence": 0,
        "spans_filtered_unreliable": 0,
        "spans_filtered_overlap": 0,
        "spans_filtered_ambiguous": 0,
        "spans_filtered_item_noise": 0,
        "spans_filtered_contradiction": 0,
        "tokens_labeled": 0,
        "tokens_downweighted": 0,
    }

    def add_target(entity: str, value: str, source: str) -> None:
        text = (value or "").strip()
        if not text or entity in target_values:
            return
        target_values[entity] = text
        target_sources[entity] = source

    if sample.ground_truth is not None:
        add_target("VENDOR_NAME", sample.ground_truth.vendor.name, "gold_sroie")
        add_target("ADDRESS", sample.ground_truth.vendor.address, "gold_sroie")
        add_target("DATE", sample.ground_truth.invoice.date, "gold_sroie")
        if sample.ground_truth.totals.total > 0:
            add_target("TOTAL", f"{sample.ground_truth.totals.total:.2f}", "gold_sroie")

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
    _label_scalar_fields(
        sample.ocr_lines,
        sample.ocr_tokens,
        labels,
        label_confidences,
        target_values,
        target_sources,
        line_to_indexes,
        filtering_summary,
    )
    item_summary = _label_item_fields(
        sample.ocr_lines,
        sample.ocr_tokens,
        labels,
        label_confidences,
        parsed.items,
        line_to_indexes,
        filtering_summary,
    )
    _enforce_valid_bio(labels)

    tokens_labeled = sum(1 for label in labels if label != "O")
    filtering_summary["tokens_labeled"] = tokens_labeled
    filtering_summary["tokens_downweighted"] = sum(
        1 for label, weight in zip(labels, label_confidences) if label != "O" and weight < 0.999
    )
    sample_quality = _build_sample_quality(labels, label_confidences, target_sources, filtering_summary, item_summary)
    drop_training_sample, drop_reason = _should_drop_sample(sample_quality)

    return WeakLabelingResult(
        labels=labels,
        label_confidences=label_confidences,
        assumptions=assumptions,
        target_values=target_values,
        target_sources=target_sources,
        item_summary=item_summary,
        filtering_summary=filtering_summary,
        sample_quality=sample_quality,
        drop_training_sample=drop_training_sample,
        drop_reason=drop_reason,
    )


def _normalize_token(text: str) -> str:
    value = (text or "").upper()
    value = re.sub(r"[^A-Z0-9]+", "", value)
    return value


def _token_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right or len(left) != len(right):
        return False
    confusion_pairs = {("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("5", "S"), ("S", "5")}
    mismatches = 0
    for lch, rch in zip(left, right):
        if lch == rch:
            continue
        if (lch, rch) in confusion_pairs:
            mismatches += 1
            continue
        return False
    return mismatches <= 1


def _phrase_to_tokens(text: str) -> list[str]:
    return [tok for tok in (_normalize_token(part) for part in (text or "").split()) if tok]


def _can_write_span(labels: list[str], start: int, end: int) -> bool:
    return start < end and all(label == "O" for label in labels[start:end])


def _write_span(
    labels: list[str],
    label_confidences: list[float],
    start: int,
    end: int,
    entity: str,
    confidence: float,
) -> None:
    labels[start] = f"B-{entity}"
    label_confidences[start] = confidence
    for idx in range(start + 1, end):
        labels[idx] = f"I-{entity}"
        label_confidences[idx] = confidence


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
    label_confidences: list[float],
    target_values: dict[str, str],
    target_sources: dict[str, str],
    line_to_indexes: dict[int, list[int]],
    filtering_summary: dict[str, int],
) -> None:
    line_biased_entities = {"BILL_NO", "ORDER_NO", "TABLE_NO", "TIME", "CASHIER", "SUBTOTAL", "TAX", "PAYMENT_METHOD"}
    for entity, value in target_values.items():
        phrase_tokens = _phrase_to_tokens(value)
        if not phrase_tokens:
            continue
        if not _entity_value_is_reliable(entity, phrase_tokens, value):
            filtering_summary["spans_filtered_unreliable"] += 1
            continue

        proposals: list[SpanProposal] = []
        if entity in line_biased_entities:
            proposals.extend(
                _line_context_proposals(
                    lines,
                    tokens,
                    phrase_tokens,
                    entity,
                    line_to_indexes,
                    source=target_sources.get(entity, "pseudo_rules"),
                )
            )
        proposals.extend(_global_proposals(tokens, phrase_tokens, entity, source=target_sources.get(entity, "pseudo_rules")))
        if len(phrase_tokens) == 1 and len(proposals) > 3:
            filtering_summary["spans_filtered_ambiguous"] += 1
            continue
        if not proposals:
            continue
        best = sorted(proposals, key=lambda proposal: proposal.confidence, reverse=True)[0]
        _try_apply_proposal(best, tokens, labels, label_confidences, filtering_summary)


def _line_context_proposals(
    lines: list[OCRLine],
    tokens: list[OCRToken],
    phrase_tokens: list[str],
    entity: str,
    line_to_indexes: dict[int, list[int]],
    *,
    source: str,
) -> list[SpanProposal]:
    proposals: list[SpanProposal] = []
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
            confidence = _proposal_confidence(entity, source, exact=True, line_local=True)
            proposals.append(
                SpanProposal(
                    start=start,
                    end=end,
                    entity=entity,
                    confidence=confidence,
                    source=source,
                    reason="line_context",
                )
            )
    return proposals


def _global_proposals(
    tokens: list[OCRToken],
    phrase_tokens: list[str],
    entity: str,
    *,
    source: str,
) -> list[SpanProposal]:
    proposals: list[SpanProposal] = []
    for start, end in _find_spans(tokens, phrase_tokens):
        confidence = _proposal_confidence(entity, source, exact=True, line_local=False)
        proposals.append(
            SpanProposal(
                start=start,
                end=end,
                entity=entity,
                confidence=confidence,
                source=source,
                reason="global_match",
            )
        )
    return proposals


def _proposal_confidence(entity: str, source: str, *, exact: bool, line_local: bool) -> float:
    score = FIELD_BASE_CONFIDENCE.get(source, 0.60) + ENTITY_CONFIDENCE_ADJUSTMENT.get(entity, 0.0)
    if exact:
        score += 0.08
    if line_local:
        score += 0.06
    return max(0.05, min(1.0, score))


def _try_apply_proposal(
    proposal: SpanProposal,
    tokens: list[OCRToken],
    labels: list[str],
    label_confidences: list[float],
    filtering_summary: dict[str, int],
) -> bool:
    if proposal.confidence < MIN_PSEUDO_SPAN_CONFIDENCE and proposal.source != "gold_sroie":
        filtering_summary["spans_filtered_low_confidence"] += 1
        return False
    if not _span_tokens_compatible(tokens, proposal.start, proposal.end, proposal.entity):
        filtering_summary["spans_filtered_contradiction"] += 1
        return False
    if not _can_write_span(labels, proposal.start, proposal.end):
        filtering_summary["spans_filtered_overlap"] += 1
        return False
    _write_span(labels, label_confidences, proposal.start, proposal.end, proposal.entity, proposal.confidence)
    filtering_summary["spans_written"] += 1
    return True


def _span_tokens_compatible(tokens: list[OCRToken], start: int, end: int, entity: str) -> bool:
    span_tokens = tokens[start:end]
    if not span_tokens:
        return False
    if entity == "ITEM_NAME":
        return all(_item_name_token_ok(token.text) for token in span_tokens)
    if entity in {"SUBTOTAL", "TAX", "TOTAL", "ITEM_PRICE", "ITEM_QTY"}:
        return all(_numeric_token_ok(token.text, entity) for token in span_tokens)
    if entity == "DATE":
        joined = " ".join(token.text for token in span_tokens)
        return bool(re.search(r"\d", joined)) and ("/" in joined or "-" in joined or "." in joined)
    if entity == "TIME":
        joined = " ".join(token.text for token in span_tokens)
        return bool(re.search(r"\d{1,2}[:.]\d{2}", joined))
    return True


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
    label_confidences: list[float],
    items: Iterable[ReceiptItem],
    line_to_indexes: dict[int, list[int]],
    filtering_summary: dict[str, int],
) -> dict[str, int]:
    item_list = list(items)
    if not item_list:
        return {
            "items_detected": 0,
            "items_labeled": 0,
            "items_missing_price": 0,
            "name_spans": 0,
            "price_spans": 0,
            "qty_spans": 0,
        }

    labeled_items = 0
    missing_price = 0
    name_spans = 0
    price_spans = 0
    qty_spans = 0

    for item in item_list:
        item_name_tokens = _phrase_to_tokens(item.name)
        if not item_name_tokens:
            continue
        if len(item_name_tokens) == 1 and len(item_name_tokens[0]) <= 2:
            filtering_summary["spans_filtered_item_noise"] += 1
            continue

        candidate_line_ids = _candidate_item_line_ids(lines, tokens, item_name_tokens, line_to_indexes)
        item_name_written = _label_item_name(
            tokens,
            labels,
            label_confidences,
            item_name_tokens,
            candidate_line_ids,
            line_to_indexes,
            filtering_summary,
        )
        if item_name_written:
            name_spans += 1

        qty_candidates = _format_float_candidates(item.quantity)
        price_candidates = _format_float_candidates(item.line_total) | _format_float_candidates(item.unit_price)

        qty_written = _label_item_numeric(
            tokens,
            labels,
            label_confidences,
            candidate_line_ids,
            line_to_indexes,
            qty_candidates,
            "ITEM_QTY",
            prefer_tail=False,
            filtering_summary=filtering_summary,
        )
        price_written = _label_item_numeric(
            tokens,
            labels,
            label_confidences,
            candidate_line_ids,
            line_to_indexes,
            price_candidates,
            "ITEM_PRICE",
            prefer_tail=True,
            filtering_summary=filtering_summary,
        )

        if qty_written:
            qty_spans += 1
        if price_written:
            price_spans += 1
        else:
            missing_price += 1
        if item_name_written and price_written:
            labeled_items += 1

    return {
        "items_detected": len(item_list),
        "items_labeled": labeled_items,
        "items_missing_price": missing_price,
        "name_spans": name_spans,
        "price_spans": price_spans,
        "qty_spans": qty_spans,
    }


def _candidate_item_line_ids(
    lines: list[OCRLine],
    tokens: list[OCRToken],
    item_name_tokens: list[str],
    line_to_indexes: dict[int, list[int]],
) -> list[int]:
    candidates: list[int] = []
    target = " ".join(item_name_tokens)
    for line in lines:
        idxs = line_to_indexes.get(line.line_id, [])
        if not idxs:
            continue
        line_tokens = [_normalize_token(tokens[idx].text) for idx in idxs if _normalize_token(tokens[idx].text)]
        if not line_tokens:
            continue
        joined = " ".join(line_tokens)
        if target in joined:
            candidates.append(line.line_id)
    if candidates:
        expanded: list[int] = []
        for line_id in candidates:
            expanded.append(line_id)
            expanded.append(line_id + 1)
        return list(dict.fromkeys(expanded))
    return [line.line_id for line in lines]


def _label_item_name(
    tokens: list[OCRToken],
    labels: list[str],
    label_confidences: list[float],
    item_name_tokens: list[str],
    candidate_line_ids: list[int],
    line_to_indexes: dict[int, list[int]],
    filtering_summary: dict[str, int],
) -> bool:
    target = " ".join(item_name_tokens)
    proposals: list[SpanProposal] = []
    for line_id in candidate_line_ids:
        idxs = line_to_indexes.get(line_id, [])
        token_norms = [_normalize_token(tokens[idx].text) for idx in idxs]
        window = len(item_name_tokens)
        for rel in range(0, len(token_norms) - window + 1):
            candidate = token_norms[rel : rel + window]
            if " ".join(candidate) != target and not all(_token_equivalent(a, b) for a, b in zip(candidate, item_name_tokens)):
                continue
            start = idxs[rel]
            end = idxs[rel + window - 1] + 1
            proposals.append(
                SpanProposal(
                    start=start,
                    end=end,
                    entity="ITEM_NAME",
                    confidence=_proposal_confidence("ITEM_NAME", "pseudo_rules", exact=True, line_local=True),
                    source="pseudo_rules",
                    reason="item_name",
                )
            )
    for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
        if _try_apply_proposal(proposal, tokens, labels, label_confidences, filtering_summary):
            return True
    return False


def _label_item_numeric(
    tokens: list[OCRToken],
    labels: list[str],
    label_confidences: list[float],
    candidate_line_ids: list[int],
    line_to_indexes: dict[int, list[int]],
    candidates: set[str],
    entity: str,
    *,
    prefer_tail: bool,
    filtering_summary: dict[str, int],
) -> bool:
    if not candidates:
        return False
    candidate_norms = {_normalize_token(candidate) for candidate in candidates if candidate}
    for line_id in candidate_line_ids:
        idxs = list(line_to_indexes.get(line_id, []))
        if prefer_tail:
            idxs = list(reversed(idxs))
        for idx in idxs:
            if labels[idx] != "O":
                continue
            norm = _normalize_token(tokens[idx].text)
            if norm not in candidate_norms or not _numeric_token_ok(tokens[idx].text, entity):
                continue
            confidence = _proposal_confidence(entity, "pseudo_rules", exact=True, line_local=True)
            proposal = SpanProposal(
                start=idx,
                end=idx + 1,
                entity=entity,
                confidence=confidence,
                source="pseudo_rules",
                reason="item_numeric",
            )
            if _try_apply_proposal(proposal, tokens, labels, label_confidences, filtering_summary):
                return True
    filtering_summary["spans_filtered_item_noise"] += 1
    return False


def _build_sample_quality(
    labels: list[str],
    label_confidences: list[float],
    target_sources: dict[str, str],
    filtering_summary: dict[str, int],
    item_summary: dict[str, int],
) -> dict[str, float | int | bool | str]:
    labeled = [label for label in labels if label != "O"]
    labeled_ratio = float(len(labeled) / max(len(labels), 1))
    avg_conf = float(
        sum(weight for label, weight in zip(labels, label_confidences) if label != "O") / max(len(labeled), 1)
    ) if labeled else 0.0
    pseudo_only = bool(target_sources) and set(target_sources.values()) == {"pseudo_rules"}
    critical_hits = sum(1 for entity in CRITICAL_ENTITY_NAMES if any(label.endswith(f"-{entity}") for label in labels))
    return {
        "labeled_tokens": len(labeled),
        "labeled_ratio": labeled_ratio,
        "avg_labeled_confidence": avg_conf,
        "pseudo_only": pseudo_only,
        "critical_fields_labeled": critical_hits,
        "filtered_spans": (
            filtering_summary.get("spans_filtered_low_confidence", 0)
            + filtering_summary.get("spans_filtered_unreliable", 0)
            + filtering_summary.get("spans_filtered_overlap", 0)
            + filtering_summary.get("spans_filtered_ambiguous", 0)
            + filtering_summary.get("spans_filtered_item_noise", 0)
            + filtering_summary.get("spans_filtered_contradiction", 0)
        ),
        "items_detected": int(item_summary.get("items_detected", 0)),
        "items_labeled": int(item_summary.get("items_labeled", 0)),
        "items_missing_price": int(item_summary.get("items_missing_price", 0)),
    }


def _should_drop_sample(sample_quality: dict[str, float | int | bool | str]) -> tuple[bool, str]:
    labeled_ratio = float(sample_quality.get("labeled_ratio", 0.0))
    avg_conf = float(sample_quality.get("avg_labeled_confidence", 0.0))
    critical_hits = int(sample_quality.get("critical_fields_labeled", 0))
    pseudo_only = bool(sample_quality.get("pseudo_only", False))
    items_detected = int(sample_quality.get("items_detected", 0))
    items_labeled = int(sample_quality.get("items_labeled", 0))
    items_missing_price = int(sample_quality.get("items_missing_price", 0))

    if labeled_ratio > NOISY_SAMPLE_MAX_LABELED_RATIO:
        return True, "over_labeled_noisy_sample"
    if pseudo_only and labeled_ratio < NOISY_SAMPLE_MIN_LABELED_RATIO:
        return True, "under_labeled_pseudo_only_sample"
    if avg_conf < 0.52 and critical_hits == 0:
        return True, "low_confidence_without_critical_fields"
    if items_detected >= 3 and items_labeled == 0 and items_missing_price >= max(items_detected - 1, 1):
        return True, "items_detected_but_not_labeled"
    return False, ""


def _entity_value_is_reliable(entity: str, phrase_tokens: list[str], raw_value: str) -> bool:
    joined = "".join(phrase_tokens)
    if entity in {"BILL_NO", "ORDER_NO", "TABLE_NO"}:
        return len(joined) >= 2 and any(ch.isdigit() for ch in joined)
    if entity == "TIME":
        return ":" in raw_value or re.search(r"\b\d{1,2}[:.]\d{2}\b", raw_value) is not None
    if entity in {"SUBTOTAL", "TAX", "TOTAL"}:
        return any(ch.isdigit() for ch in joined)
    if entity == "CASHIER":
        return len(joined) >= 4
    if entity == "PAYMENT_METHOD":
        return len(joined) >= 3
    return True


def _item_name_token_ok(text: str) -> bool:
    raw = (text or "").strip()
    norm = _normalize_token(raw)
    if not norm:
        return False
    if _looks_like_amount(raw):
        return False
    if norm.isdigit():
        return False
    return any(ch.isalpha() for ch in raw)


def _looks_like_amount(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    return re.fullmatch(r"[A-Z]{0,2}\d+[.,]?\d*", raw.replace(" ", "").upper()) is not None


def _numeric_token_ok(text: str, entity: str) -> bool:
    raw = (text or "").strip()
    norm = _normalize_token(raw)
    if not norm:
        return False
    if entity == "ITEM_QTY":
        return norm.isdigit() and len(norm) <= 3
    if entity in {"ITEM_PRICE", "SUBTOTAL", "TAX", "TOTAL"}:
        return _looks_like_amount(raw) and any(ch.isdigit() for ch in raw)
    return any(ch.isdigit() for ch in norm)


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
