from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from src.receipt_ai.parsing.normalization import extract_date_time, parse_amount
from src.receipt_ai.schemas import (
    ExtractionMetadata,
    InvoiceInfo,
    ModelEntityPrediction,
    ModelPrediction,
    OCRLine,
    OCRToken,
    PaymentInfo,
    ReceiptExtractionResult,
    ReceiptItem,
    TotalsInfo,
    VendorInfo,
)
from .labels import FIELD_TO_RESULT_KEY, normalize_model_label, split_bio


def decode_word_predictions(
    ocr_tokens: list[OCRToken],
    ocr_lines: list[OCRLine],
    predicted_labels: list[str],
    predicted_scores: list[float],
    *,
    source_image: str,
    mode: str,
    warnings: list[str] | None = None,
) -> ModelPrediction:
    normalized_labels = [normalize_model_label(label) for label in predicted_labels]
    entities = _collect_entities(ocr_tokens, normalized_labels, predicted_scores)
    result, field_confidences = _entities_to_result(
        entities,
        ocr_tokens,
        ocr_lines,
        source_image=source_image,
        mode=mode,
        warnings=warnings,
    )

    return ModelPrediction(
        fields=_result_to_field_dict(result),
        raw_entities=entities,
        token_labels=normalized_labels,
        token_scores=[float(score) for score in predicted_scores],
        warnings=list(warnings or []),
        field_confidences=field_confidences,
        result=result,
    )


def _collect_entities(
    ocr_tokens: list[OCRToken],
    labels: list[str],
    scores: list[float],
) -> list[ModelEntityPrediction]:
    entities: list[ModelEntityPrediction] = []
    current_label = ""
    current_tokens: list[str] = []
    current_scores: list[float] = []
    current_boxes: list[list[int]] = []

    def flush() -> None:
        nonlocal current_label, current_tokens, current_scores, current_boxes
        if not current_label or not current_tokens:
            current_label = ""
            current_tokens = []
            current_scores = []
            current_boxes = []
            return
        entities.append(
            ModelEntityPrediction(
                label=current_label,
                text=" ".join(current_tokens).strip(),
                score=float(sum(current_scores) / max(len(current_scores), 1)),
                bbox=_merge_boxes(current_boxes),
            )
        )
        current_label = ""
        current_tokens = []
        current_scores = []
        current_boxes = []

    for idx, token in enumerate(ocr_tokens[: len(labels)]):
        prefix, entity = split_bio(labels[idx])
        if prefix == "O":
            flush()
            continue

        score = float(scores[idx]) if idx < len(scores) else 0.0
        token_box = token.bbox.to_xyxy()
        if prefix == "B" or current_label != entity:
            flush()
            current_label = entity
            current_tokens = [token.text]
            current_scores = [score]
            current_boxes = [token_box]
        else:
            current_tokens.append(token.text)
            current_scores.append(score)
            current_boxes.append(token_box)

    flush()

    dedup: dict[tuple[str, str], ModelEntityPrediction] = {}
    for entity in entities:
        key = (entity.label, re.sub(r"\s+", " ", entity.text.upper()).strip())
        best = dedup.get(key)
        if best is None or entity.score > best.score:
            dedup[key] = entity
    return list(dedup.values())


def _merge_boxes(boxes: list[list[int]]) -> list[int] | None:
    valid = [box for box in boxes if len(box) == 4]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _entities_to_result(
    entities: list[ModelEntityPrediction],
    ocr_tokens: list[OCRToken],
    ocr_lines: list[OCRLine],
    *,
    source_image: str,
    mode: str,
    warnings: list[str] | None = None,
) -> tuple[ReceiptExtractionResult, dict[str, float]]:
    field_confidences: dict[str, float] = {}
    scalar_best: dict[str, ModelEntityPrediction] = {}
    multi_values: dict[str, list[ModelEntityPrediction]] = defaultdict(list)

    for entity in entities:
        multi_values[entity.label].append(entity)
        best = scalar_best.get(entity.label)
        if best is None or entity.score > best.score:
            scalar_best[entity.label] = entity

    vendor = VendorInfo(
        name=_best_text(scalar_best, "VENDOR_NAME", field_confidences),
        registration_number=_best_text(scalar_best, "REG_NO", field_confidences),
        address=_best_text(scalar_best, "ADDRESS", field_confidences),
    )

    invoice_date = _best_text(scalar_best, "DATE", field_confidences)
    invoice_time = _best_text(scalar_best, "TIME", field_confidences)
    if invoice_date and not invoice_time:
        normalized_date, maybe_time = extract_date_time(invoice_date)
        invoice_date = normalized_date or invoice_date
        invoice_time = maybe_time or invoice_time

    invoice = InvoiceInfo(
        invoice_type=_best_text(scalar_best, "INVOICE_TYPE", field_confidences),
        bill_number=_best_text(scalar_best, "BILL_NO", field_confidences),
        order_number=_best_text(scalar_best, "ORDER_NO", field_confidences),
        table_number=_best_text(scalar_best, "TABLE_NO", field_confidences),
        date=invoice_date,
        time=invoice_time,
        cashier=_best_text(scalar_best, "CASHIER", field_confidences),
    )

    totals = TotalsInfo(
        subtotal=parse_amount(_best_text(scalar_best, "SUBTOTAL", field_confidences)),
        tax=parse_amount(_best_text(scalar_best, "TAX", field_confidences)),
        total=parse_amount(_best_text(scalar_best, "TOTAL", field_confidences)),
    )

    payment = PaymentInfo(
        method=_best_text(scalar_best, "PAYMENT_METHOD", field_confidences),
        amount_paid=0.0,
    )

    items = _decode_items_from_lines(ocr_lines, ocr_tokens, entities)

    result = ReceiptExtractionResult(
        vendor=vendor,
        invoice=invoice,
        items=items,
        totals=totals,
        payment=payment,
        metadata=ExtractionMetadata(
            mode=mode,
            confidence=_overall_confidence(field_confidences),
            source_image=source_image,
            warnings=list(warnings or []),
            field_confidences=dict(field_confidences),
            field_provenance={field_name: mode for field_name in field_confidences},
        ),
    )

    return result, field_confidences


def _best_text(scalar_best: dict[str, ModelEntityPrediction], label: str, field_confidences: dict[str, float]) -> str:
    entity = scalar_best.get(label)
    if entity is None:
        return ""
    field_key = FIELD_TO_RESULT_KEY.get(label, label.lower())
    field_confidences[field_key] = float(entity.score)
    return entity.text.strip()


def _decode_items_from_lines(
    ocr_lines: list[OCRLine],
    ocr_tokens: list[OCRToken],
    entities: list[ModelEntityPrediction],
) -> list[ReceiptItem]:
    token_label_map: dict[int, list[tuple[str, float, str]]] = defaultdict(list)
    entity_token_texts: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        entity_token_texts[entity.label].extend(entity.text.split())

    for idx, token in enumerate(ocr_tokens):
        norm_token = re.sub(r"\s+", " ", token.text.upper()).strip()
        for entity in entities:
            if norm_token in {part.upper() for part in entity.text.split()}:
                token_label_map[idx].append((entity.label, entity.score, entity.text))

    line_groups: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    line_scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for idx, token in enumerate(ocr_tokens):
        if token.line_id is None:
            continue
        matches = token_label_map.get(idx, [])
        for label, score, _ in matches:
            line_groups[token.line_id][label].append(token.text)
            line_scores[token.line_id][label].append(score)

    items: list[ReceiptItem] = []
    pending_name_parts: list[str] = []
    pending_qty = 1.0

    for line in ocr_lines:
        groups = line_groups.get(line.line_id, {})
        name = " ".join(groups.get("ITEM_NAME", [])).strip()
        qty_text = " ".join(groups.get("ITEM_QTY", [])).strip()
        price_text = " ".join(groups.get("ITEM_PRICE", [])).strip()

        if name and not price_text:
            pending_name_parts.append(name)
            if qty_text:
                pending_qty = max(parse_amount(qty_text), 1.0)
            continue

        if name:
            pending_name_parts.append(name)

        if pending_name_parts and price_text:
            line_total = parse_amount(price_text)
            qty = max(parse_amount(qty_text), pending_qty, 1.0)
            unit_price = round(line_total / qty, 4) if line_total > 0 and qty > 0 else 0.0
            items.append(
                ReceiptItem(
                    name=" ".join(part for part in pending_name_parts if part).strip(),
                    quantity=qty,
                    unit_price=unit_price,
                    line_total=line_total,
                )
            )
            pending_name_parts = []
            pending_qty = 1.0

    if pending_name_parts:
        items.append(ReceiptItem(name=" ".join(pending_name_parts).strip(), quantity=max(pending_qty, 1.0)))

    deduped: list[ReceiptItem] = []
    seen: set[tuple[str, float]] = set()
    for item in items:
        key = (re.sub(r"\s+", " ", item.name.upper()).strip(), round(item.line_total, 2))
        if key in seen or not item.name:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _overall_confidence(field_confidences: dict[str, float]) -> float:
    if not field_confidences:
        return 0.0
    return float(sum(field_confidences.values()) / max(len(field_confidences), 1))


def _result_to_field_dict(result: ReceiptExtractionResult) -> dict[str, str]:
    return {
        "vendor_name": result.vendor.name,
        "address": result.vendor.address,
        "reg_no": result.vendor.registration_number,
        "invoice_type": result.invoice.invoice_type,
        "bill_number": result.invoice.bill_number,
        "order_number": result.invoice.order_number,
        "table_number": result.invoice.table_number,
        "date": result.invoice.date,
        "time": result.invoice.time,
        "cashier": result.invoice.cashier,
        "subtotal": f"{result.totals.subtotal:.2f}" if result.totals.subtotal > 0 else "",
        "tax": f"{result.totals.tax:.2f}" if result.totals.tax > 0 else "",
        "total": f"{result.totals.total:.2f}" if result.totals.total > 0 else "",
        "payment_method": result.payment.method,
        "amount_paid": f"{result.payment.amount_paid:.2f}" if result.payment.amount_paid > 0 else "",
    }
