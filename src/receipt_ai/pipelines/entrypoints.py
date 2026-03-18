from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.model.inference import LayoutLMv3TokenClassifier
from src.receipt_ai.ocr.easyocr_engine import EasyOCREngine
from src.receipt_ai.parsing.rules_parser import RuleBasedReceiptParser
from src.receipt_ai.schemas import (
    ExtractionMetadata,
    InvoiceInfo,
    ModelPrediction,
    OCRLine,
    OCRToken,
    PaymentInfo,
    ReceiptItem,
    ReceiptExtractionResult,
    TotalsInfo,
    VendorInfo,
)


class LayoutLMInferenceBackend(Protocol):
    """Pluggable model inference contract for layoutlm_only and hybrid modes."""

    def predict(
        self,
        image_path: Path,
        ocr_lines: list[OCRLine],
        ocr_tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> ModelPrediction:
        ...


class NullLayoutLMBackend:
    """Default placeholder backend until LayoutLMv3 inference wiring is added."""

    def predict(
        self,
        image_path: Path,
        ocr_lines: list[OCRLine],
        ocr_tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> ModelPrediction:
        prediction = ModelPrediction()
        prediction.warnings.append(
            "LayoutLM backend is not configured yet. Returning empty model prediction placeholder."
        )
        return prediction


def run_easyocr_rules(
    image_path: str | Path,
    *,
    config: ReceiptAIConfig | None = None,
    ocr_engine: EasyOCREngine | None = None,
    parser: RuleBasedReceiptParser | None = None,
) -> ReceiptExtractionResult:
    """Fully runnable end-to-end OCR + rules pipeline."""
    cfg = config or ReceiptAIConfig.from_env()
    engine = ocr_engine or EasyOCREngine(cfg.ocr)
    rule_parser = parser or RuleBasedReceiptParser()

    ocr = engine.extract(image_path)
    result = rule_parser.parse(
        ocr.lines,
        source_image=ocr.image_path,
        confidence=ocr.mean_confidence,
        mode="easyocr_rules",
    )
    return result


def run_layoutlm_only(
    image_path: str | Path,
    *,
    config: ReceiptAIConfig | None = None,
    ocr_engine: EasyOCREngine | None = None,
    model_backend: LayoutLMInferenceBackend | None = None,
) -> ReceiptExtractionResult:
    """
    OCR + model-only entrypoint.

    This function is fully wired to OCR and model interfaces.
    Model behavior is currently placeholder by default via `NullLayoutLMBackend`.
    """
    cfg = config or ReceiptAIConfig.from_env()
    engine = ocr_engine or EasyOCREngine(cfg.ocr)
    backend = model_backend or LayoutLMv3TokenClassifier(cfg.paths.model_checkpoint)

    ocr = engine.extract(image_path)
    prediction = backend.predict(
        Path(image_path).expanduser().resolve(),
        ocr.lines,
        ocr.tokens,
        ocr.image_width,
        ocr.image_height,
    )

    if prediction.result is not None:
        result = prediction.result
        result.metadata.mode = "layoutlm_only"
        if result.metadata.source_image == "":
            result.metadata.source_image = ocr.image_path.name
        if prediction.warnings:
            result.metadata.warnings = _merge_lists(result.metadata.warnings, prediction.warnings)
        if prediction.field_confidences:
            result.metadata.field_confidences.update(prediction.field_confidences)
            for field_name in prediction.field_confidences:
                result.metadata.field_provenance.setdefault(field_name, "layoutlm_only")
        return result

    fields = prediction.fields
    conf = sum(prediction.token_scores) / max(len(prediction.token_scores), 1) if prediction.token_scores else 0.0

    result = ReceiptExtractionResult(
        vendor=VendorInfo(
            name=fields.get("vendor_name", "") or fields.get("merchant", ""),
            registration_number=fields.get("reg_no", ""),
            address=fields.get("address", ""),
        ),
        invoice=InvoiceInfo(
            invoice_type=fields.get("invoice_type", ""),
            bill_number=fields.get("bill_number", "") or fields.get("bill_no", ""),
            order_number=fields.get("order_number", "") or fields.get("order_no", ""),
            table_number=fields.get("table_number", "") or fields.get("table_no", ""),
            date=fields.get("date", ""),
            time=fields.get("time", ""),
            cashier=fields.get("cashier", ""),
        ),
        totals=TotalsInfo(
            subtotal=_to_float(fields.get("subtotal", "")),
            tax=_to_float(fields.get("tax", "")),
            total=_to_float(fields.get("total", "")),
            currency=fields.get("currency", ""),
        ),
        payment=PaymentInfo(
            method=fields.get("payment_method", ""),
            amount_paid=_to_float(fields.get("amount_paid", "")),
        ),
        metadata=ExtractionMetadata(
            mode="layoutlm_only",
            confidence=float(conf),
            source_image=ocr.image_path.name,
            warnings=list(prediction.warnings),
            field_confidences=dict(prediction.field_confidences),
            field_provenance={field_name: "layoutlm_only" for field_name in prediction.field_confidences},
        ),
    )

    return result


def run_hybrid(
    image_path: str | Path,
    *,
    config: ReceiptAIConfig | None = None,
    ocr_engine: EasyOCREngine | None = None,
    parser: RuleBasedReceiptParser | None = None,
    model_backend: LayoutLMInferenceBackend | None = None,
) -> ReceiptExtractionResult:
    """
    Hybrid entrypoint with fusion hooks ready for model + rule outputs.

    Current behavior:
    - runs full easyocr_rules extraction
    - runs layoutlm_only interface (placeholder model by default)
    - fuses fields with confidence-aware preference hooks
    """
    cfg = config or ReceiptAIConfig.from_env()
    rules_result = run_easyocr_rules(
        image_path,
        config=cfg,
        ocr_engine=ocr_engine,
        parser=parser,
    )
    model_result = run_layoutlm_only(image_path, config=cfg, ocr_engine=ocr_engine, model_backend=model_backend)

    fused = _fuse_results(
        rules_result,
        model_result,
        threshold=cfg.thresholds.model_field_confidence,
        semantic_threshold=cfg.thresholds.model_semantic_strong_confidence,
        takeover_margin=cfg.thresholds.model_takeover_margin,
        low_confidence_guard=cfg.thresholds.low_confidence_guard,
        amount_tolerance=cfg.thresholds.total_item_consistency_tolerance,
    )
    fused.metadata.mode = "hybrid"
    return fused


def _fuse_results(
    rules: ReceiptExtractionResult,
    model: ReceiptExtractionResult,
    *,
    threshold: float,
    semantic_threshold: float,
    takeover_margin: float,
    low_confidence_guard: float,
    amount_tolerance: float,
) -> ReceiptExtractionResult:
    """Field-aware fusion keeping rules for numeric cleanup and model for strong semantic fields."""
    out = deepcopy(rules)
    out.metadata.warnings = _merge_lists(rules.metadata.warnings, model.metadata.warnings)
    out.metadata.field_confidences = dict(rules.metadata.field_confidences)
    out.metadata.field_provenance = dict(rules.metadata.field_provenance)

    for field_name in _non_empty_rule_fields(rules):
        out.metadata.field_confidences.setdefault(field_name, float(rules.metadata.confidence))
        out.metadata.field_provenance.setdefault(field_name, "easyocr_rules")

    _merge_text_field(
        out,
        "vendor.name",
        rules.vendor.name,
        model.vendor.name,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "vendor.registration_number",
        rules.vendor.registration_number,
        model.vendor.registration_number,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "vendor.address",
        rules.vendor.address,
        model.vendor.address,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.invoice_type",
        rules.invoice.invoice_type,
        model.invoice.invoice_type,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.bill_number",
        rules.invoice.bill_number,
        model.invoice.bill_number,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.order_number",
        rules.invoice.order_number,
        model.invoice.order_number,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.table_number",
        rules.invoice.table_number,
        model.invoice.table_number,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.date",
        rules.invoice.date,
        model.invoice.date,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.time",
        rules.invoice.time,
        model.invoice.time,
        rules,
        model,
        threshold,
        prefer_model=True,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "invoice.cashier",
        rules.invoice.cashier,
        model.invoice.cashier,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )

    _merge_numeric_field(out, "totals.subtotal", rules.totals.subtotal, model.totals.subtotal, rules, model, threshold)
    _merge_numeric_field(out, "totals.tax", rules.totals.tax, model.totals.tax, rules, model, threshold)
    _merge_numeric_field(out, "totals.total", rules.totals.total, model.totals.total, rules, model, threshold)
    _merge_text_field(
        out,
        "totals.currency",
        rules.totals.currency,
        model.totals.currency,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_text_field(
        out,
        "payment.method",
        rules.payment.method,
        model.payment.method,
        rules,
        model,
        threshold,
        prefer_model=False,
        semantic_threshold=semantic_threshold,
        takeover_margin=takeover_margin,
        low_confidence_guard=low_confidence_guard,
    )
    _merge_numeric_field(out, "payment.amount_paid", rules.payment.amount_paid, model.payment.amount_paid, rules, model, threshold)

    if _item_quality_score(model.items) > _item_quality_score(rules.items):
        out.items = deepcopy(model.items)
        item_conf = max(model.metadata.field_confidences.get("items", model.metadata.confidence), threshold)
        out.metadata.field_confidences["items"] = float(item_conf)
        out.metadata.field_provenance["items"] = "layoutlm_only"
    elif rules.items:
        out.metadata.field_confidences.setdefault("items", float(rules.metadata.confidence))
        out.metadata.field_provenance.setdefault("items", "easyocr_rules")

    out.metadata.confidence = max(rules.metadata.confidence, model.metadata.confidence)

    # Policy check: keep totals coherent with item implied sum when available.
    implied_total = _items_implied_total(out.items)
    if out.totals.total > 0 and implied_total > 0 and not _within_rel_tolerance(out.totals.total, implied_total, amount_tolerance):
        rules_total = rules.totals.total
        model_total = model.totals.total
        if rules_total > 0 and _within_rel_tolerance(rules_total, implied_total, amount_tolerance):
            out.totals.total = rules_total
            out.metadata.field_provenance["totals.total"] = "easyocr_rules"
            out.metadata.warnings.append("Hybrid adjusted totals.total to rules for item-total consistency.")
        elif model_total > 0 and _within_rel_tolerance(model_total, implied_total, amount_tolerance):
            out.totals.total = model_total
            out.metadata.field_provenance["totals.total"] = "layoutlm_only"
            out.metadata.warnings.append("Hybrid adjusted totals.total to model for item-total consistency.")
        else:
            out.metadata.warnings.append("Hybrid detected conflict between totals.total and item implied sum.")

    # Explicitly surface missing critical fields from one source for diagnostics.
    _warn_critical_missing(out, rules, model)
    return out


def _merge_text_field(
    out: ReceiptExtractionResult,
    field_name: str,
    rules_value: str,
    model_value: str,
    rules: ReceiptExtractionResult,
    model: ReceiptExtractionResult,
    threshold: float,
    *,
    prefer_model: bool,
    semantic_threshold: float,
    takeover_margin: float,
    low_confidence_guard: float,
) -> None:
    rule_text = (rules_value or "").strip()
    model_text = (model_value or "").strip()
    chosen_value = rule_text
    chosen_source = "easyocr_rules"
    chosen_conf = _field_confidence(rules, field_name)

    model_conf = _field_confidence(model, field_name)
    if model_text and (not rule_text):
        chosen_value = model_text
        chosen_source = "layoutlm_only"
        chosen_conf = model_conf
    elif model_text and prefer_model and model_conf >= semantic_threshold:
        chosen_value = model_text
        chosen_source = "layoutlm_only"
        chosen_conf = model_conf
    elif (
        model_text
        and model_conf > chosen_conf + takeover_margin
        and model_conf >= threshold
        and len(model_text) >= len(rule_text)
    ):
        chosen_value = model_text
        chosen_source = "layoutlm_only"
        chosen_conf = model_conf

    # Guard against low-confidence model overriding strong rule text.
    if rule_text and model_text and chosen_source == "layoutlm_only" and model_conf < low_confidence_guard:
        chosen_value = rule_text
        chosen_source = "easyocr_rules"
        chosen_conf = _field_confidence(rules, field_name)

    _assign_field(out, field_name, chosen_value)
    if chosen_value:
        out.metadata.field_confidences[field_name] = float(chosen_conf)
        out.metadata.field_provenance[field_name] = chosen_source


def _merge_numeric_field(
    out: ReceiptExtractionResult,
    field_name: str,
    rules_value: float,
    model_value: float,
    rules: ReceiptExtractionResult,
    model: ReceiptExtractionResult,
    threshold: float,
) -> None:
    chosen_value = rules_value
    chosen_source = "easyocr_rules"
    chosen_conf = _field_confidence(rules, field_name)

    model_conf = _field_confidence(model, field_name)
    if model_value > 0 and rules_value <= 0:
        chosen_value = model_value
        chosen_source = "layoutlm_only"
        chosen_conf = model_conf

    _assign_field(out, field_name, chosen_value)
    if chosen_value > 0:
        out.metadata.field_confidences[field_name] = float(chosen_conf)
        out.metadata.field_provenance[field_name] = chosen_source


def _assign_field(out: ReceiptExtractionResult, field_name: str, value: str | float) -> None:
    target, attr = field_name.split(".")
    getattr(out, target).__setattr__(attr, value)


def _field_confidence(result: ReceiptExtractionResult, field_name: str) -> float:
    return float(result.metadata.field_confidences.get(field_name, result.metadata.confidence))


def _non_empty_rule_fields(result: ReceiptExtractionResult) -> list[str]:
    fields: list[str] = []
    if result.vendor.name:
        fields.append("vendor.name")
    if result.vendor.registration_number:
        fields.append("vendor.registration_number")
    if result.vendor.address:
        fields.append("vendor.address")
    if result.invoice.invoice_type:
        fields.append("invoice.invoice_type")
    if result.invoice.bill_number:
        fields.append("invoice.bill_number")
    if result.invoice.order_number:
        fields.append("invoice.order_number")
    if result.invoice.table_number:
        fields.append("invoice.table_number")
    if result.invoice.date:
        fields.append("invoice.date")
    if result.invoice.time:
        fields.append("invoice.time")
    if result.invoice.cashier:
        fields.append("invoice.cashier")
    if result.totals.subtotal > 0:
        fields.append("totals.subtotal")
    if result.totals.tax > 0:
        fields.append("totals.tax")
    if result.totals.total > 0:
        fields.append("totals.total")
    if result.totals.currency:
        fields.append("totals.currency")
    if result.payment.method:
        fields.append("payment.method")
    if result.payment.amount_paid > 0:
        fields.append("payment.amount_paid")
    if result.items:
        fields.append("items")
    return fields


def _item_quality_score(items: list[ReceiptItem]) -> float:
    score = 0.0
    for item in items:
        if item.name:
            score += 1.0
        if item.line_total > 0:
            score += 1.0
        if item.quantity > 0:
            score += 0.5
    return score


def _merge_lists(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in [*(left or []), *(right or [])]:
        text = (value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _to_float(value: str) -> float:
    text = (value or "").strip().replace(",", ".")
    text = "".join(ch for ch in text if (ch.isdigit() or ch == "."))
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _items_implied_total(items: list[ReceiptItem]) -> float:
    if not items:
        return 0.0
    return round(sum(max(item.line_total, 0.0) for item in items), 2)


def _within_rel_tolerance(left: float, right: float, tolerance: float) -> bool:
    if left <= 0 and right <= 0:
        return True
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(left), abs(right)) <= tolerance


def _warn_critical_missing(
    out: ReceiptExtractionResult,
    rules: ReceiptExtractionResult,
    model: ReceiptExtractionResult,
) -> None:
    checks = [
        ("vendor.name", rules.vendor.name, model.vendor.name),
        ("invoice.date", rules.invoice.date, model.invoice.date),
        ("totals.total", str(rules.totals.total if rules.totals.total > 0 else ""), str(model.totals.total if model.totals.total > 0 else "")),
    ]
    for field_name, rule_val, model_val in checks:
        r = (rule_val or "").strip()
        m = (model_val or "").strip()
        if r and not m:
            out.metadata.warnings.append(f"Hybrid kept rules {field_name}: model missing value.")
        elif m and not r:
            out.metadata.warnings.append(f"Hybrid used model {field_name}: rules missing value.")
