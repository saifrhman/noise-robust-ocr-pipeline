from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.ocr.easyocr_engine import EasyOCREngine
from src.receipt_ai.parsing.rules_parser import RuleBasedReceiptParser
from src.receipt_ai.schemas import (
    ExtractionMetadata,
    InvoiceInfo,
    ModelPrediction,
    PaymentInfo,
    ReceiptExtractionResult,
    TotalsInfo,
    VendorInfo,
)


class LayoutLMInferenceBackend(Protocol):
    """Pluggable model inference contract for layoutlm_only and hybrid modes."""

    def predict(self, image_path: Path, words: list[str], boxes_1000: list[list[int]]) -> ModelPrediction:
        ...


class NullLayoutLMBackend:
    """Default placeholder backend until LayoutLMv3 inference wiring is added."""

    def predict(self, image_path: Path, words: list[str], boxes_1000: list[list[int]]) -> ModelPrediction:
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
    backend = model_backend or NullLayoutLMBackend()

    ocr = engine.extract(image_path)
    llm_inputs = engine.to_layoutlm_inputs(ocr.tokens, ocr.image_width, ocr.image_height)
    prediction = backend.predict(Path(image_path).expanduser().resolve(), llm_inputs["words"], llm_inputs["boxes"])

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
    model_result = run_layoutlm_only(
        image_path,
        config=cfg,
        ocr_engine=ocr_engine,
        model_backend=model_backend,
    )

    fused = _fuse_results(rules_result, model_result, threshold=cfg.thresholds.model_field_confidence)
    fused.metadata.mode = "hybrid"
    return fused


def _fuse_results(
    rules: ReceiptExtractionResult,
    model: ReceiptExtractionResult,
    *,
    threshold: float,
) -> ReceiptExtractionResult:
    """Fusion hook preferring model for semantic fields if confidence is high."""
    out = rules

    model_conf = float(model.metadata.confidence)
    use_model = model_conf >= threshold

    if use_model:
        if model.vendor.name:
            out.vendor.name = model.vendor.name
        if model.vendor.registration_number:
            out.vendor.registration_number = model.vendor.registration_number
        if model.vendor.address:
            out.vendor.address = model.vendor.address

        if model.invoice.invoice_type:
            out.invoice.invoice_type = model.invoice.invoice_type
        if model.invoice.bill_number:
            out.invoice.bill_number = model.invoice.bill_number
        if model.invoice.order_number:
            out.invoice.order_number = model.invoice.order_number
        if model.invoice.table_number:
            out.invoice.table_number = model.invoice.table_number
        if model.invoice.date:
            out.invoice.date = model.invoice.date
        if model.invoice.time:
            out.invoice.time = model.invoice.time
        if model.invoice.cashier:
            out.invoice.cashier = model.invoice.cashier

    # Rules remain primary for totals and payment cleanup consistency.
    out.totals.total = out.totals.total if out.totals.total > 0 else model.totals.total
    out.totals.subtotal = out.totals.subtotal if out.totals.subtotal > 0 else model.totals.subtotal
    out.totals.tax = out.totals.tax if out.totals.tax > 0 else model.totals.tax
    out.totals.currency = out.totals.currency or model.totals.currency

    out.payment.method = out.payment.method or model.payment.method
    out.payment.amount_paid = out.payment.amount_paid if out.payment.amount_paid > 0 else model.payment.amount_paid

    out.metadata.confidence = max(rules.metadata.confidence, model.metadata.confidence)
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
