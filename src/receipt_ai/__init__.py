"""Core package for production-oriented receipt extraction components."""

from .config import ReceiptAIConfig
from .model.inference import LayoutLMv3TokenClassifier
from .model.labels import ID2LABEL, LABEL2ID, SEMANTIC_BIO_LABELS
from .ocr.easyocr_engine import EasyOCREngine
from .parsing.rules_parser import RuleBasedReceiptParser
from .pipelines.entrypoints import run_easyocr_rules, run_layoutlm_only, run_hybrid
from .schemas import (
    BoundingBox,
    OCRToken,
    OCRLine,
    VendorInfo,
    InvoiceInfo,
    ReceiptItem,
    TotalsInfo,
    PaymentInfo,
    ExtractionMetadata,
    ReceiptExtractionResult,
    ModelEntityPrediction,
    ModelPrediction,
    ReceiptSample,
)

__all__ = [
    "ReceiptAIConfig",
    "LayoutLMv3TokenClassifier",
    "LABEL2ID",
    "ID2LABEL",
    "SEMANTIC_BIO_LABELS",
    "EasyOCREngine",
    "RuleBasedReceiptParser",
    "run_easyocr_rules",
    "run_layoutlm_only",
    "run_hybrid",
    "BoundingBox",
    "OCRToken",
    "OCRLine",
    "VendorInfo",
    "InvoiceInfo",
    "ReceiptItem",
    "TotalsInfo",
    "PaymentInfo",
    "ExtractionMetadata",
    "ReceiptExtractionResult",
    "ModelEntityPrediction",
    "ModelPrediction",
    "ReceiptSample",
]
