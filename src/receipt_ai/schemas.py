from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel space."""

    x1: int
    y1: int
    x2: int
    y2: int

    def to_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_layoutlm_1000(self, width: int, height: int) -> list[int]:
        """Convert pixel coords to LayoutLM normalized [0..1000] format."""
        width = max(width, 1)
        height = max(height, 1)
        return [
            max(0, min(1000, int(1000 * self.x1 / width))),
            max(0, min(1000, int(1000 * self.y1 / height))),
            max(0, min(1000, int(1000 * self.x2 / width))),
            max(0, min(1000, int(1000 * self.y2 / height))),
        ]


@dataclass(slots=True)
class OCRToken:
    """Single OCR token with geometry and optional confidence."""

    text: str
    bbox: BoundingBox
    confidence: float | None = None
    line_id: int | None = None


@dataclass(slots=True)
class OCRLine:
    """OCR line containing grouped tokens and line-level geometry."""

    line_id: int
    text: str
    bbox: BoundingBox
    tokens: list[OCRToken] = field(default_factory=list)
    confidence: float | None = None


@dataclass(slots=True)
class VendorInfo:
    name: str = ""
    registration_number: str = ""
    address: str = ""


@dataclass(slots=True)
class InvoiceInfo:
    invoice_type: str = ""
    bill_number: str = ""
    order_number: str = ""
    table_number: str = ""
    date: str = ""
    time: str = ""
    cashier: str = ""


@dataclass(slots=True)
class ReceiptItem:
    name: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    line_total: float = 0.0


@dataclass(slots=True)
class TotalsInfo:
    subtotal: float = 0.0
    service_charge: float = 0.0
    tax: float = 0.0
    rounding: float = 0.0
    total: float = 0.0
    currency: str = ""


@dataclass(slots=True)
class PaymentInfo:
    method: str = ""
    amount_paid: float = 0.0


@dataclass(slots=True)
class ExtractionMetadata:
    mode: str = ""
    confidence: float = 0.0
    source_image: str = ""
    warnings: list[str] = field(default_factory=list)
    field_confidences: dict[str, float] = field(default_factory=dict)
    field_provenance: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReceiptExtractionResult:
    """Final normalized output schema for downstream JSON usage."""

    vendor: VendorInfo = field(default_factory=VendorInfo)
    invoice: InvoiceInfo = field(default_factory=InvoiceInfo)
    items: list[ReceiptItem] = field(default_factory=list)
    totals: TotalsInfo = field(default_factory=TotalsInfo)
    payment: PaymentInfo = field(default_factory=PaymentInfo)
    metadata: ExtractionMetadata = field(default_factory=ExtractionMetadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelEntityPrediction:
    """Decoded model entity candidate from token classification."""

    label: str
    text: str
    score: float
    bbox: list[int] | None = None


@dataclass(slots=True)
class ModelPrediction:
    """LayoutLMv3 prediction bundle usable by fusion/evaluation modules."""

    fields: dict[str, str] = field(default_factory=dict)
    raw_entities: list[ModelEntityPrediction] = field(default_factory=list)
    token_labels: list[str] = field(default_factory=list)
    token_scores: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_confidences: dict[str, float] = field(default_factory=dict)
    result: ReceiptExtractionResult | None = None


@dataclass(slots=True)
class ReceiptSample:
    """
    Reusable dataset sample for rule-based parsing and model training.

    Notes:
    - `ground_truth` is optional and may be partial depending on split/annotation availability.
    - For SROIE, ground truth generally includes company/date/address/total.
    """

    sample_id: str
    split: str
    image_path: Path
    image_width: int
    image_height: int
    ocr_lines: list[OCRLine] = field(default_factory=list)
    ocr_tokens: list[OCRToken] = field(default_factory=list)
    raw_ocr_text: str = ""
    ground_truth: ReceiptExtractionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        return payload
