from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from src.receipt_ai.model.labels import SEMANTIC_BIO_LABELS


@dataclass(slots=True)
class PathsConfig:
    """Filesystem paths used across training, inference, and exports."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "SROIE2019")
    output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs")
    model_checkpoint: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "layoutlmv3_sroie")

    def ensure_output_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class OCRConfig:
    """EasyOCR runtime settings."""

    languages: tuple[str, ...] = ("en",)
    gpu: bool = False
    min_confidence: float = 0.15
    width_ths: float = 0.7
    paragraph: bool = False
    y_sort_tolerance_px: int = 8


@dataclass(slots=True)
class LabelConfig:
    """LayoutLMv3 BIO label schema and maps for receipt token classification."""

    labels: list[str] = field(default_factory=lambda: list(SEMANTIC_BIO_LABELS))

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {i: label for i, label in enumerate(self.labels)}


@dataclass(slots=True)
class ThresholdConfig:
    """Thresholds for parser confidence and fusion behavior."""

    model_field_confidence: float = 0.65
    model_entity_confidence: float = 0.50
    amount_tolerance: float = 0.02
    model_semantic_strong_confidence: float = 0.75
    model_takeover_margin: float = 0.12
    low_confidence_guard: float = 0.50
    total_item_consistency_tolerance: float = 0.08


@dataclass(slots=True)
class ReceiptAIConfig:
    """Top-level config container for local development and production wiring."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    @classmethod
    def from_env(cls) -> "ReceiptAIConfig":
        """
        Create config with optional environment overrides.

        Supported env vars:
        - RECEIPT_DATA_ROOT
        - RECEIPT_OUTPUT_ROOT
        - RECEIPT_MODEL_CHECKPOINT
        - RECEIPT_OCR_GPU (0/1/true/false)
        - RECEIPT_OCR_MIN_CONFIDENCE
        """
        cfg = cls()

        data_root = os.getenv("RECEIPT_DATA_ROOT")
        if data_root:
            cfg.paths.data_root = Path(data_root).expanduser().resolve()

        output_root = os.getenv("RECEIPT_OUTPUT_ROOT")
        if output_root:
            cfg.paths.output_root = Path(output_root).expanduser().resolve()

        model_checkpoint = os.getenv("RECEIPT_MODEL_CHECKPOINT")
        if model_checkpoint:
            cfg.paths.model_checkpoint = Path(model_checkpoint).expanduser().resolve()

        ocr_gpu = os.getenv("RECEIPT_OCR_GPU")
        if ocr_gpu is not None:
            cfg.ocr.gpu = ocr_gpu.strip().lower() in {"1", "true", "yes", "on"}

        min_conf = os.getenv("RECEIPT_OCR_MIN_CONFIDENCE")
        if min_conf:
            try:
                cfg.ocr.min_confidence = float(min_conf)
            except ValueError:
                pass

        model_field_conf = os.getenv("RECEIPT_MODEL_FIELD_CONFIDENCE")
        if model_field_conf:
            try:
                cfg.thresholds.model_field_confidence = float(model_field_conf)
            except ValueError:
                pass

        model_semantic_conf = os.getenv("RECEIPT_MODEL_SEMANTIC_CONFIDENCE")
        if model_semantic_conf:
            try:
                cfg.thresholds.model_semantic_strong_confidence = float(model_semantic_conf)
            except ValueError:
                pass

        cfg.paths.ensure_output_dirs()
        return cfg
