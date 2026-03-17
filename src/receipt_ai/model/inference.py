from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.receipt_ai.model.compatibility import CheckpointCompatibility, inspect_label_mapping
from src.receipt_ai.model.decoder import decode_word_predictions
from src.receipt_ai.model.preprocessing import load_layoutlmv3_processor, prepare_inference_encoding
from src.receipt_ai.schemas import ModelPrediction, OCRLine, OCRToken

try:
    from transformers import AutoModelForTokenClassification
except ImportError:  # pragma: no cover
    AutoModelForTokenClassification = None


class LayoutLMv3TokenClassifier:
    """Real LayoutLMv3 token-classification inference backend."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        max_words: int = 512,
        max_length: int = 512,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_words = max_words
        self.max_length = max_length

    def predict(
        self,
        image_path: Path,
        ocr_lines: list[OCRLine],
        ocr_tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> ModelPrediction:
        checkpoint = self._validate_checkpoint(self.model_name_or_path)
        processor, model, id2label, compatibility = self._load_model_bundle(checkpoint)

        inference = prepare_inference_encoding(
            processor=processor,
            image_path=image_path,
            ocr_tokens=ocr_tokens,
            image_width=image_width,
            image_height=image_height,
            max_words=self.max_words,
            max_length=self.max_length,
        )

        model_inputs = {key: value.to(model.device) for key, value in inference.model_inputs.items()}
        with torch.inference_mode():
            outputs = model(**model_inputs)
        logits = outputs.logits.detach().cpu()[0]
        probs = torch.softmax(logits, dim=-1)

        word_scores: dict[int, dict[int, float]] = {}
        word_counts: dict[int, int] = {}
        for token_idx, word_idx in enumerate(inference.word_ids):
            if word_idx is None or word_idx >= len(inference.words):
                continue
            word_scores.setdefault(word_idx, {})
            word_counts[word_idx] = word_counts.get(word_idx, 0) + 1
            token_probs = probs[token_idx]
            for label_id in range(token_probs.shape[0]):
                score = float(token_probs[label_id].item())
                word_scores[word_idx][label_id] = word_scores[word_idx].get(label_id, 0.0) + score

        predicted_labels: list[str] = []
        predicted_scores: list[float] = []
        effective_tokens = ocr_tokens[: len(inference.words)]
        for word_idx in range(len(inference.words)):
            label_scores = word_scores.get(word_idx)
            if not label_scores:
                predicted_labels.append("O")
                predicted_scores.append(0.0)
                continue
            best_label_id = max(label_scores.items(), key=lambda item: item[1])[0]
            count = max(word_counts.get(word_idx, 1), 1)
            predicted_labels.append(id2label.get(int(best_label_id), "O"))
            predicted_scores.append(float(label_scores[best_label_id] / count))

        warnings: list[str] = []
        if inference.truncated:
            warnings.append(f"OCR tokens were truncated to {self.max_words} words before inference.")
        if compatibility.is_legacy:
            warnings.append(compatibility.message)

        return decode_word_predictions(
            effective_tokens,
            ocr_lines,
            predicted_labels,
            predicted_scores,
            source_image=image_path.name,
            mode="layoutlm_only",
            warnings=warnings,
        )

    @staticmethod
    def _validate_checkpoint(model_name_or_path: str) -> str:
        raw = (model_name_or_path or "").strip()
        if not raw:
            raise ValueError("LayoutLMv3 checkpoint path is empty.")
        if raw == "microsoft/layoutlmv3-base":
            raise ValueError(
                "The configured checkpoint is microsoft/layoutlmv3-base. "
                "This is a pretrained backbone, not a receipt token-classification checkpoint."
            )
        maybe_local = Path(raw).expanduser()
        if ("/" in raw or raw.startswith(".")) and not maybe_local.exists():
            raise FileNotFoundError(
                f"LayoutLMv3 checkpoint not found: {maybe_local.resolve()}. "
                "Train a checkpoint first or point RECEIPT_MODEL_CHECKPOINT to a valid model directory."
            )
        return str(maybe_local.resolve()) if ("/" in raw or raw.startswith(".")) else raw

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model_bundle(model_name_or_path: str) -> tuple[Any, Any, dict[int, str], CheckpointCompatibility]:
        if AutoModelForTokenClassification is None:
            raise ImportError("Missing dependency: transformers is required for LayoutLMv3 inference.")

        processor = load_layoutlmv3_processor(model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
        compatibility = inspect_label_mapping(id2label)
        if not compatibility.normalized_labels or all(label == "O" for label in compatibility.normalized_labels):
            raise ValueError(
                "Checkpoint labels are missing semantic receipt BIO tags or are generic LABEL_N values. "
                "Use a fine-tuned receipt token-classification checkpoint."
            )
        if (not compatibility.is_compatible) and (not compatibility.is_legacy):
            raise ValueError(compatibility.message)
        return processor, model, id2label, compatibility
