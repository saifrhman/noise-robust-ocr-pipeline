from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.receipt_ai.schemas import OCRToken

try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover
    AutoProcessor = None


@dataclass(slots=True)
class InferenceEncoding:
    encoding: Any
    model_inputs: dict[str, torch.Tensor]
    word_ids: list[int | None]
    words: list[str]
    boxes: list[list[int]]
    truncated: bool


def load_layoutlmv3_processor(model_name_or_path: str) -> Any:
    if AutoProcessor is None:
        raise ImportError("Missing dependency: transformers is required for LayoutLMv3.")
    try:
        return AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False)
    except TypeError:
        return AutoProcessor.from_pretrained(model_name_or_path)


def align_word_labels_to_tokens(word_ids: list[int | None], word_labels: list[int], label_pad_token_id: int = -100) -> list[int]:
    """Explicitly align word-level labels to subword tokens for token classification training."""
    token_labels: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None:
            token_labels.append(label_pad_token_id)
            previous_word_id = word_id
            continue
        if word_id >= len(word_labels):
            token_labels.append(label_pad_token_id)
            previous_word_id = word_id
            continue
        if previous_word_id != word_id:
            token_labels.append(int(word_labels[word_id]))
        else:
            token_labels.append(label_pad_token_id)
        previous_word_id = word_id
    return token_labels


def align_word_values_to_tokens(
    word_ids: list[int | None],
    word_values: list[float],
    *,
    pad_value: float = 0.0,
) -> list[float]:
    """Align generic per-word float values to first-subword tokens only."""
    token_values: list[float] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id >= len(word_values):
            token_values.append(float(pad_value))
            previous_word_id = word_id
            continue
        if previous_word_id != word_id:
            token_values.append(float(word_values[word_id]))
        else:
            token_values.append(float(pad_value))
        previous_word_id = word_id
    return token_values


def _processor_encode(
    processor: Any,
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    *,
    truncation: bool = True,
    max_length: int = 512,
) -> Any:
    common_kwargs = {
        "images": image,
        "boxes": boxes,
        "truncation": truncation,
        "padding": "max_length",
        "max_length": max_length,
        "return_tensors": "pt",
    }
    try:
        return processor(text=words, **common_kwargs)
    except (TypeError, KeyError):
        return processor(words=words, **common_kwargs)


def prepare_training_feature(
    processor: Any,
    image_path: str | Path,
    words: list[str],
    boxes_1000: list[list[int]],
    word_labels: list[int],
    word_label_weights: list[float] | None = None,
    *,
    max_length: int = 512,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    encoding = _processor_encode(processor, image=image, words=words, boxes=boxes_1000, max_length=max_length)
    word_ids = encoding.word_ids(batch_index=0)
    labels = align_word_labels_to_tokens(word_ids, word_labels)
    label_weights = align_word_values_to_tokens(word_ids, word_label_weights or [1.0] * len(word_labels), pad_value=0.0)

    out: dict[str, Any] = {}
    for key, value in encoding.items():
        out[key] = value.squeeze(0)
    out["labels"] = torch.tensor(labels, dtype=torch.long)
    out["label_weights"] = torch.tensor(label_weights, dtype=torch.float)
    return out


def prepare_inference_encoding(
    processor: Any,
    image_path: str | Path,
    ocr_tokens: list[OCRToken],
    image_width: int,
    image_height: int,
    max_words: int = 512,
    max_length: int = 512,
) -> InferenceEncoding:
    words: list[str] = []
    boxes: list[list[int]] = []
    truncated = False
    for token in ocr_tokens:
        if not token.text:
            continue
        words.append(token.text)
        boxes.append(token.bbox.to_layoutlm_1000(image_width, image_height))
        if len(words) >= max_words:
            truncated = True
            break

    image = Image.open(image_path).convert("RGB")
    encoding = _processor_encode(processor, image=image, words=words, boxes=boxes, truncation=True, max_length=max_length)
    model_inputs = {key: value for key, value in encoding.items() if hasattr(value, "to")}
    word_ids = encoding.word_ids(batch_index=0)

    return InferenceEncoding(
        encoding=encoding,
        model_inputs=model_inputs,
        word_ids=word_ids,
        words=words,
        boxes=boxes,
        truncated=truncated,
    )
