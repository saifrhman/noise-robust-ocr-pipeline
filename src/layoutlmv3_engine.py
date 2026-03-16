from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
import torch

try:
    from transformers import AutoModelForTokenClassification, AutoProcessor
except ImportError:  # pragma: no cover - optional dependency at import time
    AutoModelForTokenClassification = None
    AutoProcessor = None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _quad_to_xyxy(quad: list[list[float]] | list[tuple[float, float]]) -> list[int]:
    xs = [int(point[0]) for point in quad]
    ys = [int(point[1]) for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_box(box_xyxy: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box_xyxy
    width = max(width, 1)
    height = max(height, 1)

    nx1 = int(1000 * x1 / width)
    ny1 = int(1000 * y1 / height)
    nx2 = int(1000 * x2 / width)
    ny2 = int(1000 * y2 / height)

    return [
        _clamp(nx1, 0, 1000),
        _clamp(ny1, 0, 1000),
        _clamp(nx2, 0, 1000),
        _clamp(ny2, 0, 1000),
    ]


def _dedup_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _to_words_and_boxes(
    ocr_results: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    max_words: int = 512,
) -> tuple[list[str], list[list[int]], list[list[int]], bool]:
    words: list[str] = []
    norm_boxes: list[list[int]] = []
    abs_boxes: list[list[int]] = []

    for row in ocr_results:
        text = (row.get("text") or "").strip()
        bbox = row.get("bbox")
        if not text or not bbox:
            continue

        xyxy = _quad_to_xyxy(bbox)
        normalized = _normalize_box(xyxy, image_width, image_height)

        for token in text.split():
            token = token.strip()
            if not token:
                continue
            words.append(token)
            norm_boxes.append(normalized)
            abs_boxes.append(xyxy)
            if len(words) >= max_words:
                return words, norm_boxes, abs_boxes, True

    return words, norm_boxes, abs_boxes, False


def _split_bio_label(label: str) -> tuple[str, str]:
    upper = (label or "O").upper()
    if upper == "O":
        return "O", "O"
    if upper.startswith("B-"):
        return "B", upper[2:]
    if upper.startswith("I-"):
        return "I", upper[2:]
    return "B", upper


@lru_cache(maxsize=2)
def _load_layoutlmv3(model_name_or_path: str):
    if AutoProcessor is None or AutoModelForTokenClassification is None:
        raise ImportError("Missing dependency: install 'transformers' to use LayoutLMv3 features.")

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_name_or_path)

    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def _aggregate_entities(word_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in word_predictions:
        prefix, entity_type = _split_bio_label(row["label"])

        if prefix == "O":
            if current is not None:
                entities.append(current)
                current = None
            continue

        if current is None or prefix == "B" or current["entity"] != entity_type:
            if current is not None:
                entities.append(current)
            current = {
                "entity": entity_type,
                "tokens": [row["word"]],
                "scores": [row["score"]],
            }
            continue

        current["tokens"].append(row["word"])
        current["scores"].append(row["score"])

    if current is not None:
        entities.append(current)

    out: list[dict[str, Any]] = []
    for row in entities:
        token_text = " ".join(row["tokens"]).strip()
        if not token_text:
            continue
        out.append(
            {
                "entity": row["entity"],
                "text": token_text,
                "score": float(sum(row["scores"]) / max(len(row["scores"]), 1)),
            }
        )
    return out


def _collect_fields(entities: list[dict[str, Any]]) -> dict[str, list[str]]:
    merchant: list[str] = []
    date: list[str] = []
    total: list[str] = []

    for row in entities:
        entity_type = row["entity"].lower()
        value = row["text"].strip()
        if not value:
            continue

        if any(key in entity_type for key in ["vendor", "merchant", "seller", "company", "store", "name"]):
            merchant.append(value)
        if "date" in entity_type:
            date.append(value)
        if any(key in entity_type for key in ["total", "amount", "sum", "grand"]):
            total.append(value)

    return {
        "merchant": _dedup_preserve_order(merchant),
        "date": _dedup_preserve_order(date),
        "total": _dedup_preserve_order(total),
    }


def predict_layoutlmv3_from_easyocr(
    image_rgb: np.ndarray,
    ocr_results: list[dict[str, Any]],
    model_name_or_path: str,
    max_words: int = 512,
) -> dict[str, Any]:
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image array")

    image_h, image_w = image_rgb.shape[:2]
    words, boxes, abs_boxes, was_truncated = _to_words_and_boxes(
        ocr_results=ocr_results,
        image_width=image_w,
        image_height=image_h,
        max_words=max_words,
    )

    if not words:
        return {
            "words": [],
            "entities": [],
            "fields": {"merchant": [], "date": [], "total": []},
            "was_truncated": False,
            "num_words": 0,
        }

    processor, model, device = _load_layoutlmv3(model_name_or_path)
    image = Image.fromarray(image_rgb)

    encoding = processor(
        images=image,
        words=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs = {key: value.to(device) for key, value in encoding.items() if hasattr(value, "to")}

    with torch.inference_mode():
        outputs = model(**model_inputs)

    logits = outputs.logits.detach().cpu()
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1)[0].tolist()

    id2label = {int(key): value for key, value in model.config.id2label.items()}

    word_ids = encoding.word_ids(batch_index=0)
    seen_word_ids: set[int] = set()
    word_predictions: list[dict[str, Any]] = []

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx in seen_word_ids:
            continue
        if word_idx >= len(words):
            continue

        seen_word_ids.add(word_idx)

        pred_id = int(pred_ids[token_idx])
        label = id2label.get(pred_id, str(pred_id))
        score = float(probs[0, token_idx, pred_id].item())

        word_predictions.append(
            {
                "word": words[word_idx],
                "label": label,
                "score": score,
                "bbox": abs_boxes[word_idx],
            }
        )

    entities = _aggregate_entities(word_predictions)
    fields = _collect_fields(entities)

    return {
        "words": word_predictions,
        "entities": entities,
        "fields": fields,
        "was_truncated": was_truncated,
        "num_words": len(words),
    }
