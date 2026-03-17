from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

import numpy as np
from PIL import Image
import torch

from label_config import FIELD_ALIASES, has_semantic_receipt_labels, split_bio

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
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    return processor, model, device, id2label


def _empty_fields() -> dict[str, str]:
    return {"merchant": "", "date": "", "address": "", "total": ""}


def _generic_checkpoint_warning(model_name_or_path: str, id2label: dict[int, str]) -> str:
    model_name = (model_name_or_path or "").strip()
    if model_name == "microsoft/layoutlmv3-base":
        return (
            "The selected checkpoint is microsoft/layoutlmv3-base, which is a pretrained backbone and "
            "not a receipt KIE model. Fine-tune LayoutLMv3 on semantic receipt labels first."
        )

    if not has_semantic_receipt_labels(id2label):
        return (
            "Checkpoint labels are generic (for example LABEL_0/LABEL_1) or non-receipt schema. "
            "Use a fine-tuned receipt KIE checkpoint with semantic BIO labels."
        )

    return ""


def _decode_word_predictions(
    logits: torch.Tensor,
    encoding: Any,
    words: list[str],
    abs_boxes: list[list[int]],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)[0]
    word_ids = encoding.word_ids(batch_index=0)

    per_word_label_scores: dict[int, dict[int, float]] = {}
    per_word_counts: dict[int, int] = {}

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(words):
            continue

        if word_idx not in per_word_label_scores:
            per_word_label_scores[word_idx] = {}
            per_word_counts[word_idx] = 0

        per_word_counts[word_idx] += 1
        token_probs = probs[token_idx]

        for label_id in range(token_probs.shape[0]):
            score = float(token_probs[label_id].item())
            per_word_label_scores[word_idx][label_id] = per_word_label_scores[word_idx].get(label_id, 0.0) + score

    word_predictions: list[dict[str, Any]] = []
    for word_idx in sorted(per_word_label_scores.keys()):
        label_scores = per_word_label_scores[word_idx]
        best_label_id = max(label_scores.items(), key=lambda item: item[1])[0]
        count = max(per_word_counts.get(word_idx, 1), 1)
        confidence = float(label_scores[best_label_id] / count)

        word_predictions.append(
            {
                "word": words[word_idx],
                "label": id2label.get(int(best_label_id), "O"),
                "score": confidence,
                "bbox": abs_boxes[word_idx] if word_idx < len(abs_boxes) else None,
            }
        )

    return word_predictions


def _merge_bbox(boxes: list[list[int]]) -> list[int] | None:
    valid = [box for box in boxes if len(box) == 4]
    if not valid:
        return None

    x1 = min(box[0] for box in valid)
    y1 = min(box[1] for box in valid)
    x2 = max(box[2] for box in valid)
    y2 = max(box[3] for box in valid)
    return [x1, y1, x2, y2]


def _aggregate_entities(word_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in word_predictions:
        prefix, entity_type = split_bio(str(row.get("label", "O")))

        if prefix == "O" or entity_type not in FIELD_ALIASES:
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
                "boxes": [row.get("bbox")],
            }
            continue

        current["tokens"].append(row["word"])
        current["scores"].append(row["score"])
        current["boxes"].append(row.get("bbox"))

    if current is not None:
        entities.append(current)

    out: list[dict[str, Any]] = []
    for row in entities:
        token_text = " ".join(row["tokens"]).strip()
        if not token_text:
            continue
        out.append(
            {
                "label": row["entity"],
                "text": token_text,
                "score": float(sum(row["scores"]) / max(len(row["scores"]), 1)),
                "bbox": _merge_bbox(row.get("boxes", [])),
            }
        )
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in out:
        key = (str(row["label"]), str(row["text"]).strip().upper())
        best = dedup.get(key)
        if best is None or float(row["score"]) > float(best["score"]):
            dedup[key] = row

    merged = list(dedup.values())
    merged.sort(key=lambda item: float(item["score"]), reverse=True)
    return merged


def _normalize_date(text: str) -> str:
    value = (text or "").strip()
    value = value.replace("-", "/").replace(".", "/")
    value = re.sub(r"\s+", "", value)

    match = re.search(r"\b\d{1,4}/\d{1,2}/\d{1,4}\b", value)
    if match:
        return match.group(0)
    return value


def _normalize_total(text: str) -> str:
    value = (text or "").strip()
    value = value.replace(",", ".")
    value = re.sub(r"(?<=\d)[Oo](?=[\d.])", "0", value)
    value = re.sub(r"[^0-9. ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    amounts = re.findall(r"\d+(?:\.\d{2})", value)
    if amounts:
        return amounts[-1]
    return value


def _best_field_value(entities: list[dict[str, Any]], label: str) -> str:
    candidates = [row for row in entities if str(row.get("label", "")) == label]
    if not candidates:
        return ""
    candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return str(candidates[0].get("text", "")).strip()


def _collect_fields(entities: list[dict[str, Any]]) -> dict[str, str]:
    company = _best_field_value(entities, "COMPANY")
    date = _normalize_date(_best_field_value(entities, "DATE"))
    address = _best_field_value(entities, "ADDRESS")
    total = _normalize_total(_best_field_value(entities, "TOTAL"))

    return {
        "merchant": company,
        "date": date,
        "address": address,
        "total": total,
    }


def _processor_encode(processor: Any, image: Image.Image, words: list[str], boxes: list[list[int]]) -> Any:
    """
    Transformers changed LayoutLMv3 processor inputs across versions.
    Prefer text=words and gracefully fallback to words=words.
    """
    common_kwargs = {
        "images": image,
        "boxes": boxes,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }

    try:
        return processor(text=words, **common_kwargs)
    except (TypeError, KeyError):
        return processor(words=words, **common_kwargs)


def predict_layoutlmv3_from_words(
    image_rgb: np.ndarray,
    words: list[str],
    boxes: list[list[int]],
    abs_boxes: list[list[int]],
    model_name_or_path: str,
    was_truncated: bool = False,
) -> dict[str, Any]:
    """
    Pure LayoutLMv3 token classification inference.
    Returns model-only predictions without receipt script parsing.
    """
    if image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image array")

    model_name = (model_name_or_path or "").strip()

    if not words:
        return {
            "words": [],
            "entities": [],
            "raw_entities": [],
            "fields": _empty_fields(),
            "warning": "No OCR words were available for LayoutLMv3 inference.",
            "model_is_receipt_finetuned": False,
            "was_truncated": False,
            "num_words": 0,
        }

    if model_name == "microsoft/layoutlmv3-base":
        neutral_words = [
            {
                "word": words[idx],
                "label": "O",
                "score": 0.0,
                "bbox": abs_boxes[idx] if idx < len(abs_boxes) else None,
            }
            for idx in range(len(words))
        ]
        return {
            "words": neutral_words,
            "entities": [],
            "raw_entities": [],
            "fields": _empty_fields(),
            "warning": (
                "The selected checkpoint is microsoft/layoutlmv3-base, which is a pretrained backbone and "
                "not a receipt KIE model. Fine-tune LayoutLMv3 on semantic receipt labels first."
            ),
            "model_is_receipt_finetuned": False,
            "was_truncated": was_truncated,
            "num_words": len(words),
        }

    processor, model, device, id2label = _load_layoutlmv3(model_name_or_path)
    warning = _generic_checkpoint_warning(model_name_or_path, id2label)
    if warning:
        neutral_words = [
            {
                "word": words[idx],
                "label": "O",
                "score": 0.0,
                "bbox": abs_boxes[idx] if idx < len(abs_boxes) else None,
            }
            for idx in range(len(words))
        ]
        return {
            "words": neutral_words,
            "entities": [],
            "raw_entities": [],
            "fields": _empty_fields(),
            "warning": warning,
            "model_is_receipt_finetuned": False,
            "was_truncated": was_truncated,
            "num_words": len(words),
        }

    image = Image.fromarray(image_rgb)
    encoding = _processor_encode(processor, image=image, words=words, boxes=boxes)
    model_inputs = {key: value.to(device) for key, value in encoding.items() if hasattr(value, "to")}

    with torch.inference_mode():
        outputs = model(**model_inputs)

    logits = outputs.logits.detach().cpu()
    word_predictions = _decode_word_predictions(
        logits=logits,
        encoding=encoding,
        words=words,
        abs_boxes=abs_boxes,
        id2label=id2label,
    )

    raw_entities = _aggregate_entities(word_predictions)
    fields = _collect_fields(raw_entities)

    return {
        "words": word_predictions,
        "entities": raw_entities,
        "raw_entities": raw_entities,
        "fields": fields,
        "warning": "",
        "model_is_receipt_finetuned": True,
        "was_truncated": was_truncated,
        "num_words": len(words),
    }


def predict_layoutlmv3_from_easyocr(
    image_rgb: np.ndarray,
    ocr_results: list[dict[str, Any]],
    model_name_or_path: str,
    max_words: int = 512,
) -> dict[str, Any]:
    """
    Pure LayoutLMv3 inference from EasyOCR results.
    Returns model-only predictions without receipt script parsing.
    """
    if image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image array")

    image_h, image_w = image_rgb.shape[:2]
    words, boxes, abs_boxes, was_truncated = _to_words_and_boxes(
        ocr_results=ocr_results,
        image_width=image_w,
        image_height=image_h,
        max_words=max_words,
    )

    prediction = predict_layoutlmv3_from_words(
        image_rgb=image_rgb,
        words=words,
        boxes=boxes,
        abs_boxes=abs_boxes,
        model_name_or_path=model_name_or_path,
        was_truncated=was_truncated,
    )

    return prediction


def predict_layoutlmv3_with_hybrid(
    image_rgb: np.ndarray,
    ocr_results: list[dict[str, Any]],
    model_name_or_path: str,
    max_words: int = 512,
) -> tuple[dict[str, Any], str]:
    """
    Run LayoutLMv3 inference and extract OCR text for hybrid mode.
    Returns (model_prediction, ocr_text) tuple for hybrid fusion.
    """
    if image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image array")

    # Extract OCR text
    ocr_text = "\n".join([(row.get("text") or "").strip() for row in ocr_results if row.get("text")])

    # Run pure model inference
    prediction = predict_layoutlmv3_from_easyocr(
        image_rgb=image_rgb,
        ocr_results=ocr_results,
        model_name_or_path=model_name_or_path,
        max_words=max_words,
    )

    return prediction, ocr_text
