from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from label_config import RECEIPT_ENTITY_TYPES
from src.ocr_engine import run_easyocr
from src.preprocess import preprocess_for_ocr


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENTITY_DIR_CANDIDATES = ["entities", "entity", "key", "keys"]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def quad_to_xyxy(quad: Any) -> list[int]:
    """Convert EasyOCR quadrilateral or flat list to [x1, y1, x2, y2]."""
    if len(quad) == 4 and isinstance(quad[0], (int, float)):
        x1, y1, x2, y2 = [int(v) for v in quad]
        return [x1, y1, x2, y2]

    xs = [int(point[0]) for point in quad]  # type: ignore[index]
    ys = [int(point[1]) for point in quad]  # type: ignore[index]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_box_1000(box_xyxy: list[int], image_width: int, image_height: int) -> list[int]:
    """Normalize [x1, y1, x2, y2] from pixel space into LayoutLM 0-1000 space."""
    x1, y1, x2, y2 = box_xyxy
    image_width = max(image_width, 1)
    image_height = max(image_height, 1)

    return [
        clamp(int(1000 * x1 / image_width), 0, 1000),
        clamp(int(1000 * y1 / image_height), 0, 1000),
        clamp(int(1000 * x2 / image_width), 0, 1000),
        clamp(int(1000 * y2 / image_height), 0, 1000),
    ]


def normalize_text_for_match(text: str) -> str:
    value = (text or "").upper()
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_match_tokens(text: str) -> list[str]:
    return [token for token in normalize_text_for_match(text).split(" ") if token]


def extract_words_boxes_from_ocr_results(
    ocr_results: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    max_words: int = 512,
) -> dict[str, Any]:
    """
    Expand OCR lines into word-level tokens, with normalized and absolute boxes.
    """
    words: list[str] = []
    boxes_1000: list[list[int]] = []
    abs_boxes: list[list[int]] = []
    confidences: list[float] = []

    for row in ocr_results:
        text = (row.get("text") or "").strip()
        bbox = row.get("bbox")
        conf = float(row.get("conf", 0.0))
        if not text or not bbox:
            continue

        xyxy = quad_to_xyxy(bbox)
        box_1000 = normalize_box_1000(xyxy, image_width, image_height)

        for token in text.split():
            token = token.strip()
            if not token:
                continue
            words.append(token)
            boxes_1000.append(box_1000)
            abs_boxes.append(xyxy)
            confidences.append(conf)
            if len(words) >= max_words:
                return {
                    "words": words,
                    "boxes": boxes_1000,
                    "abs_boxes": abs_boxes,
                    "word_confidences": confidences,
                    "truncated": True,
                }

    return {
        "words": words,
        "boxes": boxes_1000,
        "abs_boxes": abs_boxes,
        "word_confidences": confidences,
        "truncated": False,
    }


def read_sroie_box_file(box_path: Path, image_width: int, image_height: int, max_words: int = 512) -> dict[str, Any]:
    """
    Parse SROIE OCR txt format:
      x1,y1,x2,y2,x3,y3,x4,y4,text
    """
    words: list[str] = []
    boxes_1000: list[list[int]] = []
    abs_boxes: list[list[int]] = []

    lines = box_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        row = line.strip()
        if not row:
            continue

        parts = row.split(",")
        if len(parts) < 9:
            continue

        try:
            coords = [int(float(v)) for v in parts[:8]]
        except ValueError:
            continue

        text = ",".join(parts[8:]).strip()
        if not text:
            continue

        xs = [coords[0], coords[2], coords[4], coords[6]]
        ys = [coords[1], coords[3], coords[5], coords[7]]
        xyxy = [min(xs), min(ys), max(xs), max(ys)]
        box_1000 = normalize_box_1000(xyxy, image_width, image_height)

        for token in text.split():
            token = token.strip()
            if not token:
                continue
            words.append(token)
            boxes_1000.append(box_1000)
            abs_boxes.append(xyxy)
            if len(words) >= max_words:
                return {"words": words, "boxes": boxes_1000, "abs_boxes": abs_boxes, "truncated": True}

    return {"words": words, "boxes": boxes_1000, "abs_boxes": abs_boxes, "truncated": False}


def read_image_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def run_easyocr_on_image(image_rgb: np.ndarray, ocr_mode: str = "none") -> list[dict[str, Any]]:
    processed, _ = preprocess_for_ocr(image_rgb, mode=ocr_mode)
    return run_easyocr(processed)


def canonical_entity_key(raw_key: str) -> str | None:
    value = normalize_text_for_match(raw_key)
    if not value:
        return None

    if value in {"COMPANY", "MERCHANT", "VENDOR", "STORE", "SELLER"}:
        return "COMPANY"
    if value in {"DATE", "INVOICE DATE", "BILL DATE"}:
        return "DATE"
    if value in {"ADDRESS", "ADDR"}:
        return "ADDRESS"
    if value in {"TOTAL", "TOTAL AMOUNT", "AMOUNT", "AMOUNT DUE", "GRAND TOTAL"}:
        return "TOTAL"

    return None


def parse_entity_text_lines(lines: list[str]) -> dict[str, str]:
    entities: dict[str, str] = {}

    for line in lines:
        value = line.strip()
        if not value:
            continue

        if ":" in value:
            key_raw, text_raw = value.split(":", 1)
        elif "," in value:
            key_raw, text_raw = value.split(",", 1)
        else:
            continue

        key = canonical_entity_key(key_raw)
        text = text_raw.strip()
        if key and text:
            entities[key] = text

    return entities


def load_receipt_entities(entity_path: Path | None) -> dict[str, str]:
    """Load entity annotations from json/txt into canonical COMPANY/DATE/ADDRESS/TOTAL keys."""
    if entity_path is None or not entity_path.exists():
        return {}

    if entity_path.suffix.lower() == ".json":
        try:
            payload = json.loads(entity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        if not isinstance(payload, dict):
            return {}

        entities: dict[str, str] = {}
        for key_raw, value_raw in payload.items():
            key = canonical_entity_key(str(key_raw))
            if key is None:
                continue
            value = str(value_raw).strip()
            if value:
                entities[key] = value
        return entities

    lines = entity_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return parse_entity_text_lines(lines)


def find_entity_subsequence(words_norm: list[str], entity_tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not entity_tokens or len(entity_tokens) > len(words_norm):
        return spans

    length = len(entity_tokens)
    for idx in range(len(words_norm) - length + 1):
        if words_norm[idx : idx + length] == entity_tokens:
            spans.append((idx, idx + length - 1))
    return spans


def convert_entities_to_bio_labels(words: list[str], entities: dict[str, str]) -> list[str]:
    """
    Convert receipt field string annotations into token-level BIO labels.
    If a field cannot be aligned confidently, it is left as O.
    """
    labels = ["O"] * len(words)
    words_norm = split_match_tokens(" ".join(words))

    # words_norm must keep 1:1 index with words; fallback if tokenization diverges.
    if len(words_norm) != len(words):
        words_norm = [normalize_text_for_match(word) for word in words]

    occupied: set[int] = set()
    candidates: list[tuple[int, int, str, int]] = []

    for field in RECEIPT_ENTITY_TYPES:
        value = (entities.get(field) or "").strip()
        if not value:
            continue

        entity_tokens = split_match_tokens(value)
        if not entity_tokens:
            continue

        for start, end in find_entity_subsequence(words_norm, entity_tokens):
            candidates.append((start, end, field, len(entity_tokens)))

    candidates.sort(key=lambda row: row[3], reverse=True)

    for start, end, field, _ in candidates:
        if any(idx in occupied for idx in range(start, end + 1)):
            continue
        labels[start] = f"B-{field}"
        for idx in range(start + 1, end + 1):
            labels[idx] = f"I-{field}"
        occupied.update(range(start, end + 1))

    return labels


def _find_entity_file(split_dir: Path, stem: str, entity_dir_names: list[str] | None = None) -> Path | None:
    names = entity_dir_names or ENTITY_DIR_CANDIDATES
    suffixes = [".json", ".txt"]

    for dir_name in names:
        folder = split_dir / dir_name
        if not folder.exists():
            continue
        for suffix in suffixes:
            candidate = folder / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def discover_receipt_images(split_dir: Path, image_dir_name: str = "img") -> list[Path]:
    image_dir = split_dir / image_dir_name
    if not image_dir.exists():
        return []

    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    return images


def load_sroie_split_records(
    split_dir: Path,
    image_dir_name: str = "img",
    ocr_dir_name: str = "box",
    entity_dir_names: list[str] | None = None,
    max_words: int = 512,
    use_easyocr_when_missing_ocr: bool = True,
    ocr_mode: str = "none",
) -> list[dict[str, Any]]:
    """
    Load SROIE-like split into token-level training records.

    Record shape:
      {
        "id": str,
        "image_path": str,
        "tokens": [str],
        "bboxes": [[x1,y1,x2,y2], ...],  # normalized 0-1000
        "labels": [BIO labels],
        "entities": {COMPANY/DATE/ADDRESS/TOTAL: str}
      }
    """
    records: list[dict[str, Any]] = []
    images = discover_receipt_images(split_dir, image_dir_name=image_dir_name)
    if not images:
        return records

    for image_path in images:
        stem = image_path.stem

        image_rgb = read_image_rgb(image_path)
        image_h, image_w = image_rgb.shape[:2]

        ocr_path = split_dir / ocr_dir_name / f"{stem}.txt"
        if ocr_path.exists():
            word_pack = read_sroie_box_file(ocr_path, image_width=image_w, image_height=image_h, max_words=max_words)
        elif use_easyocr_when_missing_ocr:
            ocr_results = run_easyocr_on_image(image_rgb, ocr_mode=ocr_mode)
            word_pack = extract_words_boxes_from_ocr_results(
                ocr_results=ocr_results,
                image_width=image_w,
                image_height=image_h,
                max_words=max_words,
            )
        else:
            continue

        words = word_pack.get("words", [])
        boxes = word_pack.get("boxes", [])
        if not words or len(words) != len(boxes):
            continue

        entity_path = _find_entity_file(split_dir, stem=stem, entity_dir_names=entity_dir_names)
        entities = load_receipt_entities(entity_path)
        if not entities:
            continue

        labels = convert_entities_to_bio_labels(words, entities)
        if len(labels) != len(words):
            continue

        records.append(
            {
                "id": stem,
                "image_path": str(image_path.resolve()),
                "tokens": words,
                "bboxes": boxes,
                "labels": labels,
                "entities": entities,
            }
        )

    return records


def split_records_train_val(records: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    if len(records) < 2:
        return records, records

    ratio = max(0.05, min(validation_ratio, 0.5))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    val_size = max(1, int(round(len(records) * ratio)))
    val_idx = set(indices[:val_size].tolist())

    train_rows = [row for idx, row in enumerate(records) if idx not in val_idx]
    val_rows = [row for idx, row in enumerate(records) if idx in val_idx]

    if not train_rows:
        train_rows = records[:-1]
        val_rows = records[-1:]

    return train_rows, val_rows


def prepare_words_boxes_for_inference(
    image_path: Path,
    ocr_path: Path | None = None,
    ocr_mode: str = "none",
    max_words: int = 512,
) -> dict[str, Any]:
    """Prepare image + OCR word features for receipt inference."""
    image_rgb = read_image_rgb(image_path)
    image_h, image_w = image_rgb.shape[:2]

    if ocr_path is not None and ocr_path.exists():
        pack = read_sroie_box_file(ocr_path, image_width=image_w, image_height=image_h, max_words=max_words)
    else:
        ocr_results = run_easyocr_on_image(image_rgb, ocr_mode=ocr_mode)
        pack = extract_words_boxes_from_ocr_results(
            ocr_results=ocr_results,
            image_width=image_w,
            image_height=image_h,
            max_words=max_words,
        )

    return {
        "image_rgb": image_rgb,
        "words": pack.get("words", []),
        "boxes": pack.get("boxes", []),
        "abs_boxes": pack.get("abs_boxes", []),
        "truncated": bool(pack.get("truncated", False)),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
