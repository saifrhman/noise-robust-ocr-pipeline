from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schemas import (
    BoundingBox,
    ExtractionMetadata,
    InvoiceInfo,
    OCRLine,
    OCRToken,
    ReceiptExtractionResult,
    ReceiptSample,
    TotalsInfo,
    VendorInfo,
)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(slots=True)
class SROIESplitPaths:
    """Physical paths for one SROIE split folder."""

    image_dir: Path
    box_dir: Path
    entity_dir: Path


class SROIEDatasetLoader:
    """
    Defensive loader for SROIE-style datasets already present in this repository.

    Supported structure:
        data/SROIE2019/
            train/
                img/
                box/
                entities/
            test/
                img/
                box/
                entities/

    Validation split behavior:
    - If no physical `val/` folder exists, `val` is derived from `train` using a deterministic split.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

    def available_splits(self) -> list[str]:
        """Return discovered physical splits and synthetic `val` when train exists."""
        out: list[str] = []
        for split in ("train", "val", "test"):
            if (self.dataset_root / split).exists():
                out.append(split)
        if "train" in out and "val" not in out:
            out.append("val")
        return out

    def iter_samples(
        self,
        split: str,
        *,
        val_ratio: float = 0.1,
        seed: int = 42,
        strict: bool = False,
    ) -> Iterator[ReceiptSample]:
        """Yield parsed `ReceiptSample` objects for the requested split."""
        split = split.lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        sample_ids = self._get_split_ids(split=split, val_ratio=val_ratio, seed=seed)
        for sample_id in sample_ids:
            try:
                yield self.load_sample(sample_id=sample_id, split=split)
            except Exception:
                if strict:
                    raise

    def load_sample(self, sample_id: str, split: str) -> ReceiptSample:
        """Load one sample by id for the requested split."""
        split = split.lower().strip()
        base_split = "train" if split == "val" and not (self.dataset_root / "val").exists() else split

        paths = self._resolve_split_paths(base_split)
        image_path = self._find_image_path(paths.image_dir, sample_id)
        box_path = paths.box_dir / f"{sample_id}.txt"
        entity_path = paths.entity_dir / f"{sample_id}.txt"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")
        if not box_path.exists():
            raise FileNotFoundError(f"OCR box file not found for sample {sample_id}: {box_path}")

        width, height = self._read_image_size(image_path)
        ocr_lines = self._parse_box_file(box_path)
        ocr_tokens: list[OCRToken] = []
        for line in ocr_lines:
            ocr_tokens.extend(line.tokens)

        raw_text = "\n".join(line.text for line in ocr_lines if line.text).strip()
        ground_truth = self._parse_entity_file(entity_path, source_image=image_path.name)

        return ReceiptSample(
            sample_id=sample_id,
            split=split,
            image_path=image_path,
            image_width=width,
            image_height=height,
            ocr_lines=ocr_lines,
            ocr_tokens=ocr_tokens,
            raw_ocr_text=raw_text,
            ground_truth=ground_truth,
        )

    def _resolve_split_paths(self, split: str) -> SROIESplitPaths:
        split_root = self.dataset_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        image_dir = self._pick_existing_dir(split_root, ("img", "images", "image"))
        box_dir = self._pick_existing_dir(split_root, ("box", "boxes", "ocr", "ocr_box"))
        entity_dir = self._pick_existing_dir(split_root, ("entities", "entity", "gt", "annotations"))

        return SROIESplitPaths(image_dir=image_dir, box_dir=box_dir, entity_dir=entity_dir)

    @staticmethod
    def _pick_existing_dir(root: Path, candidates: tuple[str, ...]) -> Path:
        for name in candidates:
            candidate = root / name
            if candidate.exists() and candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"None of expected folders found under {root}: {candidates}")

    @staticmethod
    def _find_image_path(image_dir: Path, sample_id: str) -> Path:
        for ext in _IMAGE_EXTENSIONS:
            p = image_dir / f"{sample_id}{ext}"
            if p.exists():
                return p
        return image_dir / f"{sample_id}.jpg"

    @staticmethod
    def _read_image_size(image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as im:
            return im.size

    def _get_split_ids(self, split: str, val_ratio: float, seed: int) -> list[str]:
        if split == "test":
            return self._collect_split_ids("test")

        if split == "train" and (self.dataset_root / "val").exists():
            return self._collect_split_ids("train")

        if split == "val" and (self.dataset_root / "val").exists():
            return self._collect_split_ids("val")

        train_ids = self._collect_split_ids("train")
        if not train_ids:
            return []

        # Deterministic train/val partition when only train+test exist.
        train_ids = sorted(set(train_ids))
        rng = random.Random(seed)
        shuffled = train_ids[:]
        rng.shuffle(shuffled)

        val_count = max(1, int(len(shuffled) * max(0.0, min(val_ratio, 0.5))))
        val_ids = sorted(shuffled[:val_count])
        train_only_ids = sorted(shuffled[val_count:])

        return train_only_ids if split == "train" else val_ids

    def _collect_split_ids(self, split: str) -> list[str]:
        paths = self._resolve_split_paths(split)
        ids: set[str] = set()

        for p in paths.image_dir.iterdir():
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                ids.add(p.stem)

        # Keep only IDs that have OCR box files. Entity files are optional for inference.
        valid_ids = [sample_id for sample_id in ids if (paths.box_dir / f"{sample_id}.txt").exists()]
        return sorted(valid_ids)

    @staticmethod
    def _line_to_bbox(parts: list[str]) -> BoundingBox | None:
        if len(parts) < 8:
            return None
        try:
            coords = [int(float(v.strip())) for v in parts[:8]]
        except ValueError:
            return None

        xs = [coords[0], coords[2], coords[4], coords[6]]
        ys = [coords[1], coords[3], coords[5], coords[7]]
        return BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    def _parse_box_file(self, box_path: Path) -> list[OCRLine]:
        lines: list[OCRLine] = []
        raw_lines = box_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for i, raw_line in enumerate(raw_lines):
            row = raw_line.strip()
            if not row:
                continue

            parts = row.split(",")
            if len(parts) < 9:
                continue

            bbox = self._line_to_bbox(parts)
            if bbox is None:
                continue

            text = ",".join(parts[8:]).strip()
            if not text:
                continue

            line_tokens: list[OCRToken] = []
            for token in text.split():
                token_clean = token.strip()
                if not token_clean:
                    continue
                line_tokens.append(OCRToken(text=token_clean, bbox=bbox, confidence=None, line_id=i))

            line = OCRLine(
                line_id=i,
                text=text,
                bbox=bbox,
                tokens=line_tokens,
                confidence=None,
            )
            lines.append(line)

        return lines

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _parse_entity_file(self, entity_path: Path, source_image: str) -> ReceiptExtractionResult | None:
        if not entity_path.exists():
            return None

        content = entity_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return None

        entity_map: dict[str, str] = {}

        # Preferred SROIE format: JSON object in txt file.
        if content.startswith("{") and content.endswith("}"):
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        entity_map[str(key).strip().lower()] = self._normalize_spaces(str(value))
            except json.JSONDecodeError:
                pass

        # Fallback for colon-separated lines.
        if not entity_map:
            for line in content.splitlines():
                row = line.strip()
                if not row:
                    continue
                if ":" not in row:
                    continue
                key, value = row.split(":", 1)
                entity_map[key.strip().lower()] = self._normalize_spaces(value)

        if not entity_map:
            return None

        total_raw = entity_map.get("total", "")
        total_value = self._parse_float_safe(total_raw)

        result = ReceiptExtractionResult(
            vendor=VendorInfo(
                name=entity_map.get("company", ""),
                registration_number="",
                address=entity_map.get("address", ""),
            ),
            invoice=InvoiceInfo(
                date=entity_map.get("date", ""),
            ),
            totals=TotalsInfo(
                total=total_value,
            ),
            metadata=ExtractionMetadata(
                mode="ground_truth",
                confidence=1.0,
                source_image=source_image,
            ),
        )
        return result

    @staticmethod
    def _parse_float_safe(value: str) -> float:
        text = (value or "").replace(",", ".")
        text = re.sub(r"[^0-9.]", "", text)
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0
