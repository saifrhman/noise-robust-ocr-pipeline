from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.receipt_ai.config import OCRConfig
from src.receipt_ai.schemas import BoundingBox, OCRLine, OCRToken


@dataclass(slots=True)
class OCRExtraction:
    """Internal OCR extraction payload in project schema format."""

    image_path: Path
    image_width: int
    image_height: int
    lines: list[OCRLine]
    tokens: list[OCRToken]
    mean_confidence: float


class EasyOCREngine:
    """Reusable EasyOCR wrapper that returns schema-based lines and tokens."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(list(self.config.languages), gpu=self.config.gpu)
        return self._reader

    @staticmethod
    def load_image(image_path: str | Path) -> tuple[np.ndarray, int, int, Path]:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        width, height = image.size
        arr = np.array(image)
        return arr, width, height, path

    def extract(self, image_path: str | Path) -> OCRExtraction:
        image_rgb, width, height, path = self.load_image(image_path)
        reader = self._get_reader()

        raw_results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=self.config.paragraph,
            width_ths=self.config.width_ths,
        )

        if not raw_results:
            return OCRExtraction(
                image_path=path,
                image_width=width,
                image_height=height,
                lines=[],
                tokens=[],
                mean_confidence=0.0,
            )

        rows = self._normalize_results(raw_results)
        rows = [row for row in rows if row["text"] and row["conf"] >= self.config.min_confidence]
        rows.sort(key=lambda row: (row["bbox"].y1 // max(self.config.y_sort_tolerance_px, 1), row["bbox"].x1))

        lines: list[OCRLine] = []
        tokens: list[OCRToken] = []

        for i, row in enumerate(rows):
            line_tokens = self._split_line_into_tokens(row["text"], row["bbox"], row["conf"], line_id=i)
            line = OCRLine(
                line_id=i,
                text=row["text"],
                bbox=row["bbox"],
                tokens=line_tokens,
                confidence=row["conf"],
            )
            lines.append(line)
            tokens.extend(line_tokens)

        mean_conf = float(sum(float(r["conf"]) for r in rows) / max(len(rows), 1)) if rows else 0.0
        return OCRExtraction(
            image_path=path,
            image_width=width,
            image_height=height,
            lines=lines,
            tokens=tokens,
            mean_confidence=mean_conf,
        )

    @staticmethod
    def _split_line_into_tokens(text: str, bbox: BoundingBox, conf: float, line_id: int) -> list[OCRToken]:
        out: list[OCRToken] = []
        for token in text.split():
            token_clean = token.strip()
            if not token_clean:
                continue
            out.append(OCRToken(text=token_clean, bbox=bbox, confidence=conf, line_id=line_id))
        return out

    @staticmethod
    def _normalize_results(raw_results: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in raw_results:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue

            bbox_raw, text_raw, conf_raw = row[0], row[1], row[2]
            text = str(text_raw).strip()
            if not text:
                continue

            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.0

            bbox = EasyOCREngine._to_bbox_xyxy(bbox_raw)
            if bbox is None:
                continue

            out.append({"bbox": bbox, "text": text, "conf": conf})
        return out

    @staticmethod
    def _to_bbox_xyxy(bbox_raw: Any) -> BoundingBox | None:
        try:
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4 and isinstance(bbox_raw[0], (int, float)):
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
                return BoundingBox(x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))

            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) < 4:
                return None

            xs: list[int] = []
            ys: list[int] = []
            for point in bbox_raw:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                xs.append(int(float(point[0])))
                ys.append(int(float(point[1])))

            if not xs or not ys:
                return None

            return BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
        except Exception:
            return None

    @staticmethod
    def to_layoutlm_inputs(tokens: list[OCRToken], image_width: int, image_height: int, max_words: int = 512) -> dict[str, Any]:
        """Convert OCR tokens into LayoutLMv3 words and normalized boxes."""
        words: list[str] = []
        boxes_1000: list[list[int]] = []
        confidences: list[float] = []

        for token in tokens:
            if not token.text:
                continue
            words.append(token.text)
            boxes_1000.append(token.bbox.to_layoutlm_1000(image_width, image_height))
            confidences.append(float(token.confidence or 0.0))
            if len(words) >= max_words:
                return {
                    "words": words,
                    "boxes": boxes_1000,
                    "confidences": confidences,
                    "truncated": True,
                }

        return {
            "words": words,
            "boxes": boxes_1000,
            "confidences": confidences,
            "truncated": False,
        }
