import numpy as np
import easyocr

_reader = None

def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

def run_easyocr(img_gray_or_bin: np.ndarray) -> list[dict]:
    reader = get_reader()
    results = reader.readtext(img_gray_or_bin)

    parsed = []
    for bbox, text, conf in results:
        parsed.append({"text": text, "conf": float(conf), "bbox": bbox})
    return parsed

def best_text(results: list[dict]) -> tuple[str, float]:
    if not results:
        return "", 0.0
    best = max(results, key=lambda r: r["conf"])
    return best["text"], best["conf"]
