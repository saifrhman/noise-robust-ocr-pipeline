# src/preprocess.py
import cv2
import numpy as np


def preprocess_for_ocr(img_bgr: np.ndarray, mode: str = "clahe") -> tuple[np.ndarray, dict]:
    """
    Preprocess image for OCR.
    Modes:
      - none: grayscale only
      - denoise: grayscale + denoise
      - clahe: grayscale + contrast boost (often good for faint receipt text)
      - otsu: blur + Otsu threshold (can be harsh)
      - adaptive: denoise + adaptive threshold + morphology (often harsh on receipts)

    Returns: processed_image (uint8), debug_info
    """
    debug = {"mode": mode}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "none":
        return gray, debug

    if mode == "denoise":
        den = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        return den, debug

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(gray)
        return out, debug

    if mode == "otsu":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th, debug

    if mode == "adaptive":
        denoised = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=7, searchWindowSize=21)
        thr = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return processed, debug

    raise ValueError(f"Unknown mode: {mode}")


def candidate_modes_for_auto() -> list[str]:
    """
    Auto-mode candidates (keep it small + safe for receipts).
    We intentionally exclude otsu/adaptive by default because they often harm receipt OCR.
    """
    return ["none", "clahe", "denoise"]
