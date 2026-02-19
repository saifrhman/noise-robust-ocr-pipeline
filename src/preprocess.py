import cv2
import numpy as np

def preprocess_for_ocr(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    debug = {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    debug["step"] = "grayscale"

    denoised = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=7, searchWindowSize=21)
    debug["step"] = "denoise"

    thr = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    debug["step"] = "adaptive_threshold"

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    debug["step"] = "morph_close"

    return processed, debug
