import re

def normalize_text(s: str) -> str:
    s = s.upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    return s

def char_accuracy(pred: str, gt: str) -> float:
    pred_n = normalize_text(pred)
    gt_n = normalize_text(gt)

    if len(gt_n) == 0:
        return 1.0 if len(pred_n) == 0 else 0.0

    m = min(len(pred_n), len(gt_n))
    correct = sum(1 for i in range(m) if pred_n[i] == gt_n[i])
    correct -= abs(len(pred_n) - len(gt_n))
    correct = max(correct, 0)
    return correct / max(len(gt_n), 1)
