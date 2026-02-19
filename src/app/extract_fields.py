# src/app/extract_fields.py
from __future__ import annotations

import re
from typing import Optional, List


# ----------------------------
# Helpers
# ----------------------------
def _safe_text(text) -> str:
    return text if isinstance(text, str) else ""


def _dedup_preserve_order(items: List[str]) -> List[str]:
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


# ----------------------------
# Patterns
# ----------------------------
# Dates like 09/02/2018 or 2018-02-09
DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{2,4})\b",
    r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",
]

# Money-like number (accepts 56.80 or 56,80)
# We intentionally do NOT match pure integers to avoid picking quantities like "5"
MONEY_RE = re.compile(r"\b(\d{1,6}[.,]\d{2})\b")

# Time like 09.21 or 09:21 (avoid mistakenly treating as money)
TIMEISH_RE = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")

# Lines that typically contain "final" totals (priority order)
TOTAL_LINE_HINTS = [
    "ROUNDED TOTAL",
    "TOTAL AMT PAYABLE",
    "AMT PAYABLE",
    "AMOUNT PAYABLE",
    "TOTAL PAYABLE",
    "TOTAL:",
    "TOTAL",
    "SUB TOTAL",
    "SUBTOTAL",
    "CASH",
    "CHANGE",
]


# Merchant stopwords (don’t pick these as merchant lines)
MERCHANT_BAD = {
    "TOTAL", "SUBTOTAL", "SUB TOTAL", "INVOICE", "TAX", "GST", "THANK", "CHANGE",
    "CASH", "AMOUNT", "PAYABLE", "DATE", "TEL", "FAX", "EMAIL"
}


# ----------------------------
# Public API
# ----------------------------
def extract_date(text: Optional[str]) -> Optional[str]:
    t = _safe_text(text).upper()
    if not t:
        return None
    for pat in DATE_PATTERNS:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def extract_totals(text: Optional[str]) -> List[str]:
    """
    Extract total candidates in a receipt-aware way:
      1) Prefer numbers from lines that look like totals (TOTAL / ROUNDED / PAYABLE / CASH / CHANGE).
      2) Exclude time-like tokens (09.21) from candidates.
      3) Fallback to money-like tokens across the whole text if no hint-lines found.
    Returns de-duplicated candidates in best-first order.
    """
    raw = _safe_text(text)
    if not raw.strip():
        return []

    # Work with lines if available; otherwise treat as one line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        lines = [raw.strip()]

    # 1) Keyword/Hint line pass (best signal)
    candidates: List[str] = []
    upper_lines = [(ln, ln.upper()) for ln in lines]

    for hint in TOTAL_LINE_HINTS:
        for ln, up in upper_lines:
            if hint in up:
                for m in MONEY_RE.finditer(ln.replace(",", ".")):  # normalize comma decimal for matching
                    val = m.group(1).replace(",", ".")
                    # exclude time-like numbers such as 09.21
                    if TIMEISH_RE.fullmatch(val):
                        continue
                    candidates.append(val)

    candidates = _dedup_preserve_order(candidates)
    if candidates:
        return candidates

    # 2) Fallback: any money-like token in text (still excluding time-ish)
    fallback: List[str] = []
    norm = raw.replace(",", ".")
    for m in MONEY_RE.finditer(norm):
        val = m.group(1)
        if TIMEISH_RE.fullmatch(val):
            continue
        fallback.append(val)

    return _dedup_preserve_order(fallback)


def guess_merchant(text: Optional[str]) -> Optional[str]:
    """
    Guess merchant name:
      - Prefer early lines with letters
      - Skip common receipt meta words (TOTAL, TAX, INVOICE, etc.)
      - Works best if OCR text preserves line breaks
    """
    t = _safe_text(text)
    if not t.strip():
        return None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        # If no line breaks, just take the first ~60 chars that contain letters
        t2 = t.strip()
        return t2[:60] if any(ch.isalpha() for ch in t2) else None

    for ln in lines[:12]:
        up = ln.upper()
        if any(bad in up for bad in MERCHANT_BAD):
            continue
        # require at least 3 letters to avoid codes
        if sum(ch.isalpha() for ch in ln) >= 3:
            return ln[:60]

    return None
