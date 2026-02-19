# src/app/extract_fields.py
import re

DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{2,4})\b",   # 19/02/2026 or 19-02-26
    r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",     # 2026-02-19
]

TOTAL_KEYWORDS = [
    "TOTAL", "AMOUNT DUE", "BALANCE", "GRAND TOTAL", "TOTAL DUE",
    "TOTAL AMT", "AMT PAYABLE", "PAYABLE", "PAID", "CHANGE"
]

CURRENCY_RE = r"(?:£|\$|€|RM)?\s*\d+[.,]\d{2}"


def _safe_text(text) -> str:
    return text if isinstance(text, str) else ""


def extract_date(text: str | None) -> str | None:
    t = _safe_text(text).upper()
    if not t:
        return None
    for pat in DATE_PATTERNS:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def extract_totals(text: str | None) -> list[str]:
    t = _safe_text(text).upper()
    if not t:
        return []

    candidates: list[str] = []

    # Prefer lines containing TOTAL-like keywords
    for line in t.splitlines():
        if any(k in line for k in TOTAL_KEYWORDS):
            for m in re.finditer(CURRENCY_RE, line):
                candidates.append(m.group(0).replace(" ", ""))

    # Fallback: any currency-looking numbers
    if not candidates:
        for m in re.finditer(CURRENCY_RE, t):
            candidates.append(m.group(0).replace(" ", ""))

    # De-dup preserving order
    dedup = []
    for x in candidates:
        if x not in dedup:
            dedup.append(x)
    return dedup


def guess_merchant(text: str | None) -> str | None:
    t = _safe_text(text)
    if not t:
        return None

    bad = {"TOTAL", "VAT", "RECEIPT", "INVOICE", "TAX", "GST", "THANK", "CHANGE"}
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    for ln in lines[:10]:
        up = ln.upper()
        if any(b in up for b in bad):
            continue
        if sum(ch.isalpha() for ch in ln) >= 3:
            return ln[:60]
    return None
