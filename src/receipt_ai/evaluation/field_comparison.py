"""
Field-by-field comparison metrics for receipt extraction modes.

Provides smart normalization and comparison for:
- Text fields (vendor, address, cashier, payment_method)
- Date fields (normalized to YYYY-MM-DD)
- Amount fields (with numeric tolerance)
- Invoice numbers and identifiers
- Item-level analysis (counts, coherence, implied totals)

Comparisons are marked explicitly as gold (ground truth), heuristic, or uncertain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


# ==============================================================================
# FIELD DEFINITIONS
# ==============================================================================

COMPARISON_FIELDS = [
    # Vendor info
    ("vendor.name", "text"),
    ("vendor.address", "text"),
    ("vendor.registration_number", "text"),
    # Invoice info
    ("invoice.bill_number", "invoice_number"),
    ("invoice.order_number", "invoice_number"),
    ("invoice.table_number", "invoice_number"),
    ("invoice.date", "date"),
    ("invoice.time", "time"),
    ("invoice.cashier", "text"),
    ("invoice.invoice_type", "text"),
    # Totals
    ("totals.subtotal", "amount"),
    ("totals.tax", "amount"),
    ("totals.total", "amount"),
    ("totals.service_charge", "amount"),
    ("totals.rounding", "amount"),
    ("totals.currency", "text"),
    # Payment
    ("payment.method", "text"),
    ("payment.amount_paid", "amount"),
]


# ==============================================================================
# NORMALIZATION HELPERS
# ==============================================================================


def normalize_text(value: str) -> str:
    """Normalize text: lowercase, strip, collapse whitespace."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.lower().strip())


def normalize_date(value: str) -> str:
    """
    Normalize date to YYYY-MM-DD.
    
    Handles:
    - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    - Partial dates (blank returns "")
    """
    if not value:
        return ""
    
    value = value.strip()
    # Standardize separators
    value = value.replace("-", "/").replace(".", "/")
    value = re.sub(r"\s+", "", value)
    
    # Try DD/MM/YYYY pattern
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})", value)
    if match:
        dd, mm, yyyy = match.groups()
        if len(yyyy) == 2:
            yyyy = "19" + yyyy if int(yyyy) >= 50 else "20" + yyyy
        try:
            d = int(dd)
            m = int(mm)
            y = int(yyyy)
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                return f"{y:04d}-{m:02d}-{d:02d}"
        except (ValueError, TypeError):
            pass
    
    # Return as-is if unparseable
    return value


def normalize_amount(value: str | float) -> float:
    """
    Normalize amount to float.
    
    Handles:
    - Currency symbols ($, RM, BHD, etc.)
    - Thousand separators (,)
    - Decimal points
    - OCR errors (O→0, l→1)
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not value:
        return 0.0
    
    value = str(value).strip()
    
    # Remove currency symbols
    value = re.sub(r"[$£€₹RM BHD]", "", value, flags=re.IGNORECASE)
    
    # Replace common OCR errors
    value = re.sub(r"(?<=\d)[oO](?=[\d.])", "0", value)
    value = value.replace("l", "1")
    
    # Remove spaces (thousand separators)
    value = value.replace(" ", "").replace(",", ".")
    
    # Extract first numeric match
    match = re.search(r"(\d{1,10})(?:\.(\d{1,2}))?", value)
    if match:
        try:
            int_part = int(match.group(1))
            dec_part = match.group(2) or "00"
            if len(dec_part) == 1:
                dec_part = dec_part + "0"
            elif len(dec_part) > 2:
                dec_part = dec_part[:2]
            return float(f"{int_part}.{dec_part}")
        except (ValueError, TypeError):
            pass
    
    return 0.0


def normalize_invoice_number(value: str) -> str:
    """Normalize invoice/bill/order/table numbers: no spaces, lowercase."""
    if not value:
        return ""
    return value.strip().replace(" ", "").lower()


def normalize_time(value: str) -> str:
    """Normalize time to HH:MM or HH:MM:SS."""
    if not value:
        return ""
    
    value = value.strip()
    match = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", value)
    if match:
        hh, mm, ss = match.groups()
        ss = ss or "00"
        # Validate ranges
        try:
            h = int(hh)
            m = int(mm)
            s = int(ss)
            if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                return f"{h:02d}:{m:02d}:{s:02d}"
        except (ValueError, TypeError):
            pass
    
    return value


# ==============================================================================
# FIELD COMPARISON LOGIC
# ==============================================================================


def normalize_field(value: str | float, field_type: str) -> str | float:
    """Apply appropriate normalization based on field type."""
    if isinstance(value, float) and field_type == "amount":
        return value
    
    value_str = str(value) if value else ""
    
    if field_type == "text":
        return normalize_text(value_str)
    elif field_type == "date":
        return normalize_date(value_str)
    elif field_type == "amount":
        return normalize_amount(value_str)
    elif field_type == "invoice_number":
        return normalize_invoice_number(value_str)
    elif field_type == "time":
        return normalize_time(value_str)
    else:
        return normalize_text(value_str)


def amounts_match(val1: float, val2: float, tolerance: float = 0.01) -> bool:
    """Check if two amounts match within tolerance."""
    if val1 == 0.0 and val2 == 0.0:
        return True
    if val1 == 0.0 or val2 == 0.0:
        return False
    return abs(val1 - val2) / max(abs(val1), abs(val2)) <= tolerance


def text_match(val1: str, val2: str) -> bool:
    """Check if two text values match (both empty or both same after normalization)."""
    n1 = normalize_text(val1)
    n2 = normalize_text(val2)
    if not n1 and not n2:
        return True
    return n1 == n2


@dataclass
class FieldComparison:
    """Per-field comparison result."""
    field_name: str
    field_type: str
    
    # Per-mode values
    easyocr_rules_raw: str | float = ""
    layoutlm_only_raw: str | float = ""
    hybrid_raw: str | float = ""
    
    # Normalized values for comparison
    easyocr_rules_norm: str | float = ""
    layoutlm_only_norm: str | float = ""
    hybrid_norm: str | float = ""
    
    # Ground truth (if available)
    ground_truth_raw: str | float = ""
    ground_truth_norm: str | float = ""
    
    # Comparison results
    easyocr_vs_layoutlm: bool = False  # Do they match?
    easyocr_vs_hybrid: bool = False
    layoutlm_vs_hybrid: bool = False
    easyocr_vs_truth: bool | None = None
    layoutlm_vs_truth: bool | None = None
    hybrid_vs_truth: bool | None = None
    
    # Coverage info
    easyocr_present: bool = False  # Not empty/zero
    layoutlm_present: bool = False
    hybrid_present: bool = False
    truth_present: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_fields(
    easyocr_result: dict[str, Any],
    layoutlm_result: dict[str, Any],
    hybrid_result: dict[str, Any],
    ground_truth: dict[str, Any] | None = None,
) -> list[FieldComparison]:
    """
    Compare all configured fields across three modes.
    
    Args:
        easyocr_result: Result dict from easyocr_rules mode
        layoutlm_result: Result dict from layoutlm_only mode
        hybrid_result: Result dict from hybrid mode
        ground_truth: Optional ground truth dict (same structure as results)
    
    Returns:
        List of FieldComparison objects with all comparisons
    """
    comparisons = []
    
    for field_path, field_type in COMPARISON_FIELDS:
        # Extract values from each result
        easyocr_val = _get_nested_field(easyocr_result, field_path)
        layoutlm_val = _get_nested_field(layoutlm_result, field_path)
        hybrid_val = _get_nested_field(hybrid_result, field_path)
        truth_val = _get_nested_field(ground_truth, field_path) if ground_truth else None
        
        # Normalize
        easyocr_norm = normalize_field(easyocr_val, field_type)
        layoutlm_norm = normalize_field(layoutlm_val, field_type)
        hybrid_norm = normalize_field(hybrid_val, field_type)
        truth_norm = normalize_field(truth_val, field_type) if truth_val is not None else ""
        
        # Check if present (non-empty or non-zero)
        easyocr_present = _is_present(easyocr_norm)
        layoutlm_present = _is_present(layoutlm_norm)
        hybrid_present = _is_present(hybrid_norm)
        truth_present = _is_present(truth_norm)
        
        # Check matches
        if field_type == "amount":
            easyocr_vs_layoutlm = amounts_match(easyocr_norm, layoutlm_norm)  # type: ignore
            easyocr_vs_hybrid = amounts_match(easyocr_norm, hybrid_norm)  # type: ignore
            layoutlm_vs_hybrid = amounts_match(layoutlm_norm, hybrid_norm)  # type: ignore
            easyocr_vs_truth = amounts_match(easyocr_norm, truth_norm) if truth_present else None  # type: ignore
            layoutlm_vs_truth = amounts_match(layoutlm_norm, truth_norm) if truth_present else None  # type: ignore
            hybrid_vs_truth = amounts_match(hybrid_norm, truth_norm) if truth_present else None  # type: ignore
        else:
            easyocr_vs_layoutlm = text_match(str(easyocr_norm), str(layoutlm_norm)) if easyocr_present and layoutlm_present else False
            easyocr_vs_hybrid = text_match(str(easyocr_norm), str(hybrid_norm)) if easyocr_present and hybrid_present else False
            layoutlm_vs_hybrid = text_match(str(layoutlm_norm), str(hybrid_norm)) if layoutlm_present and hybrid_present else False
            easyocr_vs_truth = text_match(str(easyocr_norm), str(truth_norm)) if truth_present else None
            layoutlm_vs_truth = text_match(str(layoutlm_norm), str(truth_norm)) if truth_present else None
            hybrid_vs_truth = text_match(str(hybrid_norm), str(truth_norm)) if truth_present else None
        
        comp = FieldComparison(
            field_name=field_path,
            field_type=field_type,
            easyocr_rules_raw=easyocr_val,
            layoutlm_only_raw=layoutlm_val,
            hybrid_raw=hybrid_val,
            easyocr_rules_norm=easyocr_norm,
            layoutlm_only_norm=layoutlm_norm,
            hybrid_norm=hybrid_norm,
            ground_truth_raw=truth_val or "",
            ground_truth_norm=truth_norm,
            easyocr_vs_layoutlm=easyocr_vs_layoutlm,
            easyocr_vs_hybrid=easyocr_vs_hybrid,
            layoutlm_vs_hybrid=layoutlm_vs_hybrid,
            easyocr_vs_truth=easyocr_vs_truth,
            layoutlm_vs_truth=layoutlm_vs_truth,
            hybrid_vs_truth=hybrid_vs_truth,
            easyocr_present=easyocr_present,
            layoutlm_present=layoutlm_present,
            hybrid_present=hybrid_present,
            truth_present=truth_present,
        )
        comparisons.append(comp)
    
    return comparisons


# ==============================================================================
# ITEM-LEVEL ANALYSIS (HEURISTIC)
# ==============================================================================


@dataclass
class ItemAnalysis:
    """Per-sample item-level comparison (marked as heuristic)."""
    
    sample_id: str
    analysis_type: str = "heuristic"  # Because item labels are weak
    
    # Item counts
    easyocr_item_count: int = 0
    layoutlm_item_count: int = 0
    hybrid_item_count: int = 0
    
    # Implied totals (sum of item line_totals)
    easyocr_implied_total: float = 0.0
    layoutlm_implied_total: float = 0.0
    hybrid_implied_total: float = 0.0
    
    # vs extracted total
    easyocr_total: float = 0.0
    layoutlm_total: float = 0.0
    hybrid_total: float = 0.0
    
    # Coherence checks
    easyocr_items_coherent: bool = False  # All items have name+price
    layoutlm_items_coherent: bool = False
    hybrid_items_coherent: bool = False
    
    # Notes for human review
    notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_items(
    easyocr_result: dict[str, Any],
    layoutlm_result: dict[str, Any],
    hybrid_result: dict[str, Any],
    sample_id: str = "",
) -> ItemAnalysis:
    """
    Analyze items across modes (marked explicitly as heuristic).
    
    Checks:
    - Item count per mode
    - Implied total (sum of item line_totals) vs extracted total
    - Item coherence (all items have name + price)
    """
    analysis = ItemAnalysis(sample_id=sample_id)
    
    # Helper to get items and compute stats
    def analyze_mode_items(result: dict[str, Any]) -> tuple[int, float, bool]:
        items = _get_nested_field(result, "items") or []
        count = len(items) if items else 0
        
        # Compute implied total from line_totals
        implied = 0.0
        coherent = True
        
        if count > 0:
            for item in items:
                if isinstance(item, dict):
                    try:
                        line_total = float(item.get("line_total", 0.0))
                        implied += line_total
                    except (ValueError, TypeError):
                        pass
                    
                    # Check coherence: has name and price
                    name = str(item.get("name", "")).strip()
                    price = item.get("unit_price", 0.0)
                    if not name or (isinstance(price, (int, float)) and price == 0.0):
                        coherent = False
        
        return count, implied, coherent
    
    # Analyze each mode
    easyocr_count, easyocr_implied, easyocr_coherent = analyze_mode_items(easyocr_result)
    layoutlm_count, layoutlm_implied, layoutlm_coherent = analyze_mode_items(layoutlm_result)
    hybrid_count, hybrid_implied, hybrid_coherent = analyze_mode_items(hybrid_result)
    
    # Get extracted totals
    easyocr_total = normalize_amount(_get_nested_field(easyocr_result, "totals.total"))
    layoutlm_total = normalize_amount(_get_nested_field(layoutlm_result, "totals.total"))
    hybrid_total = normalize_amount(_get_nested_field(hybrid_result, "totals.total"))
    
    # Fill analysis
    analysis.easyocr_item_count = easyocr_count
    analysis.layoutlm_item_count = layoutlm_count
    analysis.hybrid_item_count = hybrid_count
    
    analysis.easyocr_implied_total = easyocr_implied
    analysis.layoutlm_implied_total = layoutlm_implied
    analysis.hybrid_implied_total = hybrid_implied
    
    analysis.easyocr_total = easyocr_total  # type: ignore
    analysis.layoutlm_total = layoutlm_total  # type: ignore
    analysis.hybrid_total = hybrid_total  # type: ignore
    
    analysis.easyocr_items_coherent = easyocr_coherent
    analysis.layoutlm_items_coherent = layoutlm_coherent
    analysis.hybrid_items_coherent = hybrid_coherent
    
    # Generate notes
    notes = []
    
    # Check item count differences
    if easyocr_count != layoutlm_count or easyocr_count != hybrid_count:
        counts_str = f"easyocr={easyocr_count}, layoutlm={layoutlm_count}, hybrid={hybrid_count}"
        notes.append(f"Item count differs: {counts_str}")
    
    # Check implied vs extracted totals
    for mode_name, implied, extracted in [
        ("easyocr_rules", easyocr_implied, easyocr_total),
        ("layoutlm_only", layoutlm_implied, layoutlm_total),
        ("hybrid", hybrid_implied, hybrid_total),
    ]:
        if implied > 0 and extracted > 0 and not amounts_match(implied, extracted, tolerance=0.05):
            diff = abs(implied - extracted)
            notes.append(f"{mode_name}: implied_total ({implied:.2f}) diverges from extracted ({extracted:.2f}) by {diff:.2f}")
    
    analysis.notes = notes
    return analysis


# ==============================================================================
# HELPERS
# ==============================================================================


def _get_nested_field(obj: dict[str, Any] | None, path: str) -> Any:
    """Get nested field from dict using dot notation (e.g., 'vendor.name')."""
    if obj is None:
        return None
    
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    
    return current


def _is_present(value: Any) -> bool:
    """Check if a value is present (non-empty, non-zero)."""
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return value != 0.0
    if isinstance(value, str):
        return len(value.strip()) > 0
    return bool(value)
