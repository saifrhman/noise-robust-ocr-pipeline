"""
Optional comparison utilities for better human review.

Includes:
- Side-by-side comparison tables
- Critical-fields-only mode
- Sorting and filtering helpers
"""

from __future__ import annotations

from typing import Any
from src.receipt_ai.evaluation.field_comparison import FieldComparison
from src.receipt_ai.evaluation.disagreement_analysis import (
    DisagreementCase,
    SampleDisagreementSummary,
)


# ==============================================================================
# SIDE-BY-SIDE COMPARISON TABLE
# ==============================================================================


def format_side_by_side_table(
    field_comparisons: list[FieldComparison],
    critical_fields_only: bool = False,
    max_width: int = 80,
) -> str:
    """
    Format field comparisons as readable side-by-side table.
    
    Args:
        field_comparisons: List of FieldComparison objects
        critical_fields_only: If True, only show vendor, date, total
        max_width: Max width per value column
    
    Returns:
        Formatted table as string
    """
    critical_fields = {"vendor.name", "invoice.date", "totals.total"}
    
    # Filter fields
    comps = field_comparisons
    if critical_fields_only:
        comps = [c for c in comps if c.field_name in critical_fields]
    
    if not comps:
        return "(No fields to display)"
    
    # Build table
    def truncate(val: Any, width: int) -> str:
        s = str(val)[:width]
        return s
    
    lines = []
    
    # Header
    lines.append("=" * (120 if not critical_fields_only else 100))
    lines.append(f"{'Field':<20} | {'EasyOCR Rules':<24} | {'LayoutLM Only':<24} | {'Hybrid':<24} | {'Match?':<10}")
    lines.append("-" * (120 if not critical_fields_only else 100))
    
    # Rows
    for comp in comps:
        easyocr_str = truncate(comp.easyocr_rules_norm, 20)
        layoutlm_str = truncate(comp.layoutlm_only_norm, 20)
        hybrid_str = truncate(comp.hybrid_norm, 20)
        
        # Match status
        if comp.easyocr_present + comp.layoutlm_present + comp.hybrid_present == 0:
            match_str = "EMPTY"
        elif comp.easyocr_vs_layoutlm and comp.layoutlm_vs_hybrid:
            match_str = "✓ all"
        elif comp.easyocr_vs_layoutlm:
            match_str = "rules=lm"
        elif comp.layoutlm_vs_hybrid:
            match_str = "lm=hybrid"
        elif comp.easyocr_vs_hybrid:
            match_str = "rules=hy"
        else:
            match_str = "✗ differ"
        
        line = (
            f"{comp.field_name:<20} | "
            f"{easyocr_str:<24} | "
            f"{layoutlm_str:<24} | "
            f"{hybrid_str:<24} | "
            f"{match_str:<10}"
        )
        lines.append(line)
    
    lines.append("=" * (120 if not critical_fields_only else 100))
    
    return "\n".join(lines)


# ==============================================================================
# CRITICAL FIELDS ANALYSIS
# ==============================================================================


CRITICAL_FIELDS_SET = {
    "vendor.name",
    "invoice.date",
    "totals.total",
}


def assess_critical_fields(
    field_comparisons: list[FieldComparison],
) -> dict[str, Any]:
    """
    Quick health check on critical fields only.
    
    Returns:
        {"status": "good"|"fair"|"poor", "details": {...}}
    """
    critical_comps = [
        c for c in field_comparisons if c.field_name in CRITICAL_FIELDS_SET
    ]
    
    issues = []
    
    for comp in critical_comps:
        # Check if all empty
        if not comp.easyocr_present and not comp.layoutlm_present and not comp.hybrid_present:
            issues.append((comp.field_name, "all_empty", 3))
        
        # Check if one mode missing
        elif (comp.easyocr_present + comp.layoutlm_present + comp.hybrid_present) == 1:
            issues.append((comp.field_name, "only_one_mode", 2))
        
        # Check if disagreement
        elif not (comp.easyocr_vs_layoutlm and comp.layoutlm_vs_hybrid and comp.easyocr_vs_hybrid):
            issues.append((comp.field_name, "disagreement", 1))
    
    if not issues:
        status = "good"
    elif max(s[2] for s in issues) >= 3:
        status = "poor"
    elif max(s[2] for s in issues) >= 2:
        status = "fair"
    else:
        status = "good"
    
    return {
        "status": status,
        "critical_fields_ok": len(CRITICAL_FIELDS_SET) - len(issues),
        "critical_fields_total": len(CRITICAL_FIELDS_SET),
        "issues": [{"field": i[0], "type": i[1]} for i in issues],
    }


# ==============================================================================
# SORTING AND FILTERING
# ==============================================================================


def sort_disagreements_by_severity(
    cases: list[DisagreementCase],
) -> list[DisagreementCase]:
    """Sort disagreement cases by severity (high first), then by field."""
    severity_order = {"high": 3, "medium": 2, "low": 1}
    return sorted(
        cases,
        key=lambda c: (
            -severity_order.get(c.severity, 0),
            c.field_name,
            c.severity_score,
        ),
        reverse=True,
    )


def sort_disagreements_by_field(
    cases: list[DisagreementCase],
) -> list[DisagreementCase]:
    """Sort disagreement cases by field name (critical fields first)."""
    critical_order = {
        "vendor.name": 0,
        "invoice.date": 1,
        "totals.total": 2,
    }
    return sorted(
        cases,
        key=lambda c: (
            critical_order.get(c.field_name, 999),
            c.field_name,
            -c.severity_score,
        ),
    )


def filter_disagreements_by_severity(
    cases: list[DisagreementCase],
    min_severity: str = "medium",
) -> list[DisagreementCase]:
    """Filter to only critical/high/medium disagreements."""
    severity_order = {"high": 3, "medium": 2, "low": 1}
    min_level = severity_order.get(min_severity, 0)
    return [c for c in cases if severity_order.get(c.severity, 0) >= min_level]


def filter_disagreements_by_type(
    cases: list[DisagreementCase],
    types: list[str],
) -> list[DisagreementCase]:
    """Filter to only specific disagreement types."""
    return [c for c in cases if c.disagreement_type in types]


# ==============================================================================
# FORMATTED REPORTS
# ==============================================================================


def format_critical_fields_report(
    field_comparisons: list[FieldComparison],
    sample_id: str = "",
) -> str:
    """Format critical fields assessment as readable text."""
    assessment = assess_critical_fields(field_comparisons)
    
    lines = []
    
    if sample_id:
        lines.append(f"# Critical Fields Assessment: {sample_id}\n")
    else:
        lines.append("# Critical Fields Assessment\n")
    
    lines.append(f"Status: {assessment['status'].upper()}\n")
    lines.append(f"OK: {assessment['critical_fields_ok']}/{assessment['critical_fields_total']} critical fields\n\n")
    
    if assessment["issues"]:
        lines.append("## Issues\n\n")
        for issue in assessment["issues"]:
            lines.append(f"- {issue['field']}: {issue['type']}\n")
    else:
        lines.append("✓ All critical fields present and consistent\n")
    
    return "".join(lines)


def format_comparison_summary(
    field_comparisons: list[FieldComparison],
    sample_id: str = "",
) -> str:
    """Format quick summary of field comparison status."""
    total = len(field_comparisons)
    all_agree = sum(
        1 for c in field_comparisons
        if c.easyocr_vs_layoutlm and c.layoutlm_vs_hybrid and c.easyocr_vs_hybrid
    )
    all_empty = sum(
        1 for c in field_comparisons
        if not c.easyocr_present and not c.layoutlm_present and not c.hybrid_present
    )
    disagreements = total - all_agree - all_empty
    
    lines = []
    if sample_id:
        lines.append(f"Sample: {sample_id}\n")
    
    lines.append(f"Field Comparison Summary:\n")
    lines.append(f"  All modes agree: {all_agree}/{total} fields\n")
    lines.append(f"  All modes empty: {all_empty}/{total} fields\n")
    lines.append(f"  Disagreements: {disagreements}/{total} fields\n")
    
    if disagreements == 0:
        lines.append("  Status: ✓ GOOD alignment\n")
    elif disagreements < 3:
        lines.append("  Status: FAIR alignment (minor discrepancies)\n")
    else:
        lines.append("  Status: ✗ POOR alignment (multiple disagreements)\n")
    
    return "".join(lines)
