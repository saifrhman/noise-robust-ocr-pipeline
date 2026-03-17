"""
Error bucketing and categorization for receipts.

Automatically categorizes common extraction failures:
- Missing critical fields (vendor, date, total)
- Amount parsing failures
- Item grouping issues
- Low-confidence semantic extraction
- Hybrid fallback patterns
- Schema compatibility warnings

Produces summary JSON and readable text/markdown reports.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from src.receipt_ai.evaluation.field_comparison import (
    FieldComparison,
    _is_present,
    normalize_amount,
)


# ==============================================================================
# ERROR BUCKET DEFINITIONS
# ==============================================================================


ERROR_BUCKETS = [
    "missing_vendor",
    "missing_date",
    "missing_total",
    "amount_parse_failure",
    "item_grouping_failure",
    "low_confidence_semantic",
    "hybrid_fallback_to_rules",
    "schema_reduced_warning",
    "all_modes_agree_but_empty",
    "single_mode_fails",
]


@dataclass
class ErrorBucket:
    """Single error bucket entry."""
    
    bucket_name: str
    sample_id: str
    field_name: str = ""
    description: str = ""
    severity: str = "medium"  # high, medium, low
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleErrorBuckets:
    """All error buckets for a single sample."""
    
    sample_id: str
    buckets: dict[str, list[ErrorBucket]] = field(default_factory=dict)
    
    # Quick access to most critical issues
    critical_errors: list[ErrorBucket] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "buckets": {
                bucket_name: [e.to_dict() for e in entries]
                for bucket_name, entries in self.buckets.items()
            },
            "critical_errors": [e.to_dict() for e in self.critical_errors],
        }


def bucket_errors(
    sample_id: str,
    field_comparisons: list[FieldComparison],
    easyocr_result: dict[str, Any],
    layoutlm_result: dict[str, Any],
    hybrid_result: dict[str, Any],
) -> SampleErrorBuckets:
    """
    Categorize errors into buckets based on field comparisons and mode outputs.
    
    Returns SampleErrorBuckets with all errors organized by type.
    """
    buckets: dict[str, list[ErrorBucket]] = {bucket: [] for bucket in ERROR_BUCKETS}
    
    critical_errors: list[ErrorBucket] = []
    
    # Helper to extract field values
    def get_field(result: dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        current = result
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
    
    # Check for missing critical fields
    vendor_name = next(
        (c for c in field_comparisons if c.field_name == "vendor.name"), None
    )
    if vendor_name and not vendor_name.easyocr_present and not vendor_name.layoutlm_present:
        error = ErrorBucket(
            bucket_name="missing_vendor",
            sample_id=sample_id,
            field_name="vendor.name",
            description="Vendor name not extracted by any mode",
            severity="high",
        )
        buckets["missing_vendor"].append(error)
        critical_errors.append(error)
    
    invoice_date = next(
        (c for c in field_comparisons if c.field_name == "invoice.date"), None
    )
    if invoice_date and not invoice_date.easyocr_present and not invoice_date.layoutlm_present:
        error = ErrorBucket(
            bucket_name="missing_date",
            sample_id=sample_id,
            field_name="invoice.date",
            description="Invoice date not extracted by any mode",
            severity="high",
        )
        buckets["missing_date"].append(error)
        critical_errors.append(error)
    
    totals_total = next(
        (c for c in field_comparisons if c.field_name == "totals.total"), None
    )
    if totals_total and not totals_total.easyocr_present and not totals_total.layoutlm_present:
        error = ErrorBucket(
            bucket_name="missing_total",
            sample_id=sample_id,
            field_name="totals.total",
            description="Total amount not extracted by any mode",
            severity="high",
        )
        buckets["missing_total"].append(error)
        critical_errors.append(error)
    
    # Check for amount parsing failures (extracted but invalid)
    amount_fields = [
        "totals.subtotal",
        "totals.tax",
        "totals.total",
        "payment.amount_paid",
    ]
    for field_name in amount_fields:
        comp = next(
            (c for c in field_comparisons if c.field_name == field_name), None
        )
        if comp:
            # If present but normalized to 0.0, likely parse failure
            if comp.easyocr_present and comp.easyocr_rules_norm == 0.0:
                error = ErrorBucket(
                    bucket_name="amount_parse_failure",
                    sample_id=sample_id,
                    field_name=field_name,
                    description=f"easyocr_rules: Could not parse amount '{comp.easyocr_rules_raw}'",
                    severity="medium",
                )
                buckets["amount_parse_failure"].append(error)
            
            if comp.layoutlm_present and comp.layoutlm_only_norm == 0.0:
                error = ErrorBucket(
                    bucket_name="amount_parse_failure",
                    sample_id=sample_id,
                    field_name=field_name,
                    description=f"layoutlm_only: Could not parse amount '{comp.layoutlm_only_raw}'",
                    severity="medium",
                )
                buckets["amount_parse_failure"].append(error)
    
    # Check for item extraction failures
    easyocr_items = get_field(easyocr_result, "items") or []
    layoutlm_items = get_field(layoutlm_result, "items") or []
    hybrid_items = get_field(hybrid_result, "items") or []
    
    # Item grouping failure: one mode has items, others don't
    item_counts = [len(easyocr_items), len(layoutlm_items), len(hybrid_items)]
    if max(item_counts) > 0 and min(item_counts) == 0:
        error = ErrorBucket(
            bucket_name="item_grouping_failure",
            sample_id=sample_id,
            field_name="items",
            description=f"Item extraction inconsistent: easyocr={len(easyocr_items)}, "
            f"layoutlm={len(layoutlm_items)}, hybrid={len(hybrid_items)}",
            severity="medium",
        )
        buckets["item_grouping_failure"].append(error)
    
    # Check for low confidence in model
    layoutlm_meta = get_field(layoutlm_result, "metadata") or {}
    if isinstance(layoutlm_meta, dict):
        conf = layoutlm_meta.get("confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < 0.5:
            error = ErrorBucket(
                bucket_name="low_confidence_semantic",
                sample_id=sample_id,
                field_name="metadata.confidence",
                description=f"layoutlm_only low confidence: {conf:.2f}",
                severity="medium",
            )
            buckets["low_confidence_semantic"].append(error)
    
    # Check for hybrid fallback patterns
    hybrid_meta = get_field(hybrid_result, "metadata") or {}
    if isinstance(hybrid_meta, dict):
        warnings = hybrid_meta.get("warnings", []) or []
        if isinstance(warnings, list):
            for warning in warnings:
                if isinstance(warning, str) and "fallback" in warning.lower():
                    error = ErrorBucket(
                        bucket_name="hybrid_fallback_to_rules",
                        sample_id=sample_id,
                        description=f"Hybrid fallback detected: {warning}",
                        severity="low",
                    )
                    buckets["hybrid_fallback_to_rules"].append(error)
    
    # Check for schema compatibility warnings
    for result_name, result in [
        ("easyocr_rules", easyocr_result),
        ("layoutlm_only", layoutlm_result),
        ("hybrid", hybrid_result),
    ]:
        meta = get_field(result, "metadata") or {}
        if isinstance(meta, dict):
            warnings = meta.get("warnings", []) or []
            if isinstance(warnings, list):
                for warning in warnings:
                    if isinstance(warning, str) and "reduced" in warning.lower():
                        error = ErrorBucket(
                            bucket_name="schema_reduced_warning",
                            sample_id=sample_id,
                            description=f"{result_name}: {warning}",
                            severity="low",
                        )
                        buckets["schema_reduced_warning"].append(error)
    
    # Check for all_modes_agree_but_empty
    for comp in field_comparisons:
        if (
            not comp.easyocr_present
            and not comp.layoutlm_present
            and not comp.hybrid_present
            and comp.field_name in ["vendor.name", "invoice.date", "totals.total"]
        ):
            error = ErrorBucket(
                bucket_name="all_modes_agree_but_empty",
                sample_id=sample_id,
                field_name=comp.field_name,
                description=f"All modes empty: {comp.field_name}",
                severity="high",
            )
            buckets["all_modes_agree_but_empty"].append(error)
            critical_errors.append(error)
    
    # Check for single mode fails
    for comp in field_comparisons:
        present_count = sum([comp.easyocr_present, comp.layoutlm_present, comp.hybrid_present])
        if present_count == 1:
            failing_modes = []
            if not comp.easyocr_present:
                failing_modes.append("easyocr_rules")
            if not comp.layoutlm_present:
                failing_modes.append("layoutlm_only")
            if not comp.hybrid_present:
                failing_modes.append("hybrid")
            
            error = ErrorBucket(
                bucket_name="single_mode_fails",
                sample_id=sample_id,
                field_name=comp.field_name,
                description=f"Failed in: {', '.join(failing_modes)}",
                severity="low",
            )
            if comp.field_name in ["vendor.name", "invoice.date", "totals.total"]:
                error.severity = "medium"
            buckets["single_mode_fails"].append(error)
    
    # Remove empty buckets
    buckets = {k: v for k, v in buckets.items() if v}
    
    return SampleErrorBuckets(
        sample_id=sample_id,
        buckets=buckets,
        critical_errors=critical_errors,
    )


# ==============================================================================
# REPORTING
# ==============================================================================


@dataclass
class ErrorSummaryReport:
    """Summary report across all samples."""
    
    total_samples: int = 0
    total_errors: int = 0
    critical_error_samples: int = 0
    
    # Count per bucket
    bucket_counts: dict[str, int] = field(default_factory=dict)
    
    # Top problematic samples
    worst_samples: list[tuple[str, int]] = field(default_factory=list)  # [(sample_id, error_count), ...]
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_all_errors(
    sample_buckets_list: list[SampleErrorBuckets],
) -> ErrorSummaryReport:
    """Create summary report across all samples."""
    report = ErrorSummaryReport(total_samples=len(sample_buckets_list))
    
    bucket_counts: dict[str, int] = {bucket: 0 for bucket in ERROR_BUCKETS}
    
    sample_error_counts: list[tuple[str, int]] = []
    
    for sample_buckets in sample_buckets_list:
        if sample_buckets.critical_errors:
            report.critical_error_samples += 1
        
        sample_error_count = 0
        for bucket_name, errors in sample_buckets.buckets.items():
            count = len(errors)
            bucket_counts[bucket_name] += count
            sample_error_count += count
        
        report.total_errors += sample_error_count
        sample_error_counts.append((sample_buckets.sample_id, sample_error_count))
    
    # Keep only non-zero buckets
    report.bucket_counts = {k: v for k, v in bucket_counts.items() if v > 0}
    
    # Top 10 worst samples
    sample_error_counts.sort(key=lambda x: x[1], reverse=True)
    report.worst_samples = sample_error_counts[:10]
    
    return report


def format_error_report_markdown(
    summary: ErrorSummaryReport,
    sample_buckets_list: list[SampleErrorBuckets],
) -> str:
    """Format error summary as readable markdown."""
    lines = ["# Error Analysis Report\n"]
    
    lines.append("## Summary\n")
    lines.append(f"- Total samples: {summary.total_samples}\n")
    lines.append(f"- Total errors: {summary.total_errors}\n")
    lines.append(f"- Samples with critical errors: {summary.critical_error_samples}\n")
    
    if summary.bucket_counts:
        lines.append("\n## Error Buckets\n")
        for bucket_name in ERROR_BUCKETS:
            count = summary.bucket_counts.get(bucket_name, 0)
            if count > 0:
                percentage = 100.0 * count / summary.total_samples
                lines.append(f"- {bucket_name}: {count} ({percentage:.1f}%)\n")
    
    if summary.worst_samples:
        lines.append("\n## Top Problematic Samples\n")
        for i, (sample_id, error_count) in enumerate(summary.worst_samples, 1):
            lines.append(f"{i}. {sample_id} ({error_count} errors)\n")
    
    return "".join(lines)
