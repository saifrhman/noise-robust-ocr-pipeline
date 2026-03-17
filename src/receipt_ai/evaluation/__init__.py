"""Evaluation and comparison tools for extraction mode analysis."""

from src.receipt_ai.evaluation.field_comparison import (
    FieldComparison,
    ItemAnalysis,
    compare_fields,
    analyze_items,
    normalize_text,
    normalize_date,
    normalize_amount,
)
from src.receipt_ai.evaluation.disagreement_analysis import (
    DisagreementCase,
    SampleDisagreementSummary,
    analyze_disagreements,
    summarize_sample_disagreements,
)
from src.receipt_ai.evaluation.error_bucketing import (
    ErrorBucket,
    SampleErrorBuckets,
    ErrorSummaryReport,
    bucket_errors,
    summarize_all_errors,
    format_error_report_markdown,
)

__all__ = [
    "FieldComparison",
    "ItemAnalysis",
    "compare_fields",
    "analyze_items",
    "normalize_text",
    "normalize_date",
    "normalize_amount",
    "DisagreementCase",
    "SampleDisagreementSummary",
    "analyze_disagreements",
    "summarize_sample_disagreements",
    "ErrorBucket",
    "SampleErrorBuckets",
    "ErrorSummaryReport",
    "bucket_errors",
    "summarize_all_errors",
    "format_error_report_markdown",
]
