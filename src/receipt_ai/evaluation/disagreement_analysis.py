"""
Disagreement analysis across extraction modes.

Identifies and ranks cases where modes strongly disagree or have missing data,
making these failures actionable for manual inspection and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from src.receipt_ai.evaluation.field_comparison import (
    FieldComparison,
    amounts_match,
    normalize_field,
    _is_present,
)


@dataclass
class DisagreementCase:
    """Single disagreement between modes (ranked by severity)."""
    
    sample_id: str
    field_name: str
    field_type: str
    
    # What each mode says
    easyocr_value: str | float = ""
    layoutlm_value: str | float = ""
    hybrid_value: str | float = ""
    ground_truth_value: str | float = ""
    
    # Disagreement type
    disagreement_type: str = ""  # "rules_vs_model", "missing_in_mode", "total_mismatch", etc.
    severity: str = ""  # "high", "medium", "low"
    severity_score: float = 0.0  # 0.0-1.0 for ranking
    
    # Explanation
    explanation: str = ""
    
    # Hybrid behavior (if hybrid chose differently)
    hybrid_chose: str = ""  # "easyocr_rules", "layoutlm_only", or "default"
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_disagreements(
    field_comparisons: list[FieldComparison],
    sample_id: str = "",
    ground_truth_available: bool = False,
) -> list[DisagreementCase]:
    """
    Identify and rank disagreements across all field comparisons.
    
    Disagreement types:
    1. "rules_vs_model" – easyocr and layoutlm disagree on this field
    2. "inconsistent_fusion" – hybrid chose a mode but differs from both
    3. "missing_in_mode" – field present in 2+ modes but missing in 1
    4. "all_empty" – all modes empty for this field
    5. "wrong_vs_truth" – mode(s) wrong compared to ground truth
    
    Severity (high→low):
    - Critical fields empty (vendor, total, date)
    - Rules and model strongly disagree
    - Item extraction divergence
    - One mode missing when others have it
    """
    cases: list[DisagreementCase] = []
    
    critical_fields = {
        "vendor.name",
        "invoice.date",
        "totals.total",
        "payment.amount_paid",
    }
    
    for comp in field_comparisons:
        field_name = comp.field_name
        field_type = comp.field_type
        
        # Skip if all modes agree
        if comp.easyocr_vs_layoutlm and comp.layoutlm_vs_hybrid and comp.easyocr_vs_hybrid:
            continue
        
        # Determine disagreement type and severity
        case = DisagreementCase(
            sample_id=sample_id,
            field_name=field_name,
            field_type=field_type,
            easyocr_value=comp.easyocr_rules_raw,
            layoutlm_value=comp.layoutlm_only_raw,
            hybrid_value=comp.hybrid_raw,
            ground_truth_value=comp.ground_truth_raw,
        )
        
        # Check if all modes are empty
        if not comp.easyocr_present and not comp.layoutlm_present and not comp.hybrid_present:
            case.disagreement_type = "all_empty"
            if field_name in critical_fields:
                case.severity = "high"
                case.severity_score = 1.0
                case.explanation = f"Critical field '{field_name}' missing in all modes"
            else:
                case.severity = "low"
                case.severity_score = 0.1
                case.explanation = f"Field '{field_name}' empty in all modes"
        
        # Check for missing in single mode
        elif (comp.easyocr_present + comp.layoutlm_present + comp.hybrid_present) == 2:
            case.disagreement_type = "missing_in_mode"
            missing_mode = ""
            if not comp.easyocr_present:
                missing_mode = "easyocr_rules"
            elif not comp.layoutlm_present:
                missing_mode = "layoutlm_only"
            else:
                missing_mode = "hybrid"
            
            if field_name in critical_fields:
                case.severity = "high"
                case.severity_score = 0.9
            else:
                case.severity = "medium"
                case.severity_score = 0.5
            
            case.explanation = f"Field missing in {missing_mode}"
        
        # Check for strong disagreement between rules and model
        elif not comp.easyocr_vs_layoutlm and comp.easyocr_present and comp.layoutlm_present:
            case.disagreement_type = "rules_vs_model"
            
            # Determine severity
            if field_name in critical_fields:
                case.severity = "high"
                case.severity_score = 0.95
            else:
                case.severity = "medium"
                case.severity_score = 0.6
            
            case.explanation = (
                f"Rules say '{comp.easyocr_rules_raw}', "
                f"Model says '{comp.layoutlm_only_raw}'"
            )
            
            # Check what hybrid chose
            if comp.hybrid_vs_layoutlm:
                case.hybrid_chose = "layoutlm_only"
            elif comp.easyocr_vs_hybrid:
                case.hybrid_chose = "easyocr_rules"
            else:
                case.hybrid_chose = "default"
        
        # Inconsistent fusion (hybrid diverges from both)
        elif not comp.easyocr_vs_hybrid and not comp.layoutlm_vs_hybrid:
            if comp.easyocr_present or comp.layoutlm_present:
                case.disagreement_type = "inconsistent_fusion"
                case.severity = "medium"
                case.severity_score = 0.7
                case.explanation = (
                    f"Hybrid value '{comp.hybrid_raw}' differs from "
                    f"rules '{comp.easyocr_rules_raw}' and model '{comp.layoutlm_only_raw}'"
                )
        
        # Wrong vs ground truth
        if ground_truth_available and comp.truth_present:
            if field_type == "amount":
                rules_wrong = not comp.easyocr_vs_truth if comp.easyocr_vs_truth is not None else False
                model_wrong = not comp.layoutlm_vs_truth if comp.layoutlm_vs_truth is not None else False
                hybrid_wrong = not comp.hybrid_vs_truth if comp.hybrid_vs_truth is not None else False
            else:
                rules_wrong = not comp.easyocr_vs_truth if comp.easyocr_vs_truth is not None else False
                model_wrong = not comp.layoutlm_vs_truth if comp.layoutlm_vs_truth is not None else False
                hybrid_wrong = not comp.hybrid_vs_truth if comp.hybrid_vs_truth is not None else False
            
            if rules_wrong or model_wrong or hybrid_wrong:
                case.disagreement_type = "wrong_vs_truth"
                case.severity = "high"
                case.severity_score = 0.85
                
                wrong_modes = []
                if rules_wrong:
                    wrong_modes.append("easyocr_rules")
                if model_wrong:
                    wrong_modes.append("layoutlm_only")
                if hybrid_wrong:
                    wrong_modes.append("hybrid")
                
                case.explanation = (
                    f"Ground truth: '{comp.ground_truth_raw}'. "
                    f"Wrong in: {', '.join(wrong_modes) if wrong_modes else 'all'}"
                )
        
        if case.disagreement_type:
            cases.append(case)
    
    # Sort by severity score (high first)
    cases.sort(key=lambda c: c.severity_score, reverse=True)
    
    return cases


@dataclass
class SampleDisagreementSummary:
    """Summary of disagreements for a single sample."""
    
    sample_id: str
    total_fields_compared: int = 0
    total_disagreements: int = 0
    critical_field_issues: int = 0  # vendor, date, total missing or wrong
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    
    # Top issues
    top_disagreements: list[DisagreementCase] = field(default_factory=list)
    
    # Overall assessment
    overall_assessment: str = ""  # "good", "fair", "poor"
    recommendation: str = ""  # Action to take
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_sample_disagreements(
    disagreements: list[DisagreementCase],
    field_comparisons: list[FieldComparison],
    sample_id: str = "",
) -> SampleDisagreementSummary:
    """
    Create summary assessment of disagreements for sample.
    
    Determines:
    - Number and severity of disagreements
    - Critical field coverage
    - Overall quality assessment
    - Recommendation for manual inspection
    """
    summary = SampleDisagreementSummary(
        sample_id=sample_id,
        total_fields_compared=len(field_comparisons),
    )
    
    summary.total_disagreements = len(disagreements)
    
    critical_fields = {"vendor.name", "invoice.date", "totals.total"}
    
    for case in disagreements:
        if case.field_name in critical_fields and case.disagreement_type in [
            "all_empty",
            "missing_in_mode",
            "wrong_vs_truth",
        ]:
            summary.critical_field_issues += 1
        
        if case.severity == "high":
            summary.high_severity_count += 1
        elif case.severity == "medium":
            summary.medium_severity_count += 1
        elif case.severity == "low":
            summary.low_severity_count += 1
    
    # Take top 5 most severe disagreements
    summary.top_disagreements = disagreements[:5]
    
    # Overall assessment
    if summary.critical_field_issues > 0:
        summary.overall_assessment = "poor"
        summary.recommendation = "Manual review required (critical fields missing/wrong)"
    elif summary.high_severity_count > 2:
        summary.overall_assessment = "fair"
        summary.recommendation = "Consider manual inspection of high-severity disagreements"
    elif summary.total_disagreements == 0:
        summary.overall_assessment = "good"
        summary.recommendation = "All modes agree; good alignment"
    else:
        summary.overall_assessment = "fair"
        summary.recommendation = "Review disagreements; may be benign"
    
    return summary
