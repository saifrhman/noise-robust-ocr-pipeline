# Mode Comparison and Error Analysis Workflow

This workflow enables comprehensive comparison of three receipt extraction modes and detailed error analysis:
- **easyocr_rules**: Rule-based extraction using EasyOCR + receipt parser
- **layoutlm_only**: Pure LayoutLMv3 model-based extraction  
- **hybrid**: Combined model + rules with fusion logic

## Quick Start

### Step 1: Generate Mode Comparison Results

Compare all three modes on the same validation samples:

```bash
# Compare on validation split (50 samples)
python scripts/compare_extraction_modes_receipt_ai.py \
  --split val \
  --max-samples 50 \
  --seed 42 \
  --output-dir outputs/comparison

# Or on test split
python scripts/compare_extraction_modes_receipt_ai.py \
  --split test \
  --max-samples 100 \
  --output-dir outputs/comparison
```

This creates `outputs/comparison/comparison_*.json` with per-sample results from all 3 modes.

### Step 2: Run Comprehensive Analysis

Analyze the comparison results to identify disagreements, errors, and insights:

```bash
python scripts/analyze_comparison_receipt_ai.py \
  --comparison-file outputs/comparison/comparison_val.json \
  --output-dir outputs/analysis
```

This generates human-review artifacts in `outputs/analysis/`.

## Output Artifacts

### Primary Review Files

- **`top_disagreements.csv`** – Spreadsheet-friendly summary of top disagreements
  - Field names, values from each mode, disagreement type, severity
  - Open in Excel/Sheets for easy filtering and sorting

- **`error_report.md`** – Human-readable error summary
  - Total errors per category (missing vendor, amount parse failure, etc.)
  - Top problematic samples ranked by error count

- **`sample_review.jsonl`** – Per-sample detailed analysis (one JSON object per line)
  - Full disagreement summary for each sample
  - Item analysis (counts, coherence)
  - Error buckets identified for this sample
  - Easy to import into analysis tools

### Detailed Analysis Files (JSON)

- **`field_comparisons.json`** – Field-by-field comparison results
  - Raw and normalized values for each mode
  - Match status (easyocr vs layoutlm, etc.)
  - Ground truth comparison (if available)

- **`disagreement_summaries.json`** – Per-sample disagreement assessment
  - Overall assessment (good/fair/poor)
  - Critical field issues count
  - High/medium/low severity counts

- **`top_disagreements.json`** – Top disagreement cases with details
  - Disagreement type (rules_vs_model, missing_in_mode, wrong_vs_truth, etc.)
  - Severity scoring for ranking

- **`error_buckets.json`** – Categorized errors for each sample
  - Errors grouped by bucket type
  - Critical errors highlighted

- **`item_analyses.json`** – Item extraction analysis (marked as heuristic)
  - Item count per mode
  - Implied total vs extracted total
  - Item coherence checks
  - Notes flagging issues

- **`error_summary.json`** – Aggregate error statistics
  - Count per error bucket type
  - Worst samples ranked by error count

## Field Comparison Details

The system compares 18 key fields across all modes:

### Vendor Info
- `vendor.name` – Text field with case/whitespace normalization
- `vendor.address` – Text field
- `vendor.registration_number` – Text field

### Invoice Info
- `invoice.bill_number`, `order_number`, `table_number` – Invoice number normalization
- `invoice.date` – Normalized to YYYY-MM-DD
- `invoice.time` – Normalized to HH:MM:SS
- `invoice.cashier` – Text field

### Totals
- `totals.subtotal`, `tax`, `total` – Amount fields with 1% numeric tolerance
- `totals.service_charge`, `rounding` – Amounts
- `totals.currency` – Text field

### Payment
- `payment.method` – Text field
- `payment.amount_paid` – Amount field

### Normalization Rules
- **Amounts**: Case-insensitive currency stripping, OCR error correction (O→0), decimal padding, numeric tolerance ±1%
- **Dates**: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY → YYYY-MM-DD
- **Text**: Lowercase, whitespace collapse, punctuation handling
- **Invoice numbers**: Whitespace removed, case-insensitive

## Disagreement Types

- **`rules_vs_model`** – easyocr_rules and layoutlm_only provide different values
- **`inconsistent_fusion`** – hybrid value differs from both rules and model
- **`missing_in_mode`** – Field present in 2 modes but missing in 1
- **`all_empty`** – All modes empty for this field
- **`wrong_vs_truth`** – Mode output conflicts with ground truth (if available)

## Error Buckets

Automatically categorized failures:
- **missing_vendor**, **missing_date**, **missing_total** – Critical fields empty in all modes
- **amount_parse_failure** – Extracted but unparseable (normalized to 0.0)
- **item_grouping_failure** – Inconsistent item count across modes
- **low_confidence_semantic** – Model confidence < 50%
- **hybrid_fallback_to_rules** – Hybrid detected fallback behavior
- **schema_reduced_warning** – Checkpoint using legacy/reduced label space
- **all_modes_agree_but_empty** – Consensus empty on critical field
- **single_mode_fails** – Field missing in only one mode

## Item Analysis (Heuristic)

Item-level comparison is explicitly marked as **heuristic** because:
- SROIE dataset has weak/partial item annotations
- Item truth is not gold standard
- Analysis includes implied total vs extracted total consistency check
- Item coherence check (all items have name + price)

## Usage Examples

### Review Top Disagreements in Spreadsheet

```bash
# Generate comparison (50 samples)
python scripts/compare_extraction_modes_receipt_ai.py \
  --split val --max-samples 50 --output-dir outputs/comparison

# Analyze
python scripts/analyze_comparison_receipt_ai.py \
  --comparison-file outputs/comparison/comparison_val.json \
  --output-dir outputs/analysis

# Open in Excel/Sheets
open outputs/analysis/top_disagreements.csv
```

### Find Samples with Missing Critical Fields

```bash
# Extract samples with critical errors from JSONL
python -c "
import json
with open('outputs/analysis/sample_review.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        if sample['disagreement_summary']['critical_field_issues'] > 0:
            print(f\"{sample['sample_id']}: {sample['disagreement_summary']['critical_field_issues']} critical issues\")
"
```

### Generate Critical Fields Only Report

```bash
# Use utilities module for focused analysis
python -c "
import json
from src.receipt_ai.evaluation.utilities import format_critical_fields_report, assess_critical_fields
from src.receipt_ai.evaluation.field_comparison import compare_fields

# Load one sample's comparison results...
# comps = compare_fields(easyocr, layoutlm, hybrid)
# print(format_critical_fields_report(comps, sample_id='X00016469670'))
"
```

## Command-Line Options

### Mode Comparison Script

```
--split {train,val,test}        Dataset split (default: val)
--max-samples N                 Max samples to process (0=all)
--val-ratio RATIO              Validation ratio for train split (default: 0.1)
--seed SEED                     Random seed for reproducibility (default: 42)
--output-dir DIR                Output directory (default: outputs/comparison)
--dataset-root DIR              SROIE2019 data root
--strict                        Fail on first error instead of collecting all
```

### Analysis Script

```
--comparison-file FILE          Path to comparison_*.json from step 1 (required)
--output-dir DIR                Output directory (default: outputs/analysis)
--critical-fields-only          Focus on vendor/date/total only
--top-disagreements N           Number of top cases to report (default: 10)
```

## Practical Workflow

### 1. Identify Problem Areas
```bash
cat outputs/analysis/error_report.md
```
Check which error types are most prevalent.

### 2. Review Top Failures
```bash
# Open CSV in spreadsheet tool
open outputs/analysis/top_disagreements.csv
```
Sort by severity or field to focus on critical issues.

### 3. Investigate Specific Samples
```bash
# Find sample in JSONL file
grep "X00016469670" outputs/analysis/sample_review.jsonl | python -m json.tool
```
See full disagreement summary, item analysis, and error buckets for this sample.

### 4. Analyze by Error Type
```bash
# Find all amount_parse_failure entries
python -c "
import json
with open('outputs/analysis/error_buckets.json') as f:
    data = json.load(f)
    for sample in data['buckets']:
        if 'amount_parse_failure' in sample['buckets']:
            print(f\"{sample['sample_id']}: {len(sample['buckets']['amount_parse_failure'])} amount errors\")
"
```

## Expected Output Structure

```
outputs/analysis/
├── field_comparisons.json              # All field-level comparisons
├── disagreement_summaries.json         # Per-sample assessments
├── top_disagreements.json              # Ranked disagreement cases
├── top_disagreements.csv               # Spreadsheet-friendly subset
├── item_analyses.json                  # Item-level heuristic analysis
├── error_buckets.json                  # Categorized errors per sample
├── error_summary.json                  # Aggregate statistics
├── error_report.md                     # Human-readable summary
├── sample_review.jsonl                 # Per-sample detailed analysis (JSONL)
└── summary_report.md                   # Quick overview and action items
```

## Implementation Details

### Modules

- **`src/receipt_ai/evaluation/field_comparison.py`**
  - Smart field normalization (amounts, dates, text)
  - Per-field comparison logic with match detection
  - Item-level analysis (counts, coherence, implied vs extracted totals)

- **`src/receipt_ai/evaluation/disagreement_analysis.py`**
  - Identifies & ranks disagreement cases
  - Severity classification (high/medium/low)
  - Per-sample summaries with recommendations

- **`src/receipt_ai/evaluation/error_bucketing.py`**
  - Automatic failure categorization
  - Summary reports (JSON & markdown)
  - Best for identifying systemic issues

- **`src/receipt_ai/evaluation/utilities.py`**
  - Optional side-by-side comparison tables
  - Critical-fields-only filtering
  - Sorting and ranking helpers

### Scripts

- **`scripts/compare_extraction_modes_receipt_ai.py`**
  - Runs all 3 modes on same samples
  - Collects results with metadata and warnings
  - Supports split selection, max-samples, seeding

- **`scripts/analyze_comparison_receipt_ai.py`**
  - Orchestrates field comparison, disagreement analysis, error bucketing
  - Generates all human-review artifacts
  - Produces CSV, JSON, Markdown, and JSONL outputs

## Assumptions & Limitations

1. **Weak Item Labels**: Item-level truth is not gold standard (SROIE partial). Item analysis is explicitly heuristic.

2. **Ground Truth Availability**: Ground truth comparison only works if provided in comparison JSON. Many fields may be unannotated.

3. **Numeric Tolerance**: Amount comparison uses 1% relative tolerance for floating-point precision issues.

4. **OCR Errors**: Amount and text normalization handles common OCR errors (O→0, 1↔l, currency symbols).

5. **Field Subset**: Comparison limited to 18 most important fields; extend via `COMPARISON_FIELDS` in field_comparison.py.

6. **Heuristic Item Analysis**: Item coherence and implied-total checks are heuristics. Item truth requires manual review.

## Extending the Comparison

To add custom fields or modify comparison logic:

1. **Add field to comparison** (field_comparison.py line 17):
   ```python
   COMPARISON_FIELDS = [
       ("vendor.name", "text"),  # existing
       ("my_custom_field", "amount"),  # add new
   ]
   ```

2. **Add normalization** (field_comparison.py):
   ```python
   elif field_type == "custom":
       return my_custom_normalize(value)
   ```

3. **Re-run analysis** with updated code.

## Stop After This Scope

This completes the experiment comparison and error-analysis workflow. No further work is planned.
