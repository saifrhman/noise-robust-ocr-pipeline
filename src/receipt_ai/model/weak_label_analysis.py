from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.alignment import build_weak_bio_labels
from src.receipt_ai.model.labels import ENTITY_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weak-label quality for receipt token classification.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--item-name-threshold", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/weak_label_analysis")
    return parser.parse_args()


def main() -> None:
    from src.receipt_ai.config import ReceiptAIConfig
    
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    summary, suspicious_samples = analyze_split(
        loader,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=args.strict,
        max_samples=args.max_samples,
        item_name_threshold=args.item_name_threshold,
    )

    summary_path = output_dir / f"{args.split}_summary.json"
    suspicious_path = output_dir / f"{args.split}_suspicious_samples.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    suspicious_path.write_text(json.dumps(suspicious_samples, indent=2), encoding="utf-8")

    print("Weak-label analysis complete")
    print(f"Split: {args.split}")
    print(f"Samples analyzed: {summary['samples_analyzed']}")
    print(f"Pseudo-only samples: {summary['pseudo_only_samples']}")
    print(f"Suspicious samples: {len(suspicious_samples)}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved suspicious samples: {suspicious_path}")


def analyze_split(
    loader: SROIEDatasetLoader,
    *,
    split: str,
    val_ratio: float,
    seed: int,
    strict: bool,
    max_samples: int,
    item_name_threshold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_counts: Counter[str] = Counter()
    key_missing_counts: Counter[str] = Counter()
    field_source_counts: dict[str, Counter[str]] = {entity: Counter() for entity in ENTITY_NAMES}
    item_coverage: Counter[str] = Counter()
    filtering_summary: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    suspicious_samples: list[dict[str, Any]] = []

    samples_analyzed = 0
    pseudo_only_samples = 0
    dropped_candidate_samples = 0
    label_confidence_sum = 0.0
    label_confidence_count = 0

    for sample in loader.iter_samples(split, val_ratio=val_ratio, seed=seed, strict=strict):
        if max_samples > 0 and samples_analyzed >= max_samples:
            break

        weak = build_weak_bio_labels(sample)
        labels = weak.labels
        label_counts.update(labels)
        filtering_summary.update({key: int(value) for key, value in weak.filtering_summary.items()})
        if weak.drop_training_sample:
            dropped_candidate_samples += 1
            drop_reasons[weak.drop_reason or "unspecified"] += 1
        for label, weight in zip(weak.labels, weak.label_confidences):
            if label == "O":
                continue
            label_confidence_sum += float(weight)
            label_confidence_count += 1

        has_vendor = _has_entity(labels, "VENDOR_NAME")
        has_date = _has_entity(labels, "DATE")
        has_total = _has_entity(labels, "TOTAL")

        if not has_vendor:
            key_missing_counts["vendor"] += 1
        if not has_date:
            key_missing_counts["date"] += 1
        if not has_total:
            key_missing_counts["total"] += 1

        for entity, source in weak.target_sources.items():
            field_source_counts.setdefault(entity, Counter())[source] += 1

        source_values = set(weak.target_sources.values())
        if source_values and source_values == {"pseudo_rules"}:
            pseudo_only_samples += 1

        item_name_count = _entity_token_count(labels, "ITEM_NAME")
        item_price_count = _entity_token_count(labels, "ITEM_PRICE")
        item_qty_count = _entity_token_count(labels, "ITEM_QTY")

        item_coverage["items_detected"] += int(weak.item_summary.get("items_detected", 0))
        item_coverage["items_labeled"] += int(weak.item_summary.get("items_labeled", 0))
        item_coverage["items_missing_price"] += int(weak.item_summary.get("items_missing_price", 0))
        item_coverage["item_name_tokens"] += item_name_count
        item_coverage["item_price_tokens"] += item_price_count
        item_coverage["item_qty_tokens"] += item_qty_count

        reasons: list[str] = []
        if not has_total:
            reasons.append("no_total")
        if not has_vendor:
            reasons.append("no_vendor")
        if not has_date:
            reasons.append("no_date")
        if item_name_count > item_name_threshold:
            reasons.append("too_many_item_name_tags")
        if item_name_count > 0 and item_price_count == 0:
            reasons.append("item_spans_without_prices")

        if reasons:
            if weak.drop_training_sample and weak.drop_reason:
                reasons.append(f"drop_candidate:{weak.drop_reason}")
            suspicious_samples.append(
                {
                    "sample_id": sample.sample_id,
                    "reasons": reasons,
                    "target_sources": weak.target_sources,
                    "item_summary": weak.item_summary,
                    "filtering_summary": weak.filtering_summary,
                    "sample_quality": weak.sample_quality,
                    "label_counts": {
                        "ITEM_NAME": item_name_count,
                        "ITEM_QTY": item_qty_count,
                        "ITEM_PRICE": item_price_count,
                    },
                }
            )

        samples_analyzed += 1

    non_o = sum(count for label, count in label_counts.items() if label != "O")
    total_tokens = sum(label_counts.values())

    summary: dict[str, Any] = {
        "samples_analyzed": samples_analyzed,
        "pseudo_only_samples": pseudo_only_samples,
        "label_counts": dict(sorted(label_counts.items())),
        "key_missing_counts": dict(key_missing_counts),
        "field_source_counts": {entity: dict(counter) for entity, counter in field_source_counts.items() if counter},
        "item_coverage": dict(item_coverage),
        "filtering_summary": dict(sorted(filtering_summary.items())),
        "drop_candidate_samples": dropped_candidate_samples,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "avg_non_o_confidence": float(label_confidence_sum / label_confidence_count) if label_confidence_count else 0.0,
        "token_distribution": {
            "total_tokens": total_tokens,
            "o_tokens": int(label_counts.get("O", 0)),
            "non_o_tokens": int(non_o),
            "o_ratio": float(label_counts.get("O", 0) / total_tokens) if total_tokens else 0.0,
            "non_o_ratio": float(non_o / total_tokens) if total_tokens else 0.0,
        },
    }
    return summary, suspicious_samples


def _has_entity(labels: list[str], entity: str) -> bool:
    suffix = f"-{entity}"
    return any(label.endswith(suffix) for label in labels)


def _entity_token_count(labels: list[str], entity: str) -> int:
    suffix = f"-{entity}"
    return sum(1 for label in labels if label.endswith(suffix))


if __name__ == "__main__":
    main()
