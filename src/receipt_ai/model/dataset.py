from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import default_data_collator

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.alignment import build_weak_bio_labels
from src.receipt_ai.model.labels import LABEL2ID, sanitize_labels
from src.receipt_ai.model.preprocessing import prepare_training_feature


@dataclass(slots=True)
class DatasetBuildResult:
    train_dataset: Dataset
    val_dataset: Dataset
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class LayoutLMv3DatasetBuilder:
    """Training-ready dataset builder backed by the real SROIE dataset in this repo."""

    def __init__(self, config: ReceiptAIConfig | None = None) -> None:
        self.config = config or ReceiptAIConfig.from_env()
        self.loader = SROIEDatasetLoader(self.config.paths.data_root)

    def build_train_val_datasets(
        self,
        processor: Any,
        *,
        train_split: str = "train",
        val_split: str = "val",
        val_ratio: float = 0.1,
        seed: int = 42,
        limit_train: int | None = None,
        limit_val: int | None = None,
        strict: bool = False,
        max_length: int = 512,
        drop_noisy_samples: bool = True,
    ) -> DatasetBuildResult:
        train_records, train_diag = self.build_records(
            train_split,
            val_ratio=val_ratio,
            seed=seed,
            limit=limit_train,
            strict=strict,
            drop_noisy_samples=drop_noisy_samples,
        )
        val_records, val_diag = self.build_records(
            val_split,
            val_ratio=val_ratio,
            seed=seed,
            limit=limit_val,
            strict=strict,
            drop_noisy_samples=False,
        )

        train_ds = Dataset.from_list(train_records)
        val_ds = Dataset.from_list(val_records)

        train_ds = train_ds.map(
            lambda ex: self._encode_record(processor, ex, max_length=max_length),
            remove_columns=train_ds.column_names,
        )
        val_ds = val_ds.map(
            lambda ex: self._encode_record(processor, ex, max_length=max_length),
            remove_columns=val_ds.column_names,
        )

        train_ds.set_format(type="torch")
        val_ds.set_format(type="torch")

        return DatasetBuildResult(
            train_dataset=train_ds,
            val_dataset=val_ds,
            train_records=train_records,
            val_records=val_records,
            diagnostics={"train": train_diag, "val": val_diag},
        )

    def build_records(
        self,
        split: str,
        *,
        val_ratio: float = 0.1,
        seed: int = 42,
        limit: int | None = None,
        strict: bool = False,
        drop_noisy_samples: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {
            "split": split,
            "samples_seen": 0,
            "samples_kept": 0,
            "samples_dropped": 0,
            "drop_reasons": {},
            "label_distribution": {},
            "token_distribution": {},
            "label_confidence": {},
            "filtering_summary": {},
        }
        count = 0
        label_counts: dict[str, int] = {}
        filtering_counts: dict[str, int] = {}
        confidence_sum = 0.0
        confidence_count = 0
        labeled_tokens = 0
        total_tokens = 0
        for sample in self.loader.iter_samples(split, val_ratio=val_ratio, seed=seed, strict=strict):
            diagnostics["samples_seen"] += 1
            weak = build_weak_bio_labels(sample)
            if drop_noisy_samples and weak.drop_training_sample:
                diagnostics["samples_dropped"] += 1
                reason = weak.drop_reason or "unspecified"
                diagnostics["drop_reasons"][reason] = int(diagnostics["drop_reasons"].get(reason, 0)) + 1
                continue
            labels = sanitize_labels(weak.labels)
            words = [token.text for token in sample.ocr_tokens]
            boxes = [token.bbox.to_layoutlm_1000(sample.image_width, sample.image_height) for token in sample.ocr_tokens]

            if not words or not boxes or len(words) != len(labels):
                continue

            label_weights = [
                float(weight if label != "O" else 1.0)
                for label, weight in zip(labels, weak.label_confidences)
            ]

            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            for key, value in weak.filtering_summary.items():
                filtering_counts[key] = filtering_counts.get(key, 0) + int(value)
            for label, weight in zip(labels, label_weights):
                total_tokens += 1
                if label != "O":
                    labeled_tokens += 1
                    confidence_sum += weight
                    confidence_count += 1

            out.append(
                {
                    "id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "words": words,
                    "boxes": boxes,
                    "labels": labels,
                    "label_weights": label_weights,
                    "line_ids": [int(token.line_id or -1) for token in sample.ocr_tokens],
                    "assumptions": weak.assumptions,
                    "target_sources": weak.target_sources,
                    "filtering_summary": weak.filtering_summary,
                    "sample_quality": weak.sample_quality,
                }
            )
            count += 1
            diagnostics["samples_kept"] += 1
            if limit is not None and count >= limit:
                break
        if not out:
            raise ValueError(f"No training records generated for split '{split}'.")
        diagnostics["label_distribution"] = dict(sorted(label_counts.items()))
        diagnostics["filtering_summary"] = dict(sorted(filtering_counts.items()))
        diagnostics["token_distribution"] = {
            "total_tokens": total_tokens,
            "labeled_tokens": labeled_tokens,
            "labeled_ratio": float(labeled_tokens / total_tokens) if total_tokens else 0.0,
            "o_ratio": float((label_counts.get("O", 0)) / total_tokens) if total_tokens else 0.0,
        }
        diagnostics["label_confidence"] = {
            "avg_non_o_weight": float(confidence_sum / confidence_count) if confidence_count else 0.0,
            "non_o_weighted_tokens": confidence_count,
        }
        return out, diagnostics

    @staticmethod
    def _encode_record(processor: Any, record: dict[str, Any], *, max_length: int) -> dict[str, Any]:
        label_ids = [LABEL2ID[label] for label in record["labels"]]
        encoded = prepare_training_feature(
            processor=processor,
            image_path=Path(record["image_path"]),
            words=record["words"],
            boxes_1000=record["boxes"],
            word_labels=label_ids,
            word_label_weights=[float(weight) for weight in record.get("label_weights", [1.0] * len(label_ids))],
            max_length=max_length,
        )
        return {key: value.numpy() if hasattr(value, "numpy") else value for key, value in encoded.items()}

    @staticmethod
    def data_collator():
        return default_data_collator
