#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

LABELS = {
    1: "other",
    2: "non-invasive epithelium",
    3: "invasive epithelium",
    4: "necrosis",
}


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def add_matrices(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [int(x) + int(y) for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def dice_from_cm(cm: list[list[int]]) -> dict[str, float]:
    dices: dict[str, float] = {}

    for label, name in LABELS.items():
        tp = int(cm[label][label])
        fp = sum(int(cm[row][label]) for row in range(len(cm)) if row != label)
        fn = sum(int(cm[label][col]) for col in range(len(cm[label])) if col != label)

        denom = 2 * tp + fp + fn
        dices[name] = float(2 * tp / denom) if denom else float("nan")

    return dices


def micro_from_cm(cm: list[list[int]]) -> float:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in LABELS:
        tp = int(cm[label][label])
        fp = sum(int(cm[row][label]) for row in range(len(cm)) if row != label)
        fn = sum(int(cm[label][col]) for col in range(len(cm[label])) if col != label)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom = 2 * total_tp + total_fp + total_fn
    return float(2 * total_tp / denom) if denom else float("nan")


def macro_from_dices(dices: dict[str, float]) -> float:
    values = [value for value in dices.values() if not math.isnan(value)]
    return safe_mean(values)


def stage_config(stage: str) -> tuple[Path, str]:
    root = Path(
        "/vol/csedu-nobackup/course/IMC037_aimi/group14/"
        "nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"
    )

    if stage == "base":
        return (
            root
            / "nnUNetTrainer_CutMixStainEMA__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d",
            "cutmixema1000_best_mirror_visual",
        )

    if stage == "context":
        return (
            root
            / "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d_context1024",
            "cutmixema1000_context1024_ft100_best_mirror_visual",
        )

    raise ValueError("stage must be base or context")


def load_stage(stage: str) -> dict[str, Any]:
    model_dir, tag = stage_config(stage)

    fold_results: list[dict[str, Any]] = []
    pooled_cm = [[0 for _ in range(5)] for _ in range(5)]

    for fold in range(5):
        result_path = (
            model_dir
            / f"fold_{fold}"
            / f"fold{fold}_{tag}_full_validation_dice_tiffslide_hybrid_visual_cm.json"
        )

        if not result_path.is_file():
            raise FileNotFoundError(f"Missing evaluation JSON: {result_path}")

        with result_path.open() as f:
            result = json.load(f)

        cm = result["confusion_matrix_rows_gt_cols_pred"]
        pooled_cm = add_matrices(pooled_cm, cm)

        fold_dices = {
            str(name): float(value)
            for name, value in result["class_dices"].items()
        }

        fold_results.append(
            {
                "fold": fold,
                "json": str(result_path),
                "checkpoint": str(model_dir / f"fold_{fold}" / "checkpoint_best.pth"),
                "class_dices": fold_dices,
                "macro_mean_dice": float(result["macro_mean_dice"]),
                "micro_overall_dice": float(result["micro_overall_dice"]),
                "processed_annotated_tiles": int(result["processed_annotated_tiles"]),
            }
        )

    per_class_stats: dict[str, dict[str, float]] = {}

    for name in LABELS.values():
        values = [fold["class_dices"][name] for fold in fold_results]

        per_class_stats[name] = {
            "mean": safe_mean(values),
            "std": safe_std(values),
            "values": values,
        }

    pooled_class_dices = dice_from_cm(pooled_cm)
    pooled_macro = macro_from_dices(pooled_class_dices)
    pooled_micro = micro_from_cm(pooled_cm)

    fold_macro_values = [fold["macro_mean_dice"] for fold in fold_results]
    fold_micro_values = [fold["micro_overall_dice"] for fold in fold_results]

    summary = {
        "stage": stage,
        "model_dir": str(model_dir),
        "checkpoint_tag": tag,
        "fold_results": fold_results,
        "fold_mean_std": {
            "class_dices": per_class_stats,
            "macro_mean_dice": {
                "mean": safe_mean(fold_macro_values),
                "std": safe_std(fold_macro_values),
                "values": fold_macro_values,
            },
            "micro_overall_dice": {
                "mean": safe_mean(fold_micro_values),
                "std": safe_std(fold_micro_values),
                "values": fold_micro_values,
            },
        },
        "pooled_across_all_validation_pixels": {
            "class_dices": pooled_class_dices,
            "macro_mean_dice": pooled_macro,
            "micro_overall_dice": pooled_micro,
            "confusion_matrix_rows_gt_cols_prediction": pooled_cm,
        },
    }

    return summary


def save_stage(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stage = summary["stage"]

    json_path = output_dir / f"cv_summary_{stage}.json"
    csv_path = output_dir / f"cv_summary_{stage}.csv"
    cm_path = output_dir / f"cv_pooled_confusion_matrix_{stage}.csv"
    manifest_path = output_dir / f"ensemble_manifest_{stage}.txt"

    with json_path.open("w") as f:
        json.dump(summary, f, indent=4)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "fold",
                "other",
                "non_invasive_epithelium",
                "invasive_epithelium",
                "necrosis",
                "macro_mean_dice",
                "micro_overall_dice",
            ]
        )

        for fold in summary["fold_results"]:
            writer.writerow(
                [
                    fold["fold"],
                    fold["class_dices"]["other"],
                    fold["class_dices"]["non-invasive epithelium"],
                    fold["class_dices"]["invasive epithelium"],
                    fold["class_dices"]["necrosis"],
                    fold["macro_mean_dice"],
                    fold["micro_overall_dice"],
                ]
            )

        stats = summary["fold_mean_std"]

        writer.writerow(
            [
                "mean",
                stats["class_dices"]["other"]["mean"],
                stats["class_dices"]["non-invasive epithelium"]["mean"],
                stats["class_dices"]["invasive epithelium"]["mean"],
                stats["class_dices"]["necrosis"]["mean"],
                stats["macro_mean_dice"]["mean"],
                stats["micro_overall_dice"]["mean"],
            ]
        )

        writer.writerow(
            [
                "std",
                stats["class_dices"]["other"]["std"],
                stats["class_dices"]["non-invasive epithelium"]["std"],
                stats["class_dices"]["invasive epithelium"]["std"],
                stats["class_dices"]["necrosis"]["std"],
                stats["macro_mean_dice"]["std"],
                stats["micro_overall_dice"]["std"],
            ]
        )

        pooled = summary["pooled_across_all_validation_pixels"]

        writer.writerow(
            [
                "pooled_pixels",
                pooled["class_dices"]["other"],
                pooled["class_dices"]["non-invasive epithelium"],
                pooled["class_dices"]["invasive epithelium"],
                pooled["class_dices"]["necrosis"],
                pooled["macro_mean_dice"],
                pooled["micro_overall_dice"],
            ]
        )

    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(
            summary["pooled_across_all_validation_pixels"][
                "confusion_matrix_rows_gt_cols_prediction"
            ]
        )

    with manifest_path.open("w") as f:
        for fold in summary["fold_results"]:
            f.write(f"fold_{fold['fold']}={fold['checkpoint']}\n")

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", cm_path)
    print("Saved:", manifest_path)


def save_comparison(
    base: dict[str, Any],
    context: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pooled = base["pooled_across_all_validation_pixels"]
    context_pooled = context["pooled_across_all_validation_pixels"]

    comparison = {
        "base": base_pooled,
        "context": context_pooled,
        "context_minus_base": {
            "class_dices": {
                name: (
                    context_pooled["class_dices"][name]
                    - base_pooled["class_dices"][name]
                )
                for name in LABELS.values()
            },
            "macro_mean_dice": (
                context_pooled["macro_mean_dice"]
                - base_pooled["macro_mean_dice"]
            ),
            "micro_overall_dice": (
                context_pooled["micro_overall_dice"]
                - base_pooled["micro_overall_dice"]
            ),
        },
    }

    json_path = output_dir / "cv_comparison_context_minus_base.json"
    csv_path = output_dir / "cv_comparison_context_minus_base.csv"

    with json_path.open("w") as f:
        json.dump(comparison, f, indent=4)

    delta = comparison["context_minus_base"]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "base",
                "context",
                "context_minus_base",
            ]
        )

        for name in LABELS.values():
            writer.writerow(
                [
                    f"dice_{name}",
                    base_pooled["class_dices"][name],
                    context_pooled["class_dices"][name],
                    delta["class_dices"][name],
                ]
            )

        writer.writerow(
            [
                "macro_mean_dice",
                base_pooled["macro_mean_dice"],
                context_pooled["macro_mean_dice"],
                delta["macro_mean_dice"],
            ]
        )

        writer.writerow(
            [
                "micro_overall_dice",
                base_pooled["micro_overall_dice"],
                context_pooled["micro_overall_dice"],
                delta["micro_overall_dice"],
            ]
        )

    print("Saved:", json_path)
    print("Saved:", csv_path)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "context", "both"}:
        raise SystemExit(
            "Usage: python aggregate_cv_results.py <base|context|both>"
        )

    requested = sys.argv[1]

    output_dir = Path(
        "/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/"
        "cv_summaries/cutmixema1000_context1024"
    )

    if requested == "base":
        base = load_stage("base")
        save_stage(base, output_dir)
        return

    if requested == "context":
        context = load_stage("context")
        save_stage(context, output_dir)
        return

    base = load_stage("base")
    context = load_stage("context")

    save_stage(base, output_dir)
    save_stage(context, output_dir)
    save_comparison(base, context, output_dir)


if __name__ == "__main__":
    main()
