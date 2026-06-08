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

ROOT = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/"
    "nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"
)

OUT = Path(
    "/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/"
    "cv_summaries/cutmixema1000_context1024"
)


def stage_config(stage: str) -> tuple[Path, str]:
    if stage == "base":
        return (
            ROOT
            / "nnUNetTrainer_CutMixStainEMA__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d",
            "cutmixema1000_best_mirror_visual",
        )

    if stage == "context":
        return (
            ROOT
            / "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d_context1024",
            "cutmixema1000_context1024_ft100_best_mirror_visual",
        )

    raise ValueError("stage must be base or context")


def read_cm(path: Path) -> list[list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        rows = [
            [int(float(value)) for value in row]
            for row in csv.reader(f)
            if row
        ]

    if len(rows) != 5 or any(len(row) != 5 for row in rows):
        raise ValueError(f"Expected a 5x5 confusion matrix: {path}")

    return rows


def add_cm(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def dice_by_class(cm: list[list[int]]) -> dict[str, float]:
    result: dict[str, float] = {}

    for label, name in LABELS.items():
        tp = cm[label][label]
        fp = sum(cm[row][label] for row in LABELS if row != label)
        fn = sum(cm[label][col] for col in LABELS if col != label)

        denom = 2 * tp + fp + fn
        result[name] = 2 * tp / denom if denom else float("nan")

    return result


def micro_dice(cm: list[list[int]]) -> float:
    tp = sum(cm[label][label] for label in LABELS)

    fp = sum(
        cm[row][label]
        for label in LABELS
        for row in LABELS
        if row != label
    )

    fn = sum(
        cm[label][col]
        for label in LABELS
        for col in LABELS
        if col != label
    )

    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else float("nan")


def macro_dice(dices: dict[str, float]) -> float:
    values = [value for value in dices.values() if not math.isnan(value)]
    return statistics.mean(values)


def confusion_rate(cm: list[list[int]], source: int, target: int) -> float:
    denom = sum(cm[source][col] for col in LABELS)
    return cm[source][target] / denom if denom else float("nan")


def write_cm(path: Path, cm: list[list[int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm)


def summarise(stage: str) -> dict[str, Any]:
    model_dir, tag = stage_config(stage)

    folds: list[dict[str, Any]] = []
    pooled = [[0 for _ in range(5)] for _ in range(5)]

    for fold in range(5):
        cm_path = (
            model_dir
            / f"fold_{fold}"
            / f"fold{fold}_{tag}_confusion_matrix_rows_gt_cols_pred.csv"
        )

        cm = read_cm(cm_path)
        pooled = add_cm(pooled, cm)

        dices = dice_by_class(cm)

        folds.append(
            {
                "fold": fold,
                "checkpoint": str(
                    model_dir / f"fold_{fold}" / "checkpoint_best.pth"
                ),
                "confusion_matrix_csv": str(cm_path),
                "class_dices": dices,
                "macro_dice": macro_dice(dices),
                "micro_dice": micro_dice(cm),
                "non_invasive_to_invasive": confusion_rate(cm, 2, 3),
                "invasive_to_non_invasive": confusion_rate(cm, 3, 2),
            }
        )

    pooled_dices = dice_by_class(pooled)

    summary = {
        "stage": stage,
        "model_dir": str(model_dir),
        "checkpoint_tag": tag,
        "folds": folds,
        "fold_mean_std": {},
        "pooled": {
            "class_dices": pooled_dices,
            "macro_dice": macro_dice(pooled_dices),
            "micro_dice": micro_dice(pooled),
            "non_invasive_to_invasive": confusion_rate(pooled, 2, 3),
            "invasive_to_non_invasive": confusion_rate(pooled, 3, 2),
            "confusion_matrix_rows_gt_cols_prediction": pooled,
        },
    }

    metrics = [
        ("other", lambda fold: fold["class_dices"]["other"]),
        (
            "non-invasive epithelium",
            lambda fold: fold["class_dices"]["non-invasive epithelium"],
        ),
        (
            "invasive epithelium",
            lambda fold: fold["class_dices"]["invasive epithelium"],
        ),
        ("necrosis", lambda fold: fold["class_dices"]["necrosis"]),
        ("macro_dice", lambda fold: fold["macro_dice"]),
        ("micro_dice", lambda fold: fold["micro_dice"]),
        (
            "non_invasive_to_invasive",
            lambda fold: fold["non_invasive_to_invasive"],
        ),
        (
            "invasive_to_non_invasive",
            lambda fold: fold["invasive_to_non_invasive"],
        ),
    ]

    for name, getter in metrics:
        values = [getter(fold) for fold in folds]

        summary["fold_mean_std"][name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "values": values,
        }

    return summary


def save_stage(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    stage = summary["stage"]

    json_path = OUT / f"cv_summary_{stage}.json"
    csv_path = OUT / f"cv_summary_{stage}.csv"
    cm_path = OUT / f"cv_pooled_confusion_matrix_{stage}.csv"
    manifest_path = OUT / f"ensemble_manifest_{stage}.txt"

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
                "macro_dice",
                "micro_dice",
                "non_invasive_to_invasive",
                "invasive_to_non_invasive",
            ]
        )

        for fold in summary["folds"]:
            writer.writerow(
                [
                    fold["fold"],
                    fold["class_dices"]["other"],
                    fold["class_dices"]["non-invasive epithelium"],
                    fold["class_dices"]["invasive epithelium"],
                    fold["class_dices"]["necrosis"],
                    fold["macro_dice"],
                    fold["micro_dice"],
                    fold["non_invasive_to_invasive"],
                    fold["invasive_to_non_invasive"],
                ]
            )

        pooled = summary["pooled"]

        writer.writerow(
            [
                "pooled",
                pooled["class_dices"]["other"],
                pooled["class_dices"]["non-invasive epithelium"],
                pooled["class_dices"]["invasive epithelium"],
                pooled["class_dices"]["necrosis"],
                pooled["macro_dice"],
                pooled["micro_dice"],
                pooled["non_invasive_to_invasive"],
                pooled["invasive_to_non_invasive"],
            ]
        )

    write_cm(cm_path, summary["pooled"]["confusion_matrix_rows_gt_cols_prediction"])

    with manifest_path.open("w") as f:
        for fold in summary["folds"]:
            f.write(f"fold_{fold['fold']}={fold['checkpoint']}\n")

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", cm_path)
    print("Saved:", manifest_path)


def save_comparison(base: dict[str, Any], context: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    base_pooled = base["pooled"]
    context_pooled = context["pooled"]

    rows = []

    for name in LABELS.values():
        rows.append(
            (
                f"dice_{name}",
                base_pooled["class_dices"][name],
                context_pooled["class_dices"][name],
            )
        )

    rows.extend(
        [
            (
                "macro_dice",
                base_pooled["macro_dice"],
                context_pooled["macro_dice"],
            ),
            (
                "micro_dice",
                base_pooled["micro_dice"],
                context_pooled["micro_dice"],
            ),
            (
                "non_invasive_to_invasive",
                base_pooled["non_invasive_to_invasive"],
                context_pooled["non_invasive_to_invasive"],
            ),
            (
                "invasive_to_non_invasive",
                base_pooled["invasive_to_non_invasive"],
                context_pooled["invasive_to_non_invasive"],
            ),
        ]
    )

    comparison = {
        "base": base_pooled,
        "context": context_pooled,
        "context_minus_base": {
            name: context_value - base_value
            for name, base_value, context_value in rows
        },
    }

    json_path = OUT / "cv_comparison_context_minus_base.json"
    csv_path = OUT / "cv_comparison_context_minus_base.csv"

    with json_path.open("w") as f:
        json.dump(comparison, f, indent=4)

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

        for name, base_value, context_value in rows:
            writer.writerow(
                [
                    name,
                    base_value,
                    context_value,
                    context_value - base_value,
                ]
            )

    print("Saved:", json_path)
    print("Saved:", csv_path)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "context", "both"}:
        raise SystemExit(
            "Usage: python aggregate_cv_results_v2.py <base|context|both>"
        )

    requested = sys.argv[1]

    if requested == "base":
        base = summarise("base")
        save_stage(base)
        return

    if requested == "context":
        context = summarise("context")
        save_stage(context)
        return

    base = summarise("base")
    context = summarise("context")

    save_stage(base)
    save_stage(context)
    save_comparison(base, context)


if __name__ == "__main__":
    main()
