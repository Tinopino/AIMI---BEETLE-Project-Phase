#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_COLS = [
    "contains_other",
    "contains_non_invasive",
    "contains_invasive",
    "contains_necrosis",
]

COUNT_COLS = [
    "n_label_1_other",
    "n_label_2_non_invasive_epithelium",
    "n_label_3_invasive_epithelium",
    "n_label_4_necrosis",
]


def patient_table(df):
    rows = []

    for patient_id, g in df.groupby("patient_id_guess"):
        row = {
            "patient_id": patient_id,
            "n_cases": len(g),
            "case_ids": sorted(g["case_id"].tolist()),
        }

        for c in CLASS_COLS:
            row[c] = bool(g[c].fillna(False).any())

        for c in COUNT_COLS:
            row[c] = int(g[c].fillna(0).sum())

        row["annotated_pixels"] = int(
            g[COUNT_COLS].fillna(0).sum(axis=1).sum()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def score_fold(fold_df):
    score = {}
    for c in CLASS_COLS:
        score[c] = int(fold_df[c].sum())
    for c in COUNT_COLS:
        score[c] = int(fold_df[c].sum())
    score["n_patients"] = len(fold_df)
    score["n_cases"] = int(fold_df["n_cases"].sum())
    return score


def make_greedy_folds(pt, n_folds, seed):
    rng = np.random.default_rng(seed)

    # Sort difficult/rare/high-annotation patients first.
    pt = pt.copy()
    pt["_difficulty"] = (
        pt["contains_necrosis"].astype(int) * 1000
        + pt["contains_invasive"].astype(int) * 100
        + pt["contains_non_invasive"].astype(int) * 10
        + np.log1p(pt["annotated_pixels"])
    )
    pt = pt.sample(frac=1.0, random_state=seed)
    pt = pt.sort_values("_difficulty", ascending=False)

    folds = [[] for _ in range(n_folds)]

    global_counts = pt[COUNT_COLS].sum().replace(0, 1)

    for _, row in pt.iterrows():
        best_fold = None
        best_score = None

        for i in range(n_folds):
            candidate = folds[i] + [row]
            cand_df = pd.DataFrame(candidate)

            # Balance number of patients/cases.
            n_patients = len(cand_df)
            n_cases = cand_df["n_cases"].sum()

            # Balance class pixel fractions.
            class_balance = 0
            for c in COUNT_COLS:
                target = pt[c].sum() / n_folds
                current = cand_df[c].sum()
                if target > 0:
                    class_balance += abs(current - target) / target

            # Balance class presence.
            presence_balance = 0
            for c in CLASS_COLS:
                target = pt[c].sum() / n_folds
                current = cand_df[c].sum()
                if target > 0:
                    presence_balance += abs(current - target) / target

            target_patients = len(pt) / n_folds
            target_cases = pt["n_cases"].sum() / n_folds

            total_score = (
                2.0 * class_balance
                + 1.0 * presence_balance
                + 0.5 * abs(n_patients - target_patients) / target_patients
                + 0.5 * abs(n_cases - target_cases) / target_cases
            )

            if best_score is None or total_score < best_score:
                best_score = total_score
                best_fold = i

        folds[best_fold].append(row)

    fold_dfs = [pd.DataFrame(f).drop(columns=["_difficulty"], errors="ignore") for f in folds]
    return fold_dfs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-csv", type=Path, default=Path("outputs/dataset_audit.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/splits"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.audit_csv)

    # Keep only successfully audited rows.
    df["error"] = df["error"].fillna("")
    df = df[df["error"] == ""].copy()

    required = ["case_id", "patient_id_guess"] + CLASS_COLS + COUNT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in audit CSV: {missing}")

    pt = patient_table(df)
    folds = make_greedy_folds(pt, args.n_folds, args.seed)

    # Save patient-level fold assignment.
    patient_rows = []
    for fold_idx, fold_df in enumerate(folds):
        for _, row in fold_df.iterrows():
            patient_rows.append({
                "fold": fold_idx,
                "patient_id": row["patient_id"],
                "n_cases": row["n_cases"],
                "case_ids": ";".join(row["case_ids"]),
                **{c: row[c] for c in CLASS_COLS},
                **{c: row[c] for c in COUNT_COLS},
            })

    patient_split = pd.DataFrame(patient_rows)
    patient_split.to_csv(args.out_dir / "patient_split.csv", index=False)

    # Save case-level fold assignment.
    case_rows = []
    for _, row in patient_split.iterrows():
        for case_id in row["case_ids"].split(";"):
            case_rows.append({
                "fold": row["fold"],
                "patient_id": row["patient_id"],
                "case_id": case_id,
            })

    case_split = pd.DataFrame(case_rows)
    case_split.to_csv(args.out_dir / "case_split.csv", index=False)

    # Create nnU-Net style splits_final.json.
    all_cases = sorted(df["case_id"].tolist())
    splits = []

    for fold_idx in range(args.n_folds):
        val_cases = sorted(case_split[case_split["fold"] == fold_idx]["case_id"].tolist())
        train_cases = sorted([c for c in all_cases if c not in set(val_cases)])

        splits.append({
            "train": train_cases,
            "val": val_cases,
        })

    with open(args.out_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Save summary.
    summary_rows = []
    for fold_idx, fold_df in enumerate(folds):
        s = score_fold(fold_df)
        s["fold"] = fold_idx
        summary_rows.append(s)

    summary = pd.DataFrame(summary_rows)
    summary = summary[["fold", "n_patients", "n_cases"] + CLASS_COLS + COUNT_COLS]
    summary.to_csv(args.out_dir / "split_summary.csv", index=False)

    print("Saved:")
    print(args.out_dir / "patient_split.csv")
    print(args.out_dir / "case_split.csv")
    print(args.out_dir / "split_summary.csv")
    print(args.out_dir / "splits_final.json")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
