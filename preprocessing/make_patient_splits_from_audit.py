#!/usr/bin/env python3
"""
Create BEETLE-like patient-level 5-fold splits from dataset_audit.csv.

What this does:
1. Reads dataset_audit.csv with at least:
   - case_id
   - patient_id_guess
   - has_mask
2. Assigns whole patients to folds, so no patient leaks across train/val.
3. Balances folds by number of WSIs/cases. If extra class-count columns exist,
   it also balances those.
4. Optionally writes nnU-Net v2 splits_final.json for a static patch dataset.

Important:
- This does NOT create patches.
- This does NOT reproduce histology-type stratification unless your audit CSV
  contains a histology/subtype column.
- For static patch datasets, run this after conversion/preprocessing so it can
  map patch IDs back to WSI case IDs and write splits_final.json.

Example:
python preprocessing/make_patient_splits_from_audit.py \
  --audit-csv outputs/dataset_audit.csv \
  --out-dir outputs/splits \
  --n-folds 5 \
  --seed 42

After creating patches + preprocessing:
python preprocessing/make_patient_splits_from_audit.py \
  --audit-csv outputs/dataset_audit.csv \
  --out-dir outputs/splits \
  --n-folds 5 \
  --seed 42 \
  --nnunet-raw "$nnUNet_raw" \
  --nnunet-preprocessed "$nnUNet_preprocessed" \
  --dataset-name Dataset301_BEETLE \
  --write-nnunet-splits
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["case_id", "patient_id_guess"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-csv", required=True, help="Path to dataset_audit.csv")
    p.add_argument("--out-dir", required=True, help="Output directory for fold CSV/JSON files")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-without-mask", action="store_true",
                   help="Include rows where has_mask is False. Default: exclude if has_mask exists.")
    p.add_argument("--nnunet-raw", default=None,
                   help="nnUNet_raw root. Required only with --write-nnunet-splits.")
    p.add_argument("--nnunet-preprocessed", default=None,
                   help="nnUNet_preprocessed root. Required only with --write-nnunet-splits.")
    p.add_argument("--dataset-name", default="Dataset301_BEETLE")
    p.add_argument("--write-nnunet-splits", action="store_true",
                   help="Write nnU-Net splits_final.json by mapping patch IDs to patient folds.")
    return p.parse_args()


def infer_optional_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect optional metadata/count columns if they exist.
    - class/count columns are used for balancing.
    - categorical columns are used for summaries only unless known stratification exists.
    """
    lower = {c.lower(): c for c in df.columns}

    count_cols = []
    # common names if you later extend dataset_audit with mask/class counts
    patterns = [
        r"^(n_|num_|count_|pixels_)?(label_)?[0-4]$",
        r"(invasive|non.?invasive|necrosis|other).*?(pixels|count|area)$",
        r"(pixels|count|area).*?(invasive|non.?invasive|necrosis|other)$",
        r"^class_[0-4](_pixels|_count)?$",
        r"^label_[0-4](_pixels|_count)?$",
    ]
    for c in df.columns:
        lc = c.lower()
        if any(re.search(pat, lc) for pat in patterns):
            if pd.api.types.is_numeric_dtype(df[c]):
                count_cols.append(c)

    # useful if your audit later contains this
    categorical_cols = []
    for possible in ["histology_type", "histological_type", "subtype", "center", "source", "scanner"]:
        if possible in lower:
            categorical_cols.append(lower[possible])

    return count_cols, categorical_cols


def patient_table_from_audit(df: pd.DataFrame, count_cols: List[str]) -> pd.DataFrame:
    rows = []
    for patient_id, g in df.groupby("patient_id_guess", sort=False):
        row = {
            "patient_id": patient_id,
            "n_cases": int(g["case_id"].nunique()),
            "case_ids": sorted(g["case_id"].astype(str).unique().tolist()),
        }

        if "has_mask" in g.columns:
            row["n_with_mask"] = int(g["has_mask"].sum())

        # Add class/count sums if available
        for c in count_cols:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0).sum())

        # Derive invasive presence if class 3 or named invasive count column exists
        invasive_cols = [c for c in count_cols if re.search(r"(^|_)3($|_)|invasive", c.lower())
                         and not re.search(r"non.?invasive", c.lower())]
        if invasive_cols:
            row["has_invasive"] = int(sum(row[c] for c in invasive_cols) > 0)

        rows.append(row)

    return pd.DataFrame(rows)


def greedy_assign_folds(patients: pd.DataFrame, n_folds: int, seed: int, count_cols: List[str]) -> pd.DataFrame:
    """
    Greedy multi-objective fold assignment.
    Primary objective: balance n_cases.
    If count_cols exist: also balance class/count sums.
    """
    rng = np.random.default_rng(seed)

    patients = patients.copy()
    # Add tiny random tie-breaker but keep large patients first
    patients["_rand"] = rng.random(len(patients))

    # Sort by total information: cases + normalized count columns
    if count_cols:
        tmp = patients[count_cols].fillna(0).astype(float)
        norm_counts = tmp.div(tmp.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
        patients["_sort_score"] = patients["n_cases"].astype(float) + norm_counts.sum(axis=1) * patients["n_cases"].mean()
    else:
        patients["_sort_score"] = patients["n_cases"].astype(float)

    patients = patients.sort_values(["_sort_score", "_rand"], ascending=[False, True]).reset_index(drop=True)

    fold_stats = []
    for f in range(n_folds):
        stat = {"fold": f, "n_cases": 0, "n_patients": 0}
        for c in count_cols:
            stat[c] = 0.0
        fold_stats.append(stat)

    assignments = {}

    total_cases = patients["n_cases"].sum()
    total_counts = {c: patients[c].sum() if c in patients.columns else 0.0 for c in count_cols}

    def score_fold_if_added(f: int, row: pd.Series) -> float:
        # Simulate adding patient to fold f. Lower is better.
        simulated = [dict(s) for s in fold_stats]
        simulated[f]["n_cases"] += int(row["n_cases"])
        simulated[f]["n_patients"] += 1
        for c in count_cols:
            simulated[f][c] += float(row.get(c, 0.0))

        # Case balance score
        target_cases = total_cases / n_folds
        case_loads = np.array([s["n_cases"] for s in simulated], dtype=float)
        score = float(np.std(case_loads / max(target_cases, 1.0)))

        # Class/count distribution balance if available
        for c in count_cols:
            total = total_counts.get(c, 0.0)
            if total <= 0:
                continue
            target = total / n_folds
            loads = np.array([s[c] for s in simulated], dtype=float)
            score += 0.5 * float(np.std(loads / max(target, 1.0)))

        # Patient count balance
        pat_loads = np.array([s["n_patients"] for s in simulated], dtype=float)
        score += 0.1 * float(np.std(pat_loads / max(len(patients) / n_folds, 1.0)))
        return score

    for _, row in patients.iterrows():
        best_fold = min(range(n_folds), key=lambda f: score_fold_if_added(f, row))
        assignments[row["patient_id"]] = best_fold

        fold_stats[best_fold]["n_cases"] += int(row["n_cases"])
        fold_stats[best_fold]["n_patients"] += 1
        for c in count_cols:
            fold_stats[best_fold][c] += float(row.get(c, 0.0))

    patients["fold"] = patients["patient_id"].map(assignments).astype(int)
    patients = patients.drop(columns=["_rand", "_sort_score"])

    return patients


def map_patch_to_case_id(patch_id: str, known_case_ids: List[str]) -> Optional[str]:
    """
    Map patch_id back to a WSI case_id.
    Works for patch names like:
        patient155_wsi1_1024_512
    and robustly handles case IDs with underscores by using known case_id prefixes.
    """
    # Exact match first
    if patch_id in known_case_ids:
        return patch_id

    # Fast common case: remove last two numeric coordinate tokens
    parts = patch_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        candidate = "_".join(parts[:-2])
        if candidate in known_case_ids:
            return candidate

    # Robust fallback: longest known case_id prefix
    for cid in sorted(known_case_ids, key=len, reverse=True):
        if patch_id == cid or patch_id.startswith(cid + "_"):
            return cid

    return None


def collect_patch_ids_from_nnunet_raw(nnunet_raw: Path, dataset_name: str) -> List[str]:
    labels_tr = nnunet_raw / dataset_name / "labelsTr"
    if not labels_tr.is_dir():
        raise FileNotFoundError(f"Could not find labelsTr: {labels_tr}")

    patch_ids = []
    for p in sorted(labels_tr.glob("*")):
        if p.is_file() and p.suffix.lower() in [".png", ".tif", ".tiff", ".nii", ".gz"]:
            # For .nii.gz, p.stem gives .nii. Handle simply:
            name = p.name
            if name.endswith(".nii.gz"):
                patch_ids.append(name[:-7])
            else:
                patch_ids.append(p.stem)
    if not patch_ids:
        raise RuntimeError(f"No label files found in {labels_tr}")
    return patch_ids


def write_nnunet_splits(
    audit_df: pd.DataFrame,
    patient_folds: pd.DataFrame,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    dataset_name: str,
    out_dir: Path,
) -> None:
    case_to_patient = dict(zip(audit_df["case_id"].astype(str), audit_df["patient_id_guess"].astype(str)))
    patient_to_fold = dict(zip(patient_folds["patient_id"].astype(str), patient_folds["fold"].astype(int)))
    known_case_ids = sorted(case_to_patient.keys(), key=len, reverse=True)

    patch_ids = collect_patch_ids_from_nnunet_raw(nnunet_raw, dataset_name)

    patch_records = []
    unmapped = []
    for pid in patch_ids:
        case_id = map_patch_to_case_id(pid, known_case_ids)
        if case_id is None:
            unmapped.append(pid)
            continue
        patient_id = case_to_patient[case_id]
        fold = patient_to_fold[patient_id]
        patch_records.append({"patch_id": pid, "case_id": case_id, "patient_id": patient_id, "fold": fold})

    if unmapped:
        preview = "\n".join(unmapped[:20])
        raise RuntimeError(
            f"Could not map {len(unmapped)} patch IDs back to audit case_id. First examples:\n{preview}"
        )

    patch_df = pd.DataFrame(patch_records)
    patch_df.to_csv(out_dir / "patch_folds.csv", index=False)

    splits = []
    all_patch_ids = set(patch_df["patch_id"])
    for f in sorted(patch_df["fold"].unique()):
        val = sorted(patch_df.loc[patch_df["fold"] == f, "patch_id"].tolist())
        train = sorted(list(all_patch_ids - set(val)))
        splits.append({"train": train, "val": val})

    split_path = nnunet_preprocessed / dataset_name / "splits_final.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as fp:
        json.dump(splits, fp, indent=2)

    # also write copy to out_dir
    with open(out_dir / "splits_final.json", "w") as fp:
        json.dump(splits, fp, indent=2)

    summary = patch_df.groupby("fold").agg(
        n_patches=("patch_id", "nunique"),
        n_cases=("case_id", "nunique"),
        n_patients=("patient_id", "nunique"),
    ).reset_index()
    summary.to_csv(out_dir / "patch_fold_summary.csv", index=False)

    print(f"Wrote nnU-Net split: {split_path}")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()

    audit_csv = Path(args.audit_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_csv)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in audit CSV: {missing}")

    df["case_id"] = df["case_id"].astype(str)
    df["patient_id_guess"] = df["patient_id_guess"].astype(str)

    if "has_mask" in df.columns and not args.include_without_mask:
        before = len(df)
        df = df[df["has_mask"].astype(bool)].copy()
        print(f"Filtered rows without mask: {before} -> {len(df)}")

    count_cols, categorical_cols = infer_optional_columns(df)
    print(f"Detected count/balance columns: {count_cols if count_cols else 'none'}")
    print(f"Detected metadata summary columns: {categorical_cols if categorical_cols else 'none'}")

    patients = patient_table_from_audit(df, count_cols)
    patient_folds = greedy_assign_folds(patients, args.n_folds, args.seed, count_cols)

    # Case-level folds
    case_folds = df[["case_id", "patient_id_guess"]].drop_duplicates().copy()
    case_folds = case_folds.merge(
        patient_folds[["patient_id", "fold"]],
        left_on="patient_id_guess",
        right_on="patient_id",
        how="left",
    ).drop(columns=["patient_id"])

    # Summaries
    fold_summary = case_folds.groupby("fold").agg(
        n_cases=("case_id", "nunique"),
        n_patients=("patient_id_guess", "nunique"),
    ).reset_index()

    if count_cols:
        patient_sum = patient_folds.groupby("fold")[count_cols].sum().reset_index()
        fold_summary = fold_summary.merge(patient_sum, on="fold", how="left")

    # Write outputs
    patient_folds_out = patient_folds.drop(columns=["case_ids"])
    patient_folds_out.to_csv(out_dir / "patient_folds.csv", index=False)
    case_folds.to_csv(out_dir / "case_folds.csv", index=False)
    fold_summary.to_csv(out_dir / "fold_summary.csv", index=False)

    # Human-readable JSON of case IDs by fold
    folds_json = {}
    for f in range(args.n_folds):
        folds_json[str(f)] = {
            "patients": sorted(patient_folds.loc[patient_folds["fold"] == f, "patient_id"].tolist()),
            "cases": sorted(case_folds.loc[case_folds["fold"] == f, "case_id"].tolist()),
        }
    with open(out_dir / "case_folds.json", "w") as fp:
        json.dump(folds_json, fp, indent=2)

    print("\nFold summary:")
    print(fold_summary.to_string(index=False))
    print(f"\nWrote:")
    print(f"  {out_dir / 'patient_folds.csv'}")
    print(f"  {out_dir / 'case_folds.csv'}")
    print(f"  {out_dir / 'fold_summary.csv'}")
    print(f"  {out_dir / 'case_folds.json'}")

    if args.write_nnunet_splits:
        if args.nnunet_raw is None or args.nnunet_preprocessed is None:
            raise ValueError("--nnunet-raw and --nnunet-preprocessed are required with --write-nnunet-splits")
        write_nnunet_splits(
            audit_df=df,
            patient_folds=patient_folds,
            nnunet_raw=Path(args.nnunet_raw),
            nnunet_preprocessed=Path(args.nnunet_preprocessed),
            dataset_name=args.dataset_name,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
