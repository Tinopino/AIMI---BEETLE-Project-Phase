import argparse
import json
from pathlib import Path

import pandas as pd


def natural_fold_key(x: str):
    x = str(x)
    if x.lower().startswith("fold"):
        try:
            return int(x.lower().replace("fold", ""))
        except ValueError:
            return x
    return x


def match_patch_to_case(patch_id: str, case_names_sorted):
    """
    Patch ids are expected to look like:
      patient155_wsi1_1024_512

    Case names look like:
      patient155_wsi1

    We match by longest case-name prefix.
    """
    for case_name in case_names_sorted:
        if patch_id == case_name or patch_id.startswith(case_name + "_"):
            return case_name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-overview",
        required=True,
        help="Path to data_overview.csv",
    )
    parser.add_argument(
        "--nnunet-raw",
        required=True,
        help="Path to nnUNet_raw",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        required=True,
        help="Path to nnUNet_preprocessed",
    )
    parser.add_argument(
        "--dataset-name",
        default="Dataset301_BEETLE",
    )
    args = parser.parse_args()

    overview = pd.read_csv(args.data_overview)

    required = {"patient_id", "name", "split", "validation_fold"}
    missing = required - set(overview.columns)
    if missing:
        raise ValueError(f"Missing required columns in data_overview.csv: {missing}")

    # Only use development cases for training/validation.
    dev = overview[overview["split"].astype(str).str.lower() == "development"].copy()
    dev = dev[dev["validation_fold"].notna()].copy()

    if dev.empty:
        raise ValueError("No development rows with validation_fold found.")

    case_to_fold = dict(zip(dev["name"].astype(str), dev["validation_fold"].astype(str)))
    case_names_sorted = sorted(case_to_fold.keys(), key=len, reverse=True)

    labels_tr = Path(args.nnunet_raw) / args.dataset_name / "labelsTr"
    if not labels_tr.exists():
        raise FileNotFoundError(f"Could not find labelsTr: {labels_tr}")

    patch_ids = sorted(p.stem for p in labels_tr.glob("*.png"))
    if not patch_ids:
        raise ValueError(f"No label PNGs found in {labels_tr}")

    patch_to_case = {}
    unmatched = []

    for patch_id in patch_ids:
        case_name = match_patch_to_case(patch_id, case_names_sorted)
        if case_name is None:
            unmatched.append(patch_id)
        else:
            patch_to_case[patch_id] = case_name

    if unmatched:
        print(f"WARNING: {len(unmatched)} patches could not be matched to data_overview cases.")
        print("First unmatched examples:")
        for u in unmatched[:20]:
            print("  ", u)

    folds = sorted(dev["validation_fold"].astype(str).unique(), key=natural_fold_key)

    splits = []
    summary_rows = []

    for fold in folds:
        val_cases = set(dev.loc[dev["validation_fold"].astype(str) == fold, "name"].astype(str))

        train = []
        val = []

        for patch_id, case_name in patch_to_case.items():
            if case_name in val_cases:
                val.append(patch_id)
            else:
                train.append(patch_id)

        train = sorted(train)
        val = sorted(val)

        splits.append({"train": train, "val": val})

        train_cases = {patch_to_case[p] for p in train}
        val_cases_present = {patch_to_case[p] for p in val}

        summary_rows.append(
            {
                "fold": fold,
                "n_train_patches": len(train),
                "n_val_patches": len(val),
                "n_train_cases": len(train_cases),
                "n_val_cases": len(val_cases_present),
            }
        )

        print(
            f"{fold}: train patches={len(train)}, val patches={len(val)}, "
            f"train cases={len(train_cases)}, val cases={len(val_cases_present)}"
        )

    out_dir = Path(args.nnunet_preprocessed) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=4)

    summary_path = out_dir / "splits_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nWrote nnU-Net split file:")
    print(splits_path)
    print(f"\nWrote summary:")
    print(summary_path)


if __name__ == "__main__":
    main()
