import argparse
import json
from pathlib import Path

import pandas as pd


def make_entry(row, data_root: Path):
    wsi_rel = str(row["wsi_path"])
    wsa_rel = str(row["annotation_json_path"])

    wsi_path = data_root / wsi_rel
    wsa_path = data_root / wsa_rel

    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing WSI: {wsi_path}")
    if not wsa_path.exists():
        raise FileNotFoundError(f"Missing annotation JSON: {wsa_path}")

    return {
        "wsi": {"path": str(wsi_path)},
        "wsa": {"path": str(wsa_path)},
    }


def fold_number(fold_name: str) -> str:
    return str(fold_name).replace("fold", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-overview", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--nnunet-preprocessed", required=True)
    parser.add_argument("--dataset-name", default="Dataset301_BEETLE")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.nnunet_preprocessed) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_overview)

    required = {
        "patient_id",
        "name",
        "wsi_path",
        "annotation_json_path",
        "split",
        "validation_fold",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Use development set only.
    df = df[df["split"].astype(str).str.lower() == "development"].copy()
    before = len(df)

    # For pathology training, every row needs a WSI and JSON annotation.
    missing_wsi = df["wsi_path"].isna().sum()
    missing_json = df["annotation_json_path"].isna().sum()
    missing_fold = df["validation_fold"].isna().sum()

    print(f"Development rows before filtering: {before}")
    print(f"Missing wsi_path: {missing_wsi}")
    print(f"Missing annotation_json_path: {missing_json}")
    print(f"Missing validation_fold: {missing_fold}")

    df = df[df["validation_fold"].notna()].copy()
    df = df[df["wsi_path"].notna()].copy()
    df = df[df["annotation_json_path"].notna()].copy()

    # Convert to string after removing NaN.
    df["wsi_path"] = df["wsi_path"].astype(str)
    df["annotation_json_path"] = df["annotation_json_path"].astype(str)

    # Verify files exist.
    keep_rows = []
    skipped = []

    for _, row in df.iterrows():
        wsi_path = data_root / row["wsi_path"]
        json_path = data_root / row["annotation_json_path"]

        if wsi_path.exists() and json_path.exists():
            keep_rows.append(row)
        else:
            skipped.append((row["name"], str(wsi_path), str(json_path)))

    df = pd.DataFrame(keep_rows)

    print(f"Rows after filtering existing WSI+JSON: {len(df)}")
    print(f"Skipped because files missing on disk: {len(skipped)}")

    if skipped:
        print("First skipped examples:")
        for name, wsi, js in skipped[:20]:
            print(f"  {name}: WSI exists={Path(wsi).exists()} JSON exists={Path(js).exists()}")
            print(f"    WSI:  {wsi}")
            print(f"    JSON: {js}")

    if df.empty:
        raise RuntimeError("No usable development rows left after filtering.")

    # files.json: all usable development cases
    files_json = {
        "training": [make_entry(row, data_root) for _, row in df.iterrows()]
    }

    with open(out_dir / "files.json", "w") as f:
        json.dump(files_json, f, indent=4)

    # splits.json: use existing validation_fold column
    folds = sorted(
        df["validation_fold"].astype(str).unique(),
        key=lambda x: int(fold_number(x)),
    )

    splits_json = {}

    for fold in folds:
        fold_id = fold_number(fold)

        train_df = df[df["validation_fold"].astype(str) != fold]
        val_df = df[df["validation_fold"].astype(str) == fold]

        splits_json[fold_id] = {
            "training": [make_entry(row, data_root) for _, row in train_df.iterrows()],
            "validation": [make_entry(row, data_root) for _, row in val_df.iterrows()],
        }

        print(
            f"fold {fold_id}: "
            f"train WSIs={len(train_df)}, val WSIs={len(val_df)}, "
            f"train patients={train_df['patient_id'].nunique()}, "
            f"val patients={val_df['patient_id'].nunique()}"
        )

    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits_json, f, indent=4)

    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B",
        },
        "labels": {
            "unannotated": 0,
            "other": 1,
            "non-invasive epithelium": 2,
            "invasive epithelium": 3,
            "necrosis": 4,
        },
        "label_sample_weights": {
            "other": 0.25,
            "non-invasive epithelium": 0.25,
            "invasive epithelium": 0.25,
            "necrosis": 0.25,
        },
    }

    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    used_csv = out_dir / "used_cases.csv"
    df.to_csv(used_csv, index=False)

    print("\nWrote:")
    print(out_dir / "files.json")
    print(out_dir / "splits.json")
    print(out_dir / "dataset.json")
    print(used_csv)


if __name__ == "__main__":
    main()
