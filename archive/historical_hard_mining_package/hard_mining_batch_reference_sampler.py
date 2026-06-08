"""
Confusion-aware hard-example sampler for BEETLE pathology training.

This sampler keeps the normal WholeSlideData sampling behavior for most samples,
but replaces a configurable fraction of TRAINING samples with coordinates mined
from class-2 <-> class-3 mistakes. Validation always uses the normal sampler.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from wholeslidedata.samplers.batchreferencesampler import BatchReferenceSampler


class HardMiningBatchReferenceSampler(BatchReferenceSampler):
    """
    Mix normal WholeSlideData samples with pre-mined hard coordinates.

    Expected manifest columns:
        file_key, slide, wsi_path, center_x, center_y, direction

    At least one of file_key, slide, or wsi_path must identify a slide present
    in the active WholeSlideData training dataset.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        label_sampler,
        annotation_sampler,
        point_sampler,
        manifest_path: str,
        hard_fraction: float = 0.25,
        jitter: int = 128,
        seed: int = 123,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            label_sampler=label_sampler,
            annotation_sampler=annotation_sampler,
            point_sampler=point_sampler,
        )

        self._manifest_path = str(manifest_path)
        self._hard_fraction = float(hard_fraction)
        self._jitter = int(jitter)
        self._seed = int(seed)
        self._rng = np.random.RandomState(self._seed)
        self._batch_counter = 0

        if not 0.0 <= self._hard_fraction <= 1.0:
            raise ValueError(
                f"hard_fraction must be between 0 and 1, got {self._hard_fraction}"
            )
        if self._jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {self._jitter}")

        mode_name = str(getattr(getattr(dataset, "mode", None), "name", ""))
        self._use_hard_sampling = (
            mode_name.lower() == "training" and self._hard_fraction > 0.0
        )

        self._hard_examples_by_direction: Dict[str, List[dict]] = {}

        if self._use_hard_sampling:
            aliases = self._build_reference_alias_map()
            rows = self._read_manifest(self._manifest_path)
            resolved_rows, skipped_rows = self._resolve_manifest_rows(rows, aliases)

            for row in resolved_rows:
                direction = str(row.get("direction", "unspecified"))
                self._hard_examples_by_direction.setdefault(direction, []).append(row)

            if not self._hard_examples_by_direction:
                raise RuntimeError(
                    "Hard-mining sampler could not resolve any manifest rows to "
                    "the WholeSlideData TRAINING dataset. Check file_key/slide names "
                    f"in: {self._manifest_path}"
                )

            summary = {
                direction: len(rows)
                for direction, rows in sorted(self._hard_examples_by_direction.items())
            }
            print(
                "[HARD MINING] Enabled for training: "
                f"fraction={self._hard_fraction}, jitter={self._jitter}, "
                f"resolved={sum(summary.values())}, skipped={skipped_rows}, "
                f"by_direction={summary}",
                flush=True,
            )
        else:
            print(
                f"[HARD MINING] Disabled for dataset mode={mode_name!r}; "
                "using normal WholeSlideData sampling.",
                flush=True,
            )

    @staticmethod
    def _read_manifest(path: str) -> List[dict]:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Hard-mining manifest not found: {manifest_path}")

        with manifest_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise RuntimeError(f"Hard-mining manifest is empty: {manifest_path}")

        required = {"center_x", "center_y"}
        missing = required - set(rows[0].keys())
        if missing:
            raise RuntimeError(
                f"Hard-mining manifest is missing columns {sorted(missing)}: "
                f"{manifest_path}"
            )
        return rows

    @staticmethod
    def _aliases(value: Optional[str]) -> Iterable[str]:
        if value is None:
            return []

        text = str(value).strip()
        if not text:
            return []

        path = Path(text)
        aliases = {
            text,
            text.lower(),
            path.name,
            path.name.lower(),
            path.stem,
            path.stem.lower(),
        }
        return aliases

    def _build_reference_alias_map(self) -> Dict[str, object]:
        """
        Map several robust aliases to a sample reference.

        BatchSampler only needs a reference with the correct file_key and
        annotation source; annotation_index itself is not used when extracting
        the patch around an explicit point.
        """
        alias_map: Dict[str, object] = {}

        for sample_references in self._dataset.sample_references.values():
            for reference in sample_references:
                file_key = str(reference.file_key)
                for alias in self._aliases(file_key):
                    alias_map.setdefault(alias, reference)

        return alias_map

    def _resolve_manifest_rows(
        self,
        rows: List[dict],
        alias_map: Dict[str, object],
    ) -> tuple[List[dict], int]:
        resolved: List[dict] = []
        skipped = 0

        for row in rows:
            reference = None
            candidates = [
                row.get("file_key"),
                row.get("slide"),
                row.get("wsi_path"),
            ]

            for candidate in candidates:
                for alias in self._aliases(candidate):
                    if alias in alias_map:
                        reference = alias_map[alias]
                        break
                if reference is not None:
                    break

            if reference is None:
                skipped += 1
                continue

            try:
                center_x = float(row["center_x"])
                center_y = float(row["center_y"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue

            row = dict(row)
            row["_reference"] = reference
            row["_center_x"] = center_x
            row["_center_y"] = center_y
            resolved.append(row)

        return resolved, skipped

    def _normal_reference(self) -> dict:
        label = next(self._label_sampler)
        index = next(self._annotation_sampler)(label)
        sample = self._dataset.sample_references[label][index]
        annotation = self._dataset.get_annotation_from_reference(sample)
        point = self._point_sampler.sample(annotation)
        return {"reference": sample, "point": point}

    def _hard_reference(self) -> dict:
        directions = sorted(self._hard_examples_by_direction)
        direction = directions[self._rng.randint(len(directions))]
        examples = self._hard_examples_by_direction[direction]
        row = examples[self._rng.randint(len(examples))]

        if self._jitter > 0:
            dx = int(self._rng.randint(-self._jitter, self._jitter + 1))
            dy = int(self._rng.randint(-self._jitter, self._jitter + 1))
        else:
            dx = dy = 0

        point = (row["_center_x"] + dx, row["_center_y"] + dy)
        return {"reference": row["_reference"], "point": point}

    def batch(self):
        if not self._use_hard_sampling:
            return super().batch()

        batch = []
        hard_count = 0

        for _ in range(self._batch_size):
            if self._rng.random_sample() < self._hard_fraction:
                batch.append(self._hard_reference())
                hard_count += 1
            else:
                batch.append(self._normal_reference())

        self._batch_counter += 1
        if self._batch_counter % 250 == 0:
            print(
                "[HARD MINING] Produced mixed batch; "
                f"latest_hard_samples={hard_count}/{self._batch_size}",
                flush=True,
            )

        return batch

    def reset(self):
        super().reset()
        self._rng = np.random.RandomState(self._seed)
        self._batch_counter = 0
