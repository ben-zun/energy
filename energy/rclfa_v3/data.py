from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


MAX_Z = 118


@dataclass(frozen=True)
class SplitRecord:
    split: str
    sample_id: str
    cif_path: Path
    target: float


def normalize_split(value: str) -> str:
    text = str(value).lower()
    if "train" in text:
        return "train"
    if "val" in text or "valid" in text:
        return "val"
    if "test" in text:
        return "test"
    return text


def target_from_row(row: dict[str, str]) -> float:
    for key in (
        "formation_energy_per_atom_ev",
        "formation_energy_peratom",
        "formation_energy_per_atom",
        "target_per_atom_ev",
        "y_true",
        "target",
    ):
        if key in row and row[key] != "":
            return float(row[key])
    raise KeyError(f"Could not find formation-energy target in columns: {sorted(row)}")


def read_split_records(dataset_root: Path, workspace_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        path = dataset_root / split / f"{split}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cif_path = Path(row["cif_path"])
                if not cif_path.is_absolute():
                    cif_path = workspace_root / cif_path
                rows.append(
                    {
                        "split": split,
                        "id": row["id"],
                        "cif_path": str(cif_path),
                        "target_from_dataset": target_from_row(row),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No split CSVs found below {dataset_root}")
    return pd.DataFrame(rows)


def composition_from_cif(cif_path: Path) -> tuple[np.ndarray, int, int, str]:
    structure = Structure.from_file(str(cif_path))
    composition = structure.composition.fractional_composition
    vec = np.zeros(MAX_Z, dtype=np.float64)
    for element, amount in composition.items():
        z = int(element.Z)
        if 1 <= z <= MAX_Z:
            vec[z - 1] = float(amount)
    total = vec.sum()
    if total <= 0:
        raise ValueError(f"Empty composition vector for {cif_path}")
    vec /= total
    formula = structure.composition.reduced_formula
    return vec, int(len(structure)), int(np.count_nonzero(vec)), formula


def build_feature_frame(
    rows: pd.DataFrame,
    *,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Attach composition fractions and simple metadata.

    The cache is keyed by sample id and CIF path. It is intentionally plain CSV so
    the calibration step can be inspected and reused across foundation models.
    """

    feature_cols = [f"x_Z{z}" for z in range(1, MAX_Z + 1)]
    required_cols = ["id", "cif_path", "natoms", "nelements", "formula", *feature_cols]
    merge_base = rows.drop(
        columns=[col for col in required_cols if col not in {"id", "cif_path"} and col in rows.columns],
        errors="ignore",
    )
    rows_to_build = rows[["id", "cif_path"]].drop_duplicates()
    cached: pd.DataFrame | None = None
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, keep_default_na=False)
        if all(col in cached.columns for col in required_cols):
            merged = merge_base.merge(cached[required_cols], on=["id", "cif_path"], how="left")
            missing_mask = merged[["natoms", "nelements", "formula", *feature_cols]].isna().any(axis=1)
            if not bool(missing_mask.any()):
                return merged
            rows_to_build = merged.loc[missing_mask, ["id", "cif_path"]].drop_duplicates()

    features: list[dict[str, object]] = []
    for idx, item in enumerate(rows_to_build.itertuples(index=False), start=1):
        cif_path = Path(str(item.cif_path))
        vec, natoms, nelements, formula = composition_from_cif(cif_path)
        record: dict[str, object] = {
            "id": item.id,
            "cif_path": str(cif_path),
            "natoms": natoms,
            "nelements": nelements,
            "formula": formula,
        }
        for z, value in enumerate(vec, start=1):
            record[f"x_Z{z}"] = value
        features.append(record)
        if idx % 5000 == 0:
            print(f"[features] built {idx}/{len(rows_to_build)} composition records", flush=True)

    feature_frame = pd.DataFrame(features)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cached is not None and all(col in cached.columns for col in required_cols):
            feature_frame = (
                pd.concat([cached[required_cols], feature_frame[required_cols]], ignore_index=True)
                .drop_duplicates(subset=["id", "cif_path"], keep="last")
            )
        feature_frame.to_csv(cache_path, index=False)
    return merge_base.merge(feature_frame[required_cols], on=["id", "cif_path"], how="left")


def composition_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"x_Z{z}" for z in range(1, MAX_Z + 1)]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing composition columns: {missing[:5]}")
    return frame[cols].to_numpy(dtype=np.float64)
