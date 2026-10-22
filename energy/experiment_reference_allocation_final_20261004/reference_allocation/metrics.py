from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd


def _finite_1d(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def absolute_metrics(truth: Iterable[float], prediction: Iterable[float]) -> dict[str, float | int]:
    """Metrics in the input unit; callers perform eV↔meV conversion explicitly."""
    truth_array = _finite_1d(truth, "truth")
    prediction_array = _finite_1d(prediction, "prediction")
    if truth_array.shape != prediction_array.shape:
        raise ValueError("truth and prediction lengths differ")
    error = prediction_array - truth_array
    absolute = np.abs(error)
    return {
        "n": int(error.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
    }


def build_pairs(frame: pd.DataFrame, *, id_col: str = "id", group_col: str = "composition_key") -> pd.DataFrame:
    """Build one deterministic unordered pair list from complete exact-composition groups."""
    required = {id_col, group_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[id_col].duplicated().any():
        raise ValueError("duplicate structure IDs")
    rows: list[dict[str, str]] = []
    stable = frame[[id_col, group_col]].astype(str).sort_values([group_col, id_col])
    for group, block in stable.groupby(group_col, sort=True):
        ids = block[id_col].tolist()
        if len(ids) < 2:
            continue
        for left, right in combinations(ids, 2):
            rows.append({"group_id": group, "id_i": left, "id_j": right})
    return pd.DataFrame(rows, columns=["group_id", "id_i", "id_j"])


def _pair_values(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    value_col: str,
    *,
    id_col: str = "id",
) -> tuple[np.ndarray, np.ndarray]:
    if pairs.empty:
        raise ValueError("same-composition pair list is empty")
    lookup = frame.set_index(id_col)[value_col]
    missing = (set(pairs.id_i) | set(pairs.id_j)).difference(lookup.index)
    if missing:
        raise ValueError(f"pair list refers to {len(missing)} missing IDs")
    return (lookup.loc[pairs.id_i].to_numpy(np.float64), lookup.loc[pairs.id_j].to_numpy(np.float64))


def same_composition_metrics(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    truth_col: str = "truth",
    prediction_col: str = "prediction",
    id_col: str = "id",
    group_col: str = "composition_key",
    near_degenerate_threshold: float = 0.010,
    major_misselection_threshold: float = 0.010,
) -> dict[str, float | int]:
    """Evaluate exact-composition relations on a caller-frozen pair list.

    Pair accuracy excludes exact target ties. Near-degenerate accuracy includes
    nonzero target differences with magnitude at most the fixed threshold.
    Top-1 credits any prediction whose selected structure is tied for the true
    minimum within numerical tolerance.
    """
    required = {id_col, group_col, truth_col, prediction_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[id_col].duplicated().any():
        raise ValueError("duplicate structure IDs")
    block = frame[frame[id_col].isin(set(pairs.id_i) | set(pairs.id_j))].copy()
    if block.empty:
        raise ValueError("no rows are represented in pair list")
    for column in (truth_col, prediction_col):
        block[column] = _finite_1d(block[column], column)

    errors = block[prediction_col].to_numpy() - block[truth_col].to_numpy()
    centered = pd.Series(errors, index=block.index) - pd.Series(errors, index=block.index).groupby(
        block[group_col].to_numpy()
    ).transform("mean")

    truth_i, truth_j = _pair_values(block, pairs, truth_col, id_col=id_col)
    pred_i, pred_j = _pair_values(block, pairs, prediction_col, id_col=id_col)
    delta_truth = truth_i - truth_j
    delta_prediction = pred_i - pred_j
    non_tie = ~np.isclose(delta_truth, 0.0, atol=1e-12, rtol=0.0)
    near = non_tie & (np.abs(delta_truth) <= near_degenerate_threshold)
    pair_correct = np.sign(delta_truth[non_tie]) == np.sign(delta_prediction[non_tie])

    top1: list[float] = []
    major: list[float] = []
    represented = block[block[group_col].isin(pairs.group_id.unique())]
    for _, group in represented.groupby(group_col, sort=True):
        selected = group.loc[group[prediction_col].idxmin(), truth_col]
        minimum = float(group[truth_col].min())
        cost = float(selected - minimum)
        top1.append(float(cost <= 1e-12))
        major.append(float(cost > major_misselection_threshold))

    return {
        "n_structures": len(represented),
        "n_groups": int(represented[group_col].nunique()),
        "n_pairs": len(pairs),
        "ec_mae": float(np.abs(centered).mean()),
        "pair_delta_mae": float(np.mean(np.abs(delta_prediction - delta_truth))),
        "pair_accuracy": float(pair_correct.mean()) if pair_correct.size else float("nan"),
        "near_degenerate_n": int(near.sum()),
        "near_degenerate_pair_accuracy": float(
            np.mean(np.sign(delta_truth[near]) == np.sign(delta_prediction[near]))
        ) if near.any() else float("nan"),
        "top1_accuracy": float(np.mean(top1)),
        "major_misselection_rate": float(np.mean(major)),
    }
