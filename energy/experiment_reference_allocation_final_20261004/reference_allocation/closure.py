from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def projection_metrics(target_direction: Iterable[float], update: Iterable[float]) -> dict[str, float]:
    """Project an update onto a known target direction without centering."""
    target = np.asarray(target_direction, dtype=np.float64)
    actual = np.asarray(update, dtype=np.float64)
    if target.ndim != 1 or actual.ndim != 1 or target.shape != actual.shape:
        raise ValueError("target_direction and update must be same-length vectors")
    if target.size == 0 or not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("projection inputs must be nonempty and finite")
    target_squared = float(np.dot(target, target))
    if target_squared <= 0.0:
        raise ValueError("target direction has zero norm")
    target_norm = float(np.sqrt(target_squared))
    actual_norm = float(np.linalg.norm(actual))
    beta = float(np.dot(actual, target) / target_squared)
    residual = actual - beta * target
    residual_norm = float(np.linalg.norm(residual))
    centered_target = target - target.mean()
    centered_actual = actual - actual.mean()
    centered_denominator = float(
        np.linalg.norm(centered_target) * np.linalg.norm(centered_actual)
    )
    pearson = (
        float(np.dot(centered_target, centered_actual) / centered_denominator)
        if centered_denominator > 0.0
        else float("nan")
    )
    cosine = (
        float(np.dot(target, actual) / (target_norm * actual_norm))
        if actual_norm > 0.0
        else float("nan")
    )
    explained_fraction = (
        float(1.0 - residual_norm**2 / actual_norm**2)
        if actual_norm > 0.0
        else float("nan")
    )
    return {
        "beta": beta,
        "update_norm_ratio": actual_norm / target_norm,
        "orthogonal_norm_ratio": residual_norm / target_norm,
        "pearson": pearson,
        "cosine": cosine,
        "projection_explained_fraction": explained_fraction,
        "residual_rms_ev_atom": float(np.sqrt(np.mean(residual**2))),
    }


def weighted_mae_from_clusters(
    absolute_sums: np.ndarray,
    counts: np.ndarray,
    sampled_indices: np.ndarray,
) -> float:
    numerator = float(absolute_sums[sampled_indices].sum())
    denominator = float(counts[sampled_indices].sum())
    if denominator <= 0.0:
        raise ValueError("cluster count denominator is nonpositive")
    return numerator / denominator


def weighted_rmse_from_clusters(
    squared_sums: np.ndarray,
    counts: np.ndarray,
    sampled_indices: np.ndarray,
) -> float:
    numerator = float(squared_sums[sampled_indices].sum())
    denominator = float(counts[sampled_indices].sum())
    if denominator <= 0.0:
        raise ValueError("cluster count denominator is nonpositive")
    return float(np.sqrt(numerator / denominator))
