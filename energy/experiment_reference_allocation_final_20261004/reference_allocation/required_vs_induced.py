from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import _pair_values


def pair_corrections(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    id_col: str = "id",
    target_col: str = "target",
    pretrained_col: str = "pretrained",
    prediction_cols: tuple[str, ...] = ("direct", "aligned"),
) -> pd.DataFrame:
    output = pairs.copy()
    pre_i, pre_j = _pair_values(frame, pairs, pretrained_col, id_col=id_col)
    target_i, target_j = _pair_values(frame, pairs, target_col, id_col=id_col)
    output["delta_pre"] = pre_i - pre_j
    output["delta_target"] = target_i - target_j
    output["r_target"] = output.delta_target - output.delta_pre
    for column in prediction_cols:
        pred_i, pred_j = _pair_values(frame, pairs, column, id_col=id_col)
        output[f"delta_{column}"] = pred_i - pred_j
        output[f"r_{column}"] = output[f"delta_{column}"] - output.delta_pre
        output[f"excess_{column}"] = output[f"r_{column}"] - output.r_target
    return output


def correction_summary(target: np.ndarray, induced: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.float64)
    induced = np.asarray(induced, dtype=np.float64)
    if target.ndim != 1 or target.shape != induced.shape or target.size == 0:
        raise ValueError("target and induced corrections must be matching non-empty vectors")
    if not np.isfinite(target).all() or not np.isfinite(induced).all():
        raise ValueError("correction vectors contain NaN or infinity")
    target_norm = float(np.linalg.norm(target))
    induced_norm = float(np.linalg.norm(induced))
    if target_norm <= 0:
        raise ValueError("target correction norm is zero; use synthetic drift metrics instead")
    alpha = float(np.dot(induced, target) / np.dot(target, target))
    orthogonal = induced - alpha * target
    centered_target = target - target.mean()
    centered_induced = induced - induced.mean()
    denominator = float(np.linalg.norm(centered_target) * np.linalg.norm(centered_induced))
    pearson = float(np.dot(centered_target, centered_induced) / denominator) if denominator else float("nan")
    cosine_denominator = target_norm * induced_norm
    cosine = float(np.dot(target, induced) / cosine_denominator) if cosine_denominator else float("nan")
    return {
        "n_pairs": int(target.size),
        "excess_mae": float(np.mean(np.abs(induced - target))),
        "target_rms": float(np.sqrt(np.mean(target**2))),
        "induced_rms": float(np.sqrt(np.mean(induced**2))),
        "norm_ratio": float(induced_norm / target_norm),
        "pearson": pearson,
        "cosine": cosine,
        "projection_alpha": alpha,
        "orthogonal_norm_ratio": float(np.linalg.norm(orthogonal) / target_norm),
    }
