from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rclfa_v3.data import composition_matrix


@dataclass(frozen=True)
class CalibrationResult:
    name: str
    coefficients: dict[str, float]
    train_predictions: np.ndarray
    all_predictions: np.ndarray


def fit_ridge(design: np.ndarray, target: np.ndarray, ridge: float, *, regularize_last: bool = False) -> np.ndarray:
    lhs = design.T @ design
    reg = np.eye(lhs.shape[0], dtype=np.float64) * float(ridge)
    if not regularize_last and reg.shape[0] > 0:
        reg[-1, -1] = 0.0
    rhs = design.T @ target
    return np.linalg.solve(lhs + reg, rhs)


def fit_calibrators(
    frame: pd.DataFrame,
    *,
    energy_column: str,
    target_column: str,
    ridge: float = 1e-6,
) -> list[CalibrationResult]:
    train_mask = frame["split"].to_numpy() == "train"
    x_all = composition_matrix(frame)
    e_all = frame[energy_column].to_numpy(dtype=np.float64)
    y_all = frame[target_column].to_numpy(dtype=np.float64)

    x_train = x_all[train_mask]
    e_train = e_all[train_mask]
    y_train = y_all[train_mask]
    results: list[CalibrationResult] = []

    # Formation-specific element reference:
    # y_form ~= e_model_per_atom - sum_i x_i * mu_i - bias
    # Fit e_model_per_atom - y_form ~= sum_i x_i * mu_i + bias on train only.
    design_ref_train = np.column_stack([x_train, np.ones(len(x_train), dtype=np.float64)])
    theta_ref = fit_ridge(design_ref_train, e_train - y_train, ridge)
    design_ref_all = np.column_stack([x_all, np.ones(len(x_all), dtype=np.float64)])
    pred_ref = e_all - design_ref_all @ theta_ref
    coeff_ref = {f"mu_Z{idx}": float(value) for idx, value in enumerate(theta_ref[:-1], start=1)}
    coeff_ref["bias"] = float(theta_ref[-1])
    results.append(
        CalibrationResult(
            name="element_reference",
            coefficients=coeff_ref,
            train_predictions=pred_ref[train_mask],
            all_predictions=pred_ref,
        )
    )

    # More flexible upper-bound alignment:
    # y_form ~= alpha * e_model_per_atom + sum_i x_i * beta_i + bias.
    design_aff_train = np.column_stack([e_train, x_train, np.ones(len(x_train), dtype=np.float64)])
    theta_aff = fit_ridge(design_aff_train, y_train, ridge)
    design_aff_all = np.column_stack([e_all, x_all, np.ones(len(x_all), dtype=np.float64)])
    pred_aff = design_aff_all @ theta_aff
    coeff_aff = {"alpha_energy": float(theta_aff[0])}
    coeff_aff.update({f"beta_Z{idx}": float(value) for idx, value in enumerate(theta_aff[1:-1], start=1)})
    coeff_aff["bias"] = float(theta_aff[-1])
    results.append(
        CalibrationResult(
            name="affine_energy_composition",
            coefficients=coeff_aff,
            train_predictions=pred_aff[train_mask],
            all_predictions=pred_aff,
        )
    )

    # Composition-only baseline with exactly the same composition features.
    design_comp_train = np.column_stack([x_train, np.ones(len(x_train), dtype=np.float64)])
    theta_comp = fit_ridge(design_comp_train, y_train, ridge)
    design_comp_all = np.column_stack([x_all, np.ones(len(x_all), dtype=np.float64)])
    pred_comp = design_comp_all @ theta_comp
    coeff_comp = {f"beta_Z{idx}": float(value) for idx, value in enumerate(theta_comp[:-1], start=1)}
    coeff_comp["bias"] = float(theta_comp[-1])
    results.append(
        CalibrationResult(
            name="composition_only_linear",
            coefficients=coeff_comp,
            train_predictions=pred_comp[train_mask],
            all_predictions=pred_comp,
        )
    )
    return results


def apply_calibration_results(
    frame: pd.DataFrame,
    results: list[CalibrationResult],
    *,
    energy_column: str,
) -> dict[str, np.ndarray]:
    x_all = composition_matrix(frame)
    e_all = pd.to_numeric(frame[energy_column], errors="coerce").to_numpy(dtype=np.float64)
    ones = np.ones(len(frame), dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for result in results:
        if result.name == "element_reference":
            theta = np.asarray(
                [result.coefficients.get(f"mu_Z{idx}", 0.0) for idx in range(1, x_all.shape[1] + 1)]
                + [result.coefficients.get("bias", 0.0)],
                dtype=np.float64,
            )
            design = np.column_stack([x_all, ones])
            out[result.name] = e_all - design @ theta
            continue
        if result.name == "affine_energy_composition":
            theta = np.asarray(
                [result.coefficients.get("alpha_energy", 0.0)]
                + [result.coefficients.get(f"beta_Z{idx}", 0.0) for idx in range(1, x_all.shape[1] + 1)]
                + [result.coefficients.get("bias", 0.0)],
                dtype=np.float64,
            )
            design = np.column_stack([e_all, x_all, ones])
            out[result.name] = design @ theta
            continue
        if result.name == "composition_only_linear":
            theta = np.asarray(
                [result.coefficients.get(f"beta_Z{idx}", 0.0) for idx in range(1, x_all.shape[1] + 1)]
                + [result.coefficients.get("bias", 0.0)],
                dtype=np.float64,
            )
            design = np.column_stack([x_all, ones])
            out[result.name] = design @ theta
            continue
        raise KeyError(f"Unsupported calibration result {result.name!r}")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"n": 0}
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - float(np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae_mev_atom": float(abs_err.mean() * 1000.0),
        "rmse_mev_atom": float(np.sqrt(np.mean(err**2)) * 1000.0),
        "median_abs_mev_atom": float(np.median(abs_err) * 1000.0),
        "p90_abs_mev_atom": float(np.quantile(abs_err, 0.90) * 1000.0),
        "p95_abs_mev_atom": float(np.quantile(abs_err, 0.95) * 1000.0),
        "p99_abs_mev_atom": float(np.quantile(abs_err, 0.99) * 1000.0),
        "bias_mev_atom": float(err.mean() * 1000.0),
        "r2": r2,
    }


def slice_metrics(frame: pd.DataFrame, pred_col: str, target_col: str) -> dict[str, dict[str, float | int]]:
    y = frame[target_col].to_numpy(dtype=np.float64)
    slices: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "positive_true": y >= 0.0,
        "negative_true": y < 0.0,
        "near_zero_50mev_true": np.abs(y) <= 0.05,
        "near_zero_100mev_true": np.abs(y) <= 0.10,
    }
    if "nelements" in frame.columns:
        ne = frame["nelements"].to_numpy()
        slices.update(
            {
                "unary": ne == 1,
                "binary": ne == 2,
                "ternary": ne == 3,
                "quaternary_plus": ne >= 4,
            }
        )
    if "natoms" in frame.columns:
        natoms = frame["natoms"].to_numpy()
        slices.update({"small_cell_le4": natoms <= 4, "small_cell_le8": natoms <= 8})

    out: dict[str, dict[str, float | int]] = {}
    for name, mask in slices.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            out[name] = regression_metrics(
                frame.loc[mask, target_col].to_numpy(dtype=np.float64),
                frame.loc[mask, pred_col].to_numpy(dtype=np.float64),
            )
        else:
            out[name] = {"n": 0}
    return out
