from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("zero_direct", "reference_direct", "reference_aligned")
SEEDS = (20260528, 20260529, 20260530)


def aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identity, block in frame.groupby(keys, sort=True):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(keys, identity_values, strict=True))
        row["seeds"] = int(block.seed.nunique())
        for metric in metrics:
            values = block[metric].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite {metric} in {row}")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--energy-root", type=Path, required=True)
    parser.add_argument("--force-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve(strict=True)
    energy_root = args.energy_root.resolve(strict=True)
    force_root = args.force_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "figures").mkdir(parents=True, exist_ok=True)

    histories: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = (
                experiment
                / "synthetic_reference_control/runs"
                / condition
                / f"seed{seed}"
                / "history.csv"
            )
            history = pd.read_csv(path, keep_default_na=False)
            if len(history) != 120 or history.epoch.tolist() != list(range(1, 121)):
                raise AssertionError(f"incomplete fixed trajectory: {path}")
            if set(history.condition) != {condition} or set(history.seed.astype(int)) != {seed}:
                raise AssertionError(f"trajectory identity mismatch: {path}")
            histories.append(history)
    epoch_metrics = pd.concat(histories, ignore_index=True)
    epoch_metrics.to_csv(output / "epoch_metrics.csv", index=False)

    metrics = pd.read_csv(energy_root / "metrics/per_seed_metrics.csv", keep_default_na=False)
    expected = {
        (condition, seed, calibration)
        for condition in CONDITIONS
        for seed in SEEDS
        for calibration in ("native", "calibrated")
    }
    observed = set(
        zip(
            metrics.condition,
            metrics.seed.astype(int),
            metrics.calibration,
            strict=True,
        )
    )
    if observed != expected or len(metrics) != len(expected):
        raise AssertionError("synthetic energy metric matrix is incomplete")
    pair_columns = [
        "condition",
        "seed",
        "calibration",
        "drift_ec_mae_mev",
        "drift_pair_delta_mae_mev",
        "drift_pair_accuracy",
        "drift_top1_accuracy",
        "drift_major_misselection_rate",
    ]
    metrics[pair_columns].to_csv(output / "pair_drift.csv", index=False)
    energy_metrics = [
        "synthetic_mae_mev",
        "synthetic_rmse_mev",
        "synthetic_bias_mev",
        "synthetic_p90_abs_error_mev",
        "synthetic_p95_abs_error_mev",
        "drift_ec_mae_mev",
        "drift_pair_delta_mae_mev",
        "drift_pair_accuracy",
        "drift_top1_accuracy",
        "drift_major_misselection_rate",
    ]
    energy_summary = aggregate(metrics, ["condition", "calibration"], energy_metrics)
    energy_summary.to_csv(output / "energy_summary.csv", index=False)

    force = pd.read_csv(force_root / "force_drift_summary.csv", keep_default_na=False)
    if len(force) != 9 or set(zip(force.condition, force.seed.astype(int), strict=True)) != {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }:
        raise AssertionError("synthetic force metric matrix is incomplete")
    force_metrics = [
        "force_drift_rmse_ev_ang",
        "force_drift_mae_ev_ang",
        "base_force_rms_ev_ang",
        "adapted_force_rms_ev_ang",
        "force_cosine",
    ]
    force_summary = aggregate(force, ["condition"], force_metrics)
    force_summary.to_csv(output / "force_summary.csv", index=False)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {
        "zero_direct": "#666666",
        "reference_direct": "#D55E00",
        "reference_aligned": "#0072B2",
    }
    trajectory_metrics = (
        ("val_calibrated_mae_mev_atom", "validation calibrated MAE (meV/atom)"),
        ("val_calibrated_ec_drift_mev_atom", "validation EC drift (meV/atom)"),
    )
    for axis, (metric, label) in zip(axes, trajectory_metrics, strict=True):
        for condition in CONDITIONS:
            block = epoch_metrics[epoch_metrics.condition == condition]
            grouped = block.groupby("epoch", sort=True)[metric]
            mean = grouped.mean()
            standard_deviation = grouped.std(ddof=1)
            axis.plot(mean.index, mean, color=colors[condition], label=condition)
            axis.fill_between(
                mean.index,
                mean - standard_deviation,
                mean + standard_deviation,
                color=colors[condition],
                alpha=0.15,
                linewidth=0,
            )
        axis.set_xlabel("epoch")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "figures/training_dynamics.png", dpi=220)
    plt.close(figure)

    audit = {
        "status": "PASS",
        "checks": {
            "complete_trajectories": 9,
            "epochs_per_trajectory": 120,
            "energy_metric_rows": len(metrics),
            "force_metric_rows": len(force),
            "aggregation": "mean and sample standard deviation across three frozen seeds",
            "result_dependent_filtering": False,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
