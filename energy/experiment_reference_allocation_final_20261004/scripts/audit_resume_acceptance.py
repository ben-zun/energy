from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

METRIC_COLUMNS = [
    "train_loss",
    "val_native_mae_mev_atom",
    "val_calibrated_mae_mev_atom",
    "val_ec_drift_mev_atom",
    "val_calibrated_ec_drift_mev_atom",
    "parameter_update_l2",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interrupted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interrupted = args.interrupted.resolve(strict=True)
    control = args.control.resolve(strict=True)
    left = torch.load(interrupted / "epoch002.pt", map_location="cpu", weights_only=False)
    right = torch.load(control / "epoch002.pt", map_location="cpu", weights_only=False)
    left_history = pd.read_csv(interrupted / "history.csv")
    right_history = pd.read_csv(control / "history.csv")
    if len(left_history) != 2 or len(right_history) != 2:
        raise AssertionError("resume acceptance expects two completed epochs")
    model_differences = [
        float((left["model_state"][name] - right["model_state"][name]).abs().max())
        for name in left["model_state"]
        if left["model_state"][name].numel()
    ]
    metric_differences = (left_history[METRIC_COLUMNS] - right_history[METRIC_COLUMNS]).abs()
    checks = {
        "global_step_interrupted": int(left["global_step"]),
        "global_step_control": int(right["global_step"]),
        "data_loader_rng_equal": bool(
            torch.equal(left["data_loader_generator_state"], right["data_loader_generator_state"])
        ),
        "torch_rng_equal": bool(torch.equal(left["random_state_torch"], right["random_state_torch"])),
        "model_state_max_abs_difference": max(model_differences),
        "val_calibrated_mae_max_abs_difference_mev_atom": float(
            metric_differences.val_calibrated_mae_mev_atom.max()
        ),
        "val_ec_drift_max_abs_difference_mev_atom": float(
            metric_differences.val_ec_drift_mev_atom.max()
        ),
        "metric_differences": {
            column: float(metric_differences[column].max()) for column in METRIC_COLUMNS
        },
    }
    passed = (
        checks["global_step_interrupted"] == checks["global_step_control"] == 1024
        and checks["data_loader_rng_equal"]
        and checks["torch_rng_equal"]
        and checks["val_calibrated_mae_max_abs_difference_mev_atom"] < 0.2
        and checks["val_ec_drift_max_abs_difference_mev_atom"] < 0.01
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "claim": "resume equivalent within observed GPU process-level numerical tolerance; not bitwise",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
