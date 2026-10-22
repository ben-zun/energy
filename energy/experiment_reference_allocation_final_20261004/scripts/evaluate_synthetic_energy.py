from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from reference_allocation.metrics import (
    absolute_metrics,
    build_pairs,
    same_composition_metrics,
)
from torch_geometric.loader import DataLoader

CONDITIONS = {
    "zero_direct": {"route": "U", "shift": "zero"},
    "reference_direct": {"route": "U", "shift": "reference"},
    "reference_aligned": {"route": "A", "shift": "reference"},
}
SEEDS = (20260528, 20260529, 20260530)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    energy = args.energy_root.resolve(strict=True)
    experiment = args.experiment_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    upstream = energy / "experiment12_20260929_alignment"
    sys.path.insert(0, str(upstream / "code"))
    from alignment_common import (
        BATCH_EVAL,
        alignment_offset,
        atomic_fractions,
        build_model,
        group_labels,
        load_element_reference,
        load_graphs,
        load_meta,
        per_atom,
        route_prediction,
    )

    manifest_path = (
        energy
        / "runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision"
        / "fair_finetune_293/graph_cache/medium/graph_metadata.csv"
    )
    graph_path = manifest_path.parent / "test_graphs.pt"
    labels_path = experiment / "synthetic_reference_control/labels_v1/synthetic_labels.csv"
    manifest = load_meta()
    labels = pd.read_csv(labels_path, keep_default_na=False)
    if manifest.id.duplicated().any() or labels.id.duplicated().any():
        raise ValueError("duplicate medium-split/synthetic-label IDs")
    if not np.array_equal(manifest.id.to_numpy(), labels.id.to_numpy()):
        raise AssertionError("frozen synthetic labels are not aligned to medium-split metadata")
    test_positions = np.flatnonzero(manifest.split.to_numpy() == "test")
    if len(test_positions) != 2048:
        raise AssertionError("synthetic test split is not the frozen 2048 structures")
    test_ids = manifest.id.iloc[test_positions].tolist()
    frozen_labels = labels.set_index("id")
    fractions = atomic_fractions(manifest)
    groups = group_labels(manifest)
    pair_frame = pd.DataFrame(
        {"id": manifest.id.iloc[test_positions].to_numpy(), "composition_key": groups[test_positions]}
    )
    pairs = build_pairs(pair_frame)
    if len(pairs) != 53 or pairs.group_id.nunique() != 22:
        raise AssertionError("unexpected synthetic test same-composition cohort")
    pairs.to_csv(output / "metrics/synthetic_test_pairs.csv", index=False)
    mu, bias = load_element_reference()
    offset = alignment_offset(fractions, mu, bias)
    graphs = load_graphs(("test",))["test"]
    loader = DataLoader(graphs, batch_size=BATCH_EVAL, shuffle=False)
    model = build_model(torch.device(args.device))
    offset_tensor = torch.tensor(offset, dtype=torch.float32, device=args.device)

    pretrained = np.full(len(manifest), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            pretrained[batch.manifest_row.cpu().numpy()] = per_atom(model, batch).cpu().numpy()
    if not np.isfinite(pretrained[test_positions]).all():
        raise FloatingPointError("incomplete synthetic-test teacher prediction")
    frozen_pretrained = frozen_labels.loc[
        test_ids, "pretrained_ev_atom"
    ].to_numpy(np.float64)
    teacher_difference = np.abs(pretrained[test_positions] - frozen_pretrained)
    # Repeated FP32 GPU forwards are stable to a few micro-eV/atom; the
    # independent upstream initialization audit observed the same scale.
    teacher_tolerance_ev_atom = 1e-5
    if teacher_difference.max() > teacher_tolerance_ev_atom:
        raise AssertionError(
            "fresh medium-test teacher predictions disagree with frozen synthetic labels: "
            f"max={teacher_difference.max():.9g} eV/atom, "
            f"tolerance={teacher_tolerance_ev_atom:.9g} eV/atom"
        )

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, str]] = [
        {"role": "medium_split_manifest", "path": str(manifest_path), "sha256": sha256(manifest_path)},
        {"role": "medium_test_graphs", "path": str(graph_path), "sha256": sha256(graph_path)},
        {"role": "synthetic_labels", "path": str(labels_path), "sha256": sha256(labels_path)},
    ]
    for condition_name, condition in CONDITIONS.items():
        for seed in SEEDS:
            run = experiment / "synthetic_reference_control/runs" / condition_name / f"seed{seed}"
            status = json.loads((run / "status.json").read_text())
            if status.get("status") != "complete" or int(status.get("completed_epochs", 0)) != 120:
                raise RuntimeError(f"incomplete synthetic run: {run}")
            history = pd.read_csv(run / "history.csv")
            if len(history) != 120 or history.epoch.tolist() != list(range(1, 121)):
                raise AssertionError(f"invalid fixed training trajectory: {run}")
            best_row = history.loc[history.val_calibrated_mae_mev_atom.idxmin()]
            best_epoch = int(best_row.epoch)
            if best_epoch != int(status["best_epoch"]):
                raise AssertionError("stored best epoch differs from validation-only recomputation")
            checkpoint = run / f"epoch{best_epoch:03d}.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            native = np.full(len(manifest), np.nan, dtype=np.float64)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(args.device)
                    native[batch.manifest_row.cpu().numpy()] = route_prediction(
                        condition["route"], model, batch, offset_tensor
                    ).cpu().numpy()
            calibration_path = run / "calibration_history.csv"
            calibration = pd.read_csv(calibration_path)
            calibration = calibration[calibration.epoch.astype(int) == best_epoch]
            if len(calibration) != 1:
                raise AssertionError("selected train-only calibration missing")
            weights = calibration.iloc[0][[f"w_{number}" for number in range(1, 119)]].to_numpy(float)
            calibrated = native + fractions @ weights + float(calibration.iloc[0].b)
            frozen_target_column = (
                "target_zero_ev_atom"
                if condition["shift"] == "zero"
                else "target_reference_ev_atom"
            )
            frozen_target = frozen_labels.loc[test_ids, frozen_target_column].to_numpy(np.float64)
            reconstructed_target = (
                pretrained[test_positions]
                if condition["shift"] == "zero"
                else pretrained[test_positions] + offset[test_positions]
            )
            target_difference = np.abs(reconstructed_target - frozen_target)
            if target_difference.max() > teacher_tolerance_ev_atom:
                raise AssertionError(
                    f"evaluation target disagrees with frozen {frozen_target_column}: "
                    f"max={target_difference.max():.9g} eV/atom"
                )
            selected = test_positions
            block = pd.DataFrame(
                {
                    "id": manifest.id.iloc[selected].to_numpy(),
                    "composition_key": groups[selected],
                    "pretrained_ev_atom": frozen_pretrained,
                    "target_ev_atom": frozen_target,
                    "prediction_native_ev_atom": native[selected],
                    "prediction_calibrated_ev_atom": calibrated[selected],
                    "condition": condition_name,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                }
            )
            prediction_rows.append(block)
            for calibration_name, values in (("native", native[selected]), ("calibrated", calibrated[selected])):
                absolute = absolute_metrics(frozen_target, values)
                relation = pd.DataFrame(
                    {
                        "id": block.id,
                        "composition_key": block.composition_key,
                        "truth": frozen_pretrained,
                        "prediction": values,
                    }
                )
                drift = same_composition_metrics(relation, pairs)
                metric_rows.append(
                    {
                        "condition": condition_name,
                        "seed": seed,
                        "calibration": calibration_name,
                        **{
                            f"synthetic_{key}_mev" if key != "n" else "n": value * 1000
                            if key != "n" else value
                            for key, value in absolute.items()
                        },
                        **{
                            f"drift_{key}_mev" if key in {"ec_mae", "pair_delta_mae"} else f"drift_{key}":
                            value * 1000 if key in {"ec_mae", "pair_delta_mae"} else value
                            for key, value in drift.items()
                        },
                    }
                )
            selection_rows.append(
                {
                    "condition": condition_name,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selection_metric": "val_calibrated_mae_mev_atom",
                    "selection_value": float(best_row.val_calibrated_mae_mev_atom),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            input_rows.extend(
                [
                    {"role": f"checkpoint_{condition_name}_{seed}", "path": str(checkpoint), "sha256": sha256(checkpoint)},
                    {"role": f"calibration_{condition_name}_{seed}", "path": str(calibration_path), "sha256": sha256(calibration_path)},
                ]
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output / "predictions/synthetic_test_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "metrics/per_seed_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "metrics/selected_checkpoints.csv", index=False)
    pd.DataFrame(input_rows).to_csv(output / "input_manifest.csv", index=False)
    audit = {
        "status": "PASS",
        "checks": {
            "runs": len(selection_rows),
            "all_fixed_120_epochs": True,
            "checkpoint_selection": "validation calibrated MAE only",
            "test_n": len(test_positions),
            "same_composition_structures": int(
                pair_frame.id.isin(set(pairs.id_i) | set(pairs.id_j)).sum()
            ),
            "same_composition_groups": int(pairs.group_id.nunique()),
            "same_composition_pairs": len(pairs),
            "test_or_confirmation_labels_used_for_selection": False,
            "fresh_teacher_vs_frozen_label_max_abs_ev_atom": float(teacher_difference.max()),
            "fresh_teacher_vs_frozen_label_tolerance_ev_atom": teacher_tolerance_ev_atom,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
