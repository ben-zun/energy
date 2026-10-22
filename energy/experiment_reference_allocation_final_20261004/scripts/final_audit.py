from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

EXPECTED_PROTOCOL_SHA256 = "5637cdd2cae838c6f4f6adc7a18d6e00f99f1f98d5c9f5c2f9517b32aece5d32"
SEEDS = (20260528, 20260529, 20260530)
CONDITIONS = ("zero_direct", "reference_direct", "reference_aligned")
REQUIRED_CHECKPOINT_KEYS = {
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "grad_scaler_state",
    "epoch",
    "global_step",
    "condition",
    "seed",
    "initialization_hash",
    "best_epoch",
    "best_val_calibrated_mae_mev_atom",
    "random_state_python",
    "random_state_numpy",
    "random_state_torch",
    "random_state_cuda",
    "data_loader_generator_state",
}


def fp32_loss_equivalence() -> dict[str, float]:
    """Check the locked physical/encoded Smooth-L1 identity without autocast."""
    features = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 0.25]], dtype=torch.float32
    )
    weight = torch.tensor([0.3, -0.2], dtype=torch.float32, requires_grad=True)
    raw = features @ weight
    sigma = torch.tensor(0.6274358034133911, dtype=torch.float32)
    offset = torch.tensor([-5.0, -7.0, -3.0], dtype=torch.float32)
    residual = torch.tensor([0.001, 0.2, -0.3], dtype=torch.float32)
    target = (raw.detach() * sigma + offset + residual).detach()
    physical = torch.nn.functional.smooth_l1_loss(
        raw * sigma + offset,
        target,
        beta=0.05,
    )
    encoded_target = (target - offset) / sigma
    encoded = sigma * torch.nn.functional.smooth_l1_loss(
        raw,
        encoded_target,
        beta=0.05 / float(sigma),
    )
    raw_physical = torch.autograd.grad(physical, raw, retain_graph=True)[0]
    raw_encoded = torch.autograd.grad(encoded, raw, retain_graph=True)[0]
    parameter_physical = torch.autograd.grad(physical, weight, retain_graph=True)[0]
    parameter_encoded = torch.autograd.grad(encoded, weight)[0]
    result = {
        "loss_abs_difference": float(abs(physical.detach() - encoded.detach())),
        "raw_gradient_max_abs_difference": float(
            (raw_physical - raw_encoded).abs().max()
        ),
        "parameter_gradient_max_abs_difference": float(
            (parameter_physical - parameter_encoded).abs().max()
        ),
    }
    if max(result.values()) > 1e-6:
        raise AssertionError(f"FP32 physical/encoded loss identity failed: {result}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pass(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise AssertionError(f"audit is not PASS: {path}")
    return payload


def id_hash(ids: pd.Series) -> str:
    payload = "\n".join(sorted(ids.astype(str))) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    checks: list[dict[str, object]] = []
    hashes: list[dict[str, str]] = []

    protocol = root / "PROTOCOL.md"
    config_path = root / "config.yaml"
    protocol_hash = sha256(protocol)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError("frozen protocol hash changed")
    config = yaml.safe_load(config_path.read_text())
    if tuple(config["seeds"]) != SEEDS:
        raise AssertionError("config seeds changed")
    checks.append({"check": "frozen protocol/config", "status": "PASS", "detail": protocol_hash})
    hashes.extend(
        [
            {"role": "protocol", "path": str(protocol), "sha256": protocol_hash},
            {"role": "config", "path": str(config_path), "sha256": sha256(config_path)},
        ]
    )

    loss_equivalence = fp32_loss_equivalence()
    checks.append(
        {
            "check": "FP32/no-autocast physical and encoded loss/gradient equivalence",
            "status": "PASS",
            "detail": loss_equivalence,
        }
    )

    audit_paths = {
        "eqv3": root / "eqv3_final_eval/audit.json",
        "sevennet": root / "sevennet_final_eval_v2/audit.json",
        "required_vs_induced": root / "required_vs_induced/audit.json",
        "synthetic_labels": root / "synthetic_reference_control/labels_v1/audit.json",
        "resume": root / "synthetic_reference_control/resume_accept/RESUME_AUDIT.json",
        "synthetic_energy": root / "synthetic_reference_control/final_energy/audit.json",
        "synthetic_force": root / "synthetic_reference_control/final_force/audit.json",
        "synthetic_analysis": root / "synthetic_reference_control/final_analysis/audit.json",
        "cross_protocol": root / "cross_protocol_summary/audit.json",
        "closure_postprocessing": root / "closure_postprocessing/audit.json",
        "publication_lock": root / "publication_lock/audit.json",
    }
    for name, path in audit_paths.items():
        load_pass(path)
        checks.append({"check": f"component audit: {name}", "status": "PASS", "detail": str(path)})
        hashes.append({"role": f"audit_{name}", "path": str(path), "sha256": sha256(path)})

    labels_path = root / "synthetic_reference_control/labels_v1/synthetic_labels.csv"
    labels = pd.read_csv(labels_path, keep_default_na=False)
    if labels.id.duplicated().any() or not np.isfinite(
        labels[["pretrained_ev_atom", "reference_shift_ev_atom", "target_zero_ev_atom",
                "target_reference_ev_atom"]].to_numpy(float)
    ).all():
        raise AssertionError("invalid synthetic label table")
    split_counts = labels.split.value_counts().to_dict()
    if split_counts != {"train": 8192, "test": 2048, "val": 1024}:
        raise AssertionError(f"synthetic split sizes changed: {split_counts}")
    split_sets = {split: set(labels.loc[labels.split == split, "id"]) for split in split_counts}
    if any(split_sets[left] & split_sets[right] for left in split_sets for right in split_sets if left < right):
        raise AssertionError("synthetic train/val/test IDs overlap")
    for split in ("train", "val", "test"):
        hashes.append(
            {"role": f"synthetic_{split}_ids", "path": str(labels_path), "sha256": id_hash(labels.loc[labels.split == split, "id"])}
        )
    checks.append({"check": "synthetic split identity/disjointness", "status": "PASS", "detail": split_counts})
    reference_spread = (
        labels.groupby("composition_key", sort=True).reference_shift_ev_atom.max()
        - labels.groupby("composition_key", sort=True).reference_shift_ev_atom.min()
    )
    repeated_reference_spread = reference_spread[
        labels.composition_key.value_counts().reindex(reference_spread.index).to_numpy() >= 2
    ]
    epoch0_group_spread = float(repeated_reference_spread.max())
    if epoch0_group_spread > 1e-10:
        raise AssertionError(
            "synthetic aligned/direct epoch-0 difference is not composition-only: "
            f"{epoch0_group_spread} eV/atom"
        )
    checks.append(
        {
            "check": "synthetic epoch-0 aligned/direct centered prediction identity",
            "status": "PASS",
            "detail": epoch0_group_spread,
        }
    )

    initialization_hashes: set[str] = set()
    selected_checkpoint_hashes: dict[tuple[str, int], str] = {}
    selection = pd.read_csv(
        root / "synthetic_reference_control/final_energy/metrics/selected_checkpoints.csv",
        keep_default_na=False,
    )
    for condition in CONDITIONS:
        for seed in SEEDS:
            run = root / "synthetic_reference_control/runs" / condition / f"seed{seed}"
            status = json.loads((run / "status.json").read_text())
            history = pd.read_csv(run / "history.csv")
            if status.get("status") != "complete" or int(status.get("completed_epochs", 0)) != 120:
                raise AssertionError(f"incomplete run: {condition}/{seed}")
            if len(history) != 120 or history.epoch.tolist() != list(range(1, 121)):
                raise AssertionError(f"invalid fixed trajectory: {condition}/{seed}")
            latest = torch.load(run / "epoch120.pt", map_location="cpu", weights_only=False)
            missing = REQUIRED_CHECKPOINT_KEYS.difference(latest)
            if missing or int(latest["global_step"]) != 120 * 512:
                raise AssertionError(f"incomplete resume state for {condition}/{seed}: {sorted(missing)}")
            initialization_hashes.add(str(latest["initialization_hash"]))
            chosen = selection[(selection.condition == condition) & (selection.seed.astype(int) == seed)]
            if len(chosen) != 1:
                raise AssertionError("missing frozen synthetic selection")
            checkpoint = Path(chosen.iloc[0].checkpoint).resolve(strict=True)
            digest = sha256(checkpoint)
            if digest != chosen.iloc[0].checkpoint_sha256:
                raise AssertionError("selected checkpoint hash changed")
            selected_checkpoint_hashes[(condition, seed)] = digest
            hashes.append(
                {"role": f"synthetic_checkpoint_{condition}_{seed}", "path": str(checkpoint), "sha256": digest}
            )
    if len(initialization_hashes) != 1:
        raise AssertionError("synthetic conditions/seeds do not share the same initialization")
    checks.append(
        {"check": "nine fixed 120-epoch runs and full resume states", "status": "PASS", "detail": next(iter(initialization_hashes))}
    )

    energy_root = root.parent
    eqv3_upstream = energy_root / "experiment17_eqv3_physical_matched_20261001"
    eqv3_hashes: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    maximum_eqv3_group_spread = 0.0
    maximum_eqv3_loss_difference = 0.0
    maximum_eqv3_raw_gradient_difference = 0.0
    maximum_eqv3_loss_difference_gradient = 0.0
    for route in ("direct", "clean"):
        for seed in SEEDS:
            gate_path = eqv3_upstream / "runs" / route / f"seed{seed}" / "gate.json"
            gate = json.loads(gate_path.read_text())
            eqv3_hashes[seed].add(str(gate["initial_weight_hash"]))
            maximum_eqv3_group_spread = max(
                maximum_eqv3_group_spread,
                float(gate["within_composition_prediction_difference_max_ev_atom"]),
            )
            if int(gate["parameter_updates_during_gate"]) != 0:
                raise AssertionError("EqV3 initialization gate changed parameters")
            for gradient in gate["gradient_checks"]:
                maximum_eqv3_loss_difference = max(
                    maximum_eqv3_loss_difference,
                    abs(float(gradient["loss_decoded"]) - float(gradient["loss_encoded"])),
                )
                maximum_eqv3_raw_gradient_difference = max(
                    maximum_eqv3_raw_gradient_difference,
                    float(gradient["raw_output_gradient_max_difference"]),
                )
                maximum_eqv3_loss_difference_gradient = max(
                    maximum_eqv3_loss_difference_gradient,
                    float(gradient["gradient_of_loss_difference_relative_norm"]),
                )
            hashes.append(
                {
                    "role": f"eqv3_gate_{route}_{seed}",
                    "path": str(gate_path),
                    "sha256": sha256(gate_path),
                }
            )
    if any(len(values) != 1 for values in eqv3_hashes.values()):
        raise AssertionError("EqV3 paired routes do not share initialization")
    if maximum_eqv3_group_spread > 1e-10:
        raise AssertionError("EqV3 initialization changed centered within-group predictions")
    if (
        maximum_eqv3_loss_difference > 1e-8
        or maximum_eqv3_raw_gradient_difference != 0.0
        or maximum_eqv3_loss_difference_gradient > 1e-6
    ):
        raise AssertionError("EqV3 physical/encoded gate evidence failed")
    checks.append(
        {
            "check": "EqV3 paired initialization and production loss gates",
            "status": "PASS",
            "detail": {
                "paired_initialization_hashes": {
                    seed: next(iter(values)) for seed, values in eqv3_hashes.items()
                },
                "maximum_within_group_spread_ev_atom": maximum_eqv3_group_spread,
                "maximum_loss_abs_difference": maximum_eqv3_loss_difference,
                "maximum_raw_gradient_abs_difference": maximum_eqv3_raw_gradient_difference,
                "maximum_gradient_of_loss_difference_relative_norm": (
                    maximum_eqv3_loss_difference_gradient
                ),
            },
        }
    )

    sevennet_upstream = energy_root / "experiment12_20260929_alignment"
    sevennet_hashes: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    sevennet_group_spreads: list[float] = []
    for route in ("U", "A"):
        for seed in SEEDS:
            status_path = sevennet_upstream / "runs" / f"{route}_{seed}" / "status.json"
            status = json.loads(status_path.read_text())
            if status.get("status") != "complete" or int(status.get("epochs", 0)) != 120:
                raise AssertionError("SevenNet upstream matched run is incomplete")
            sevennet_hashes[seed].add(str(status["init_hash"]))
            sevennet_group_spreads.append(
                float(status["init_checks"]["medium_d_within_group_max_spread_ev_atom"])
            )
            hashes.append(
                {
                    "role": f"sevennet_status_{route}_{seed}",
                    "path": str(status_path),
                    "sha256": sha256(status_path),
                }
            )
    if any(len(values) != 1 for values in sevennet_hashes.values()):
        raise AssertionError("SevenNet paired routes do not share initialization")
    if max(sevennet_group_spreads) > 1e-6:
        raise AssertionError("SevenNet initialization reference is not composition-only")
    checks.append(
        {
            "check": "SevenNet paired initialization and epoch-0 group identity",
            "status": "PASS",
            "detail": {
                "paired_initialization_hashes": {
                    seed: next(iter(values)) for seed, values in sevennet_hashes.items()
                },
                "maximum_within_group_reference_spread_ev_atom": max(
                    sevennet_group_spreads
                ),
            },
        }
    )

    for model, path, expected_rows in (
        ("eqv3", root / "eqv3_final_eval/predictions/eqv3_test_predictions.csv", 14616),
        ("sevennet", root / "sevennet_final_eval_v2/predictions/sevennet_test_predictions.csv", 14616),
    ):
        predictions = pd.read_csv(path, keep_default_na=False)
        if len(predictions) != expected_rows or predictions[["id", "seed", "route"]].duplicated().any():
            raise AssertionError(f"invalid {model} prediction table")
        numeric = predictions.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise AssertionError(f"nonfinite {model} predictions")
        hashes.append({"role": f"predictions_{model}", "path": str(path), "sha256": sha256(path)})
    checks.append({"check": "EqV3/SevenNet prediction identity and finiteness", "status": "PASS", "detail": "14616 rows each"})

    synthetic_prediction_path = root / "synthetic_reference_control/final_energy/predictions/synthetic_test_predictions.csv"
    synthetic_predictions = pd.read_csv(synthetic_prediction_path, keep_default_na=False)
    if len(synthetic_predictions) != 9 * 2048:
        raise AssertionError("unexpected synthetic prediction count")
    pair_effects: list[float] = []
    for (_, _), block in synthetic_predictions.groupby(["condition", "seed"]):
        native = block.set_index("id").prediction_native_ev_atom
        calibrated = block.set_index("id").prediction_calibrated_ev_atom
        for _, group in block.groupby("composition_key"):
            ids = group.id.tolist()
            if len(ids) >= 2:
                pair_effects.append(float((calibrated[ids[0]] - calibrated[ids[1]]) - (native[ids[0]] - native[ids[1]])))
    cancellation = max(abs(np.asarray(pair_effects)))
    if cancellation > 1e-10:
        raise AssertionError(f"composition calibration changed same-composition energy: {cancellation}")
    checks.append({"check": "synthetic post-calibration same-composition cancellation", "status": "PASS", "detail": cancellation})
    hashes.append({"role": "synthetic_predictions", "path": str(synthetic_prediction_path), "sha256": sha256(synthetic_prediction_path)})

    force_summary = pd.read_csv(
        root / "synthetic_reference_control/final_force/force_drift_summary.csv"
    )
    finite = pd.read_csv(
        root / "synthetic_reference_control/final_force/finite_difference_checks.csv"
    )
    if len(force_summary) != 9 or not np.isfinite(force_summary.select_dtypes(include="number")).all().all():
        raise AssertionError("force summary incomplete/nonfinite")
    if len(finite) != 4 * 8 * 4 or not np.isfinite(finite.select_dtypes(include="number")).all().all():
        raise AssertionError("finite-difference audit incomplete/nonfinite")
    checks.append({"check": "force drift and finite difference tables", "status": "PASS", "detail": f"{len(finite)} finite differences"})
    for role, path in (
        ("force_drift_summary", root / "synthetic_reference_control/final_force/force_drift_summary.csv"),
        ("finite_difference_checks", root / "synthetic_reference_control/final_force/finite_difference_checks.csv"),
    ):
        hashes.append({"role": role, "path": str(path), "sha256": sha256(path)})

    deliverables = (
        ("final_report", root / "FINAL_REPORT.md"),
        ("required_summary", root / "required_vs_induced/summary_with_group_bootstrap.csv"),
        ("required_scatter_figure", root / "required_vs_induced/figures/required_vs_induced.png"),
        ("required_metric_figure", root / "required_vs_induced/figures/required_vs_induced_metrics.png"),
        ("synthetic_energy_summary", root / "synthetic_reference_control/final_analysis/energy_summary.csv"),
        ("synthetic_force_summary", root / "synthetic_reference_control/final_analysis/force_summary.csv"),
        ("synthetic_training_figure", root / "synthetic_reference_control/final_analysis/figures/training_dynamics.png"),
        ("cross_protocol_summary", root / "cross_protocol_summary/summary.csv"),
        (
            "synthetic_paired_bootstrap_summary",
            root / "closure_postprocessing/synthetic_paired_bootstrap_summary.csv",
        ),
        (
            "synthetic_paired_bootstrap_draws",
            root / "closure_postprocessing/synthetic_paired_bootstrap_draws.csv",
        ),
        (
            "synthetic_projection_summary",
            root / "closure_postprocessing/synthetic_projection_summary.csv",
        ),
        (
            "raw_postcalibration_summary",
            root / "closure_postprocessing/raw_postcalibration_summary.csv",
        ),
        (
            "synthetic_bootstrap_figure",
            root / "closure_postprocessing/figures/synthetic_paired_bootstrap.png",
        ),
        (
            "synthetic_projection_figure",
            root / "closure_postprocessing/figures/synthetic_composition_projection.png",
        ),
        ("current_number_registry", root / "publication_lock/CURRENT_NUMBER_REGISTRY.csv"),
        ("legacy_number_audit", root / "publication_lock/LEGACY_NUMBER_AUDIT.md"),
        ("experiment_closed", root / "publication_lock/EXPERIMENT_CLOSED.json"),
    )
    for role, path in deliverables:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing/empty final deliverable: {path}")
        hashes.append({"role": role, "path": str(path), "sha256": sha256(path)})
    checks.append(
        {
            "check": "final reports, tables and figures",
            "status": "PASS",
            "detail": len(deliverables),
        }
    )

    final_report_text = (root / "FINAL_REPORT.md").read_text()
    forbidden_report_terms = ("Common401", "Relative401", "85 groups / 209")
    found_forbidden = [term for term in forbidden_report_terms if term in final_report_text]
    required_report_terms = (
        "fixed confirmation cohort",
        "90 groups / 214 structures / 202 pairs",
        "83.667",
        "Synthetic paired uncertainty and composition-direction projection",
        "Raw versus post-calibrated supplementary summary",
    )
    missing_required = [term for term in required_report_terms if term not in final_report_text]
    if found_forbidden or missing_required:
        raise AssertionError(
            "final-report terminology/number lock failed: "
            f"forbidden={found_forbidden}, missing={missing_required}"
        )
    checks.append(
        {
            "check": "final-report terminology and canonical-number lock",
            "status": "PASS",
            "detail": "fixed confirmation; 90/214/202; direct 83.667",
        }
    )

    hashes_frame = pd.DataFrame(hashes)
    hashes_frame.to_csv(root / "FINAL_HASHES.csv", index=False)
    report = ["# Final numerical audit", "", "All checks below were executed; none are pre-marked.", ""]
    for item in checks:
        report.append(f"- PASS — {item['check']}: `{item['detail']}`")
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "PASS certifies the frozen definitions, artifacts, IDs, checkpoint states and numerical invariants above. "
                "It does not by itself establish a scientific mechanism or guarantee that aligned must outperform direct."
            ),
            "",
            "AUDIT_OK",
        ]
    )
    (root / "FINAL_AUDIT.md").write_text("\n".join(report) + "\n")
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
