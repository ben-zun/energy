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
from reference_allocation.required_vs_induced import (
    correction_summary,
    pair_corrections,
)
from torch_geometric.loader import DataLoader

SEEDS = (20260528, 20260529, 20260530)
ROUTES = {"U": "direct", "A": "aligned"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    energy = args.energy_root.resolve(strict=True)
    upstream = energy / "experiment12_20260929_alignment"
    exp11 = energy / "experiment11_20260928"
    cohort_path = energy / "experiment15_prediction_audit_20261001/support/cohort.csv"
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(upstream / "code"))
    from alignment_common import (
        BATCH_EVAL,
        alignment_offset,
        atomic_fractions,
        build_model,
        load_element_reference,
        per_atom,
        route_prediction,
        run_dir,
    )

    manifest_path = exp11 / "confirmation/manifest.csv"
    graph_path = exp11 / "confirmation/sevennet_graphs.pt"
    base_path = exp11 / "confirmation/sevennet_20260528.csv"
    selected_path = upstream / "results/selected_checkpoints.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    cohort = pd.read_csv(cohort_path, keep_default_na=False)
    base_archive = pd.read_csv(base_path, keep_default_na=False)
    selected = pd.read_csv(selected_path, keep_default_na=False)
    for name, frame in (("manifest", manifest), ("cohort", cohort), ("base", base_archive)):
        if frame.id.duplicated().any():
            raise ValueError(f"{name} has duplicate IDs")
    if not np.array_equal(manifest.id, base_archive.id):
        raise AssertionError("archived SevenNet base predictions are not manifest-aligned")
    common_ids = cohort.id.tolist()
    if len(common_ids) != 401 or not set(common_ids).issubset(manifest.id):
        raise AssertionError("the frozen common 401-structure cohort is unavailable")
    overall_ids = manifest.loc[manifest.overall_cohort.astype(bool), "id"].tolist()
    legacy_ids = manifest.loc[manifest.relative_cohort.astype(bool), "id"].tolist()
    if len(overall_ids) != 2048 or len(legacy_ids) != 422:
        raise AssertionError("unexpected SevenNet confirmation cohort sizes")

    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
    fractions = atomic_fractions(manifest)
    mu, bias = load_element_reference()
    offset = alignment_offset(fractions, mu, bias)
    offset_tensor = torch.tensor(offset, dtype=torch.float32, device=args.device)
    loader = DataLoader(graphs, batch_size=BATCH_EVAL, shuffle=False)
    model = build_model(torch.device(args.device))
    base = base_archive.base.to_numpy(np.float64)
    truth = manifest.y_true.to_numpy(np.float64)
    count_groups = np.rint(fractions * manifest.natoms.to_numpy()[:, None]).astype(int)
    composition_keys = np.array(["|".join(map(str, row)) for row in count_groups])

    common_frame = pd.DataFrame({"id": manifest.id, "composition_key": composition_keys})
    common_frame = common_frame[common_frame.id.isin(common_ids)]
    common_pairs = build_pairs(common_frame)
    legacy_frame = pd.DataFrame({"id": manifest.id, "composition_key": composition_keys})
    legacy_frame = legacy_frame[legacy_frame.id.isin(legacy_ids)]
    legacy_pairs = build_pairs(legacy_frame)
    common_pairs.to_csv(output / "metrics/common401_pairs.csv", index=False)
    legacy_pairs.to_csv(output / "metrics/legacy422_pairs.csv", index=False)

    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    paired_init_hashes: dict[int, list[str]] = {seed: [] for seed in SEEDS}

    for source_route, route in ROUTES.items():
        for seed in SEEDS:
            pick = selected[
                (selected.seed.astype(int) == seed)
                & (selected.route == source_route)
                & (selected.comparison == "final_best")
                & (selected.criterion == "final_best")
            ]
            if len(pick) != 1:
                raise AssertionError(f"non-unique frozen checkpoint choice: {source_route}/{seed}")
            epoch = int(pick.iloc[0].epoch)
            run = run_dir(source_route, seed)
            checkpoint = run / f"epoch{epoch:03d}.pt"
            status = json.loads((run / "status.json").read_text())
            paired_init_hashes[seed].append(status["init_hash"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(state["epoch"]) != epoch or state["route"] != source_route or int(state["seed"]) != seed:
                raise AssertionError("checkpoint metadata differs from frozen selection")
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            native = np.full(len(manifest), np.nan, dtype=np.float64)
            raw_model = np.full(len(manifest), np.nan, dtype=np.float64)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(args.device)
                    positions = batch.manifest_row.cpu().numpy()
                    native[positions] = route_prediction(
                        source_route, model, batch, offset_tensor
                    ).cpu().numpy()
                    raw_model[positions] = per_atom(model, batch).cpu().numpy()
            if not np.isfinite(native).all() or not np.isfinite(raw_model).all():
                raise FloatingPointError("nonfinite SevenNet forward predictions")
            calibration_path = run / "calibration_history.csv"
            calibration = pd.read_csv(calibration_path)
            calibration = calibration[calibration.epoch.astype(int) == epoch]
            if len(calibration) != 1:
                raise AssertionError("missing or duplicate frozen train-only calibration")
            weights = calibration.iloc[0][[f"w_{number}" for number in range(1, 119)]].to_numpy(float)
            calibrated = native + fractions @ weights + float(calibration.iloc[0].b)
            frame = pd.DataFrame(
                {
                    "id": manifest.id,
                    "seed": seed,
                    "route": route,
                    "split": "confirmation",
                    "formula": manifest.formula,
                    "composition_key": composition_keys,
                    "n_atoms": manifest.natoms.astype(int),
                    "y_target_mev_atom": 1000.0 * truth,
                    "y_pretrained_mev_atom": 1000.0 * base,
                    "y_pred_raw_mev_atom": 1000.0 * native,
                    "y_pred_calibrated_mev_atom": 1000.0 * calibrated,
                    "raw_model_output_mev_atom": 1000.0 * raw_model,
                    "selected_epoch": epoch,
                }
            )
            rows.append(frame)

            local = frame.set_index("id")
            for cohort_name, ids, pairs in (
                ("overall2048", overall_ids, None),
                ("common401", common_ids, common_pairs),
                ("legacy422", legacy_ids, legacy_pairs),
            ):
                block = local.loc[ids].reset_index()
                target_values = block.y_target_mev_atom.to_numpy() / 1000.0
                for calibration_name, column in (
                    ("raw", "y_pred_raw_mev_atom"),
                    ("calibrated", "y_pred_calibrated_mev_atom"),
                ):
                    predicted_values = block[column].to_numpy() / 1000.0
                    absolute = absolute_metrics(target_values, predicted_values)
                    record: dict[str, object] = {
                        "seed": seed,
                        "route": route,
                        "calibration": calibration_name,
                        "cohort": cohort_name,
                        **{
                            f"{key}_mev" if key != "n" else key: value * 1000 if key != "n" else value
                            for key, value in absolute.items()
                        },
                    }
                    if pairs is not None:
                        relation = pd.DataFrame(
                            {
                                "id": block.id,
                                "composition_key": block.composition_key,
                                "truth": target_values,
                                "prediction": predicted_values,
                            }
                        )
                        same = same_composition_metrics(relation, pairs)
                        record.update(
                            {
                                f"{key}_mev" if key in {"ec_mae", "pair_delta_mae"} else key:
                                value * 1000 if key in {"ec_mae", "pair_delta_mae"} else value
                                for key, value in same.items()
                            }
                        )
                    metric_rows.append(record)

            mechanism = local.loc[common_ids].reset_index()
            mechanism_input = pd.DataFrame(
                {
                    "id": mechanism.id,
                    "target": mechanism.y_target_mev_atom / 1000.0,
                    "pretrained": mechanism.y_pretrained_mev_atom / 1000.0,
                    route: mechanism.y_pred_calibrated_mev_atom / 1000.0,
                }
            )
            pair_table = pair_corrections(mechanism_input, common_pairs, prediction_cols=(route,))
            summary = correction_summary(pair_table.r_target.to_numpy(), pair_table[f"r_{route}"].to_numpy())
            mechanism_rows.append({"model": "sevennet", "seed": seed, "route": route, **summary})
            pair_table.to_csv(output / "metrics" / f"required_vs_induced_seed{seed}_{route}.csv", index=False)
            input_rows.extend(
                [
                    {"role": f"checkpoint_{source_route}_{seed}", "path": str(checkpoint), "sha256": sha256(checkpoint)},
                    {"role": f"calibration_{source_route}_{seed}", "path": str(calibration_path), "sha256": sha256(calibration_path)},
                ]
            )

    for seed, hashes in paired_init_hashes.items():
        if len(hashes) != 2 or hashes[0] != hashes[1]:
            raise AssertionError(f"direct/aligned initialization hash mismatch for seed {seed}")
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(output / "predictions/sevennet_test_predictions.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "metrics/per_seed_metrics.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(output / "metrics/required_vs_induced_summary.csv", index=False)

    # Independent regression against the upstream evaluator. Small GPU forward
    # nondeterminism is tolerated, but any scientific-scale discrepancy fails.
    upstream_metrics_path = upstream / "results/final_metrics.csv"
    upstream_metrics = pd.read_csv(upstream_metrics_path)
    upstream_metrics = upstream_metrics[
        (upstream_metrics.comparison == "final_best")
        & (upstream_metrics.criterion == "final_best")
    ].copy()
    upstream_metrics["route"] = upstream_metrics.route.map(ROUTES)
    checks: list[float] = []
    for _, archived in upstream_metrics.iterrows():
        common = metrics_frame[
            (metrics_frame.seed.astype(int) == int(archived.seed))
            & (metrics_frame.route == archived.route)
            & (metrics_frame.calibration == "calibrated")
        ]
        overall = common[common.cohort == "overall2048"].iloc[0]
        legacy = common[common.cohort == "legacy422"].iloc[0]
        checks.extend(
            [
                abs(float(overall.mae_mev) - float(archived.overall_abs_mae_calibrated)),
                abs(float(legacy.mae_mev) - float(archived.relative_abs_mae_calibrated)),
                abs(float(legacy.ec_mae_mev) - float(archived.relative_ec_mae)),
            ]
        )
    upstream_regression_max_abs_mev = max(checks)
    if upstream_regression_max_abs_mev > 1e-3:
        raise AssertionError(
            f"SevenNet upstream metric regression mismatch: {upstream_regression_max_abs_mev} meV/atom"
        )
    for role, path in (
        ("manifest", manifest_path),
        ("graphs", graph_path),
        ("pretrained", base_path),
        ("selected_checkpoints", selected_path),
        ("upstream_final_metrics", upstream_metrics_path),
        ("common_cohort", cohort_path),
    ):
        input_rows.append({"role": role, "path": str(path), "sha256": sha256(path)})
    pd.DataFrame(input_rows).to_csv(output / "input_manifest.csv", index=False)
    audit = {
        "status": "PASS",
        "checks": {
            "overall_n": len(overall_ids),
            "common_n": len(common_ids),
            "common_groups": int(common_frame.composition_key.nunique()),
            "legacy_n": len(legacy_ids),
            "legacy_groups": int(legacy_frame.composition_key.nunique()),
            "paired_initialization_hashes": paired_init_hashes,
            "prediction_rows": len(predictions),
            "upstream_metric_regression_max_abs_mev_atom": upstream_regression_max_abs_mev,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
