from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reference_allocation.metrics import (
    absolute_metrics,
    build_pairs,
    same_composition_metrics,
)
from reference_allocation.required_vs_induced import (
    correction_summary,
    pair_corrections,
)

SEEDS = (20260528, 20260529, 20260530)
ROUTES = {"direct": "direct", "clean": "aligned"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    if frame.id.duplicated().any():
        raise ValueError(f"{name} contains duplicate IDs")


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"value": values, "group": groups}).groupby("group").value.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    numerator = np.zeros(draws, dtype=np.float64)
    denominator = np.zeros(draws, dtype=np.float64)
    # Chunking avoids a draws×groups allocation and is deterministic.
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        picked = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        numerator[start:stop] = grouped["sum"].to_numpy()[picked].sum(axis=1)
        denominator[start:stop] = grouped["count"].to_numpy()[picked].sum(axis=1)
    distribution = numerator / denominator
    return float(np.mean(values)), float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    args = parser.parse_args()

    inputs = args.input_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "bootstrap").mkdir(parents=True, exist_ok=True)

    paths = {
        "metadata": inputs / "metadata.csv",
        "relative_cohort": inputs / "cohort.csv",
        "confirmation_manifest": inputs / "confirmation_manifest.csv",
        "pretrained": inputs / "eqv3_base.csv",
        "selected_checkpoints": inputs / "selected_checkpoints.csv",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    metadata = read_csv(paths["metadata"])
    relative_manifest = read_csv(paths["relative_cohort"])
    confirmation_manifest = read_csv(paths["confirmation_manifest"])
    pretrained = read_csv(paths["pretrained"])
    selected = read_csv(paths["selected_checkpoints"])
    for name, frame in (("metadata", metadata), ("relative", relative_manifest),
                        ("confirmation", confirmation_manifest), ("pretrained", pretrained)):
        assert_unique(frame, name)

    overall_ids = confirmation_manifest.loc[confirmation_manifest.overall_cohort.astype(bool), "id"].tolist()
    relative_ids = relative_manifest.id.tolist()
    if len(overall_ids) != 2048 or len(relative_ids) != 401:
        raise AssertionError(f"unexpected frozen cohort sizes: overall={len(overall_ids)}, relative={len(relative_ids)}")
    if not set(overall_ids).issubset(pretrained.id) or not set(relative_ids).issubset(pretrained.id):
        raise AssertionError("pretrained predictions do not cover frozen cohorts")

    meta = metadata.set_index("id")
    base = pretrained.set_index("id")["base"]
    relative_rows = relative_manifest[["id", "formula", "natoms"]].copy()
    # Use the locked 118-dimensional integer-count vector, not reduced_formula
    # or a display formula, for all same-composition operations.
    relative_rows["composition_key"] = meta.loc[relative_rows.id, "group"].to_numpy()
    pairs = build_pairs(relative_rows)
    if relative_rows.composition_key.nunique() != 190 or len(pairs) == 0:
        raise AssertionError("relative cohort does not match the frozen 190-group definition")
    pairs.to_csv(output / "metrics" / "same_composition_pairs.csv", index=False)

    long_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    predictions: dict[tuple[int, str], pd.DataFrame] = {}

    for source_route, route in ROUTES.items():
        for seed in SEEDS:
            path = inputs / source_route / f"seed{seed}" / "confirmation_calibrated.csv"
            frame = read_csv(path)
            assert_unique(frame, f"{source_route}/{seed}")
            if set(frame.id) != set(confirmation_manifest.id):
                raise AssertionError(f"prediction ID mismatch for {source_route}/{seed}")
            selection = selected[(selected.seed.astype(int) == seed) & (selected.route == source_route)]
            if len(selection) != 1:
                raise AssertionError(f"checkpoint selection is not unique for {source_route}/{seed}")
            epoch = int(selection.iloc[0].best_epoch)
            indexed = frame.set_index("id").loc[confirmation_manifest.id].reset_index()
            np.testing.assert_allclose(indexed.y_true, confirmation_manifest.y_true, atol=1e-12, rtol=0)
            indexed["pretrained"] = base.loc[indexed.id].to_numpy()
            indexed["seed"] = seed
            indexed["route"] = route
            indexed["selected_epoch"] = epoch
            indexed["composition_key"] = meta.loc[indexed.id, "group"].to_numpy()
            indexed["y_target_mev_atom"] = 1000.0 * indexed.y_true
            indexed["y_pretrained_mev_atom"] = 1000.0 * indexed.pretrained
            indexed["y_pred_raw_mev_atom"] = 1000.0 * indexed.native_prediction
            indexed["y_pred_calibrated_mev_atom"] = 1000.0 * indexed.calibrated_prediction
            indexed["n_atoms"] = indexed.natoms.astype(int)
            indexed["split"] = "confirmation"
            export_columns = [
                "id", "seed", "route", "split", "formula", "composition_key", "n_atoms",
                "y_target_mev_atom", "y_pretrained_mev_atom", "y_pred_raw_mev_atom",
                "y_pred_calibrated_mev_atom", "selected_epoch",
            ]
            long_rows.append(indexed[export_columns])
            predictions[(seed, route)] = indexed.set_index("id")

            for calibration, prediction_column in (("raw", "native_prediction"),
                                                   ("calibrated", "calibrated_prediction")):
                for cohort, ids in (("overall2048", overall_ids), ("relative401", relative_ids)):
                    block = indexed.set_index("id").loc[ids].reset_index()
                    absolute = absolute_metrics(block.y_true, block[prediction_column])
                    record: dict[str, object] = {
                        "seed": seed, "route": route, "calibration": calibration, "cohort": cohort,
                        **{f"{key}_mev" if key != "n" else key: value * 1000 if key != "n" else value
                           for key, value in absolute.items()},
                    }
                    if cohort == "relative401":
                        relation = block[["id", "y_true", prediction_column, "composition_key"]].rename(
                            columns={"y_true": "truth", prediction_column: "prediction"}
                        )
                        same = same_composition_metrics(relation, pairs)
                        record.update({f"{key}_mev" if key in {"ec_mae", "pair_delta_mae"} else key:
                                       value * 1000 if key in {"ec_mae", "pair_delta_mae"} else value
                                       for key, value in same.items()})
                    metric_rows.append(record)

            mechanism_input = indexed.set_index("id").loc[relative_ids].reset_index()[
                ["id", "y_true", "pretrained", "calibrated_prediction"]
            ].rename(columns={"y_true": "target", "calibrated_prediction": route})
            pair_table = pair_corrections(mechanism_input, pairs, prediction_cols=(route,))
            summary = correction_summary(pair_table.r_target.to_numpy(), pair_table[f"r_{route}"].to_numpy())
            mechanism_rows.append({"model": "eqv3", "seed": seed, "route": route, **summary})
            pair_table.to_csv(output / "metrics" / f"required_vs_induced_seed{seed}_{route}.csv", index=False)

    prediction_table = pd.concat(long_rows, ignore_index=True)
    prediction_table.to_csv(output / "predictions" / "eqv3_test_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics" / "per_seed_metrics.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(output / "metrics" / "required_vs_induced_summary.csv", index=False)

    for seed in SEEDS:
        for cohort, ids in (("overall2048", overall_ids), ("relative401", relative_ids)):
            aligned = predictions[(seed, "aligned")].loc[ids]
            direct = predictions[(seed, "direct")].loc[ids]
            truth = aligned.y_true.to_numpy()
            if not np.array_equal(aligned.index, direct.index):
                raise AssertionError("paired routes are not ID-aligned")
            groups = aligned.composition_key.to_numpy() if cohort == "relative401" else aligned.index.to_numpy()
            delta = 1000.0 * (np.abs(aligned.calibrated_prediction.to_numpy() - truth)
                              - np.abs(direct.calibrated_prediction.to_numpy() - truth))
            point, low, high = cluster_bootstrap_mean(delta, groups, draws=args.bootstrap_draws,
                                                       seed=args.bootstrap_seed)
            paired_rows.append({"seed": seed, "cohort": cohort, "metric": "absolute_mae",
                                "aligned_minus_direct": point, "ci95_low": low, "ci95_high": high})
            if cohort == "relative401":
                def center(values: np.ndarray, group_values: np.ndarray) -> np.ndarray:
                    return values - pd.Series(values).groupby(group_values).transform("mean").to_numpy()

                aligned_error = center(aligned.calibrated_prediction.to_numpy() - truth, groups)
                direct_error = center(direct.calibrated_prediction.to_numpy() - truth, groups)
                delta_ec = 1000.0 * (np.abs(aligned_error) - np.abs(direct_error))
                point, low, high = cluster_bootstrap_mean(delta_ec, groups, draws=args.bootstrap_draws,
                                                           seed=args.bootstrap_seed)
                paired_rows.append({"seed": seed, "cohort": cohort, "metric": "ec_mae",
                                    "aligned_minus_direct": point, "ci95_low": low, "ci95_high": high})
    pd.DataFrame(paired_rows).to_csv(output / "bootstrap" / "paired_route_differences.csv", index=False)

    manifest_rows = [{"role": role, "path": str(path), "sha256": sha256(path)} for role, path in paths.items()]
    for source_route in ROUTES:
        for seed in SEEDS:
            path = inputs / source_route / f"seed{seed}" / "confirmation_calibrated.csv"
            manifest_rows.append({"role": f"prediction_{source_route}_{seed}", "path": str(path), "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(output / "input_manifest.csv", index=False)
    audit = {
        "status": "PASS",
        "checks": {
            "frozen_overall_n": len(overall_ids),
            "frozen_relative_n": len(relative_ids),
            "frozen_relative_groups": int(relative_rows.composition_key.nunique()),
            "routes": sorted(ROUTES.values()),
            "seeds": list(SEEDS),
            "prediction_rows": len(prediction_table),
            "test_not_used_for_refit": "verified_from_upstream artifacts; this script performs no fit",
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
