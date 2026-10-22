from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from reference_allocation.closure import (
    projection_metrics,
    weighted_mae_from_clusters,
    weighted_rmse_from_clusters,
)

CONDITIONS = ("zero_direct", "reference_direct", "reference_aligned")
SEEDS = (20260528, 20260529, 20260530)
COMPARISONS = (
    ("reference_direct", "reference_aligned"),
    ("reference_aligned", "zero_direct"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ec(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_ids = sorted(pairs.group_id.astype(str).unique())
    represented_ids = set(pairs.id_i) | set(pairs.id_j)
    frame = predictions[predictions.id.isin(represented_ids)].copy()
    if frame.groupby(["condition", "seed"]).size().nunique() != 1:
        raise AssertionError("synthetic conditions/seeds do not share the same relative cohort")
    frame["update_ev_atom"] = (
        frame.prediction_native_ev_atom - frame.pretrained_ev_atom
    )
    frame["centered_update_ev_atom"] = frame.update_ev_atom - frame.groupby(
        ["condition", "seed", "composition_key"], sort=False
    ).update_ev_atom.transform("mean")
    components = (
        frame.assign(absolute=frame.centered_update_ev_atom.abs())
        .groupby(["condition", "seed", "composition_key"], sort=True)
        .absolute.agg(["sum", "count"])
        .reset_index()
    )
    component_index = components.set_index(["condition", "seed", "composition_key"])
    for condition in CONDITIONS:
        for seed_value in SEEDS:
            observed = set(
                components.loc[
                    (components.condition == condition) & (components.seed == seed_value),
                    "composition_key",
                ].astype(str)
            )
            if observed != set(group_ids):
                raise AssertionError("EC cluster identities differ across conditions/seeds")

    def metric(condition: str, seed_value: int, sampled: np.ndarray) -> float:
        block = component_index.loc[(condition, seed_value)]
        block = block.loc[group_ids]
        return weighted_mae_from_clusters(
            block["sum"].to_numpy(np.float64),
            block["count"].to_numpy(np.float64),
            sampled,
        )

    full = np.arange(len(group_ids))
    points = {
        (condition, seed_value): metric(condition, seed_value, full)
        for condition in CONDITIONS
        for seed_value in SEEDS
    }
    random = np.random.default_rng(seed)
    draw_rows: list[dict[str, object]] = []
    for draw in range(draws):
        sampled = random.integers(0, len(group_ids), size=len(group_ids))
        for left, right in COMPARISONS:
            per_seed = [
                metric(left, seed_value, sampled) - metric(right, seed_value, sampled)
                for seed_value in SEEDS
            ]
            draw_rows.append(
                {
                    "endpoint": "ec_drift",
                    "comparison": f"{left}_minus_{right}",
                    "draw": draw,
                    "difference": 1000 * float(np.mean(per_seed)),
                }
            )
    draws_frame = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, object]] = []
    for left, right in COMPARISONS:
        comparison = f"{left}_minus_{right}"
        distribution = draws_frame.loc[
            draws_frame.comparison == comparison, "difference"
        ].to_numpy(np.float64)
        per_seed_points = {
            seed_value: 1000 * (points[(left, seed_value)] - points[(right, seed_value)])
            for seed_value in SEEDS
        }
        summary_rows.append(
            {
                "endpoint": "ec_drift",
                "comparison": comparison,
                "point_difference": float(np.mean(list(per_seed_points.values()))),
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
                "unit": "meV/atom",
                "clusters": len(group_ids),
                "draws": draws,
                "bootstrap_seed": seed,
                **{
                    f"seed{seed_value}_difference": value
                    for seed_value, value in per_seed_points.items()
                },
            }
        )
    return pd.DataFrame(summary_rows), draws_frame


def bootstrap_force(
    vectors: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure_ids = sorted(vectors.id.astype(str).unique())
    if len(structure_ids) != 12:
        raise AssertionError("force bootstrap requires the frozen 12 structures")
    base = vectors[["base_fx", "base_fy", "base_fz"]].to_numpy(np.float64)
    adapted = vectors[["adapted_fx", "adapted_fy", "adapted_fz"]].to_numpy(np.float64)
    differences = adapted - base
    work = vectors[["condition", "seed", "id"]].copy()
    work["squared_sum"] = np.sum(differences**2, axis=1)
    work["count"] = 3
    components = (
        work.groupby(["condition", "seed", "id"], sort=True)[["squared_sum", "count"]]
        .sum()
        .reset_index()
    )
    component_index = components.set_index(["condition", "seed", "id"])
    for condition in CONDITIONS:
        for seed_value in SEEDS:
            observed = set(
                components.loc[
                    (components.condition == condition) & (components.seed == seed_value), "id"
                ].astype(str)
            )
            if observed != set(structure_ids):
                raise AssertionError("force structure identities differ across conditions/seeds")

    def metric(condition: str, seed_value: int, sampled: np.ndarray) -> float:
        block = component_index.loc[(condition, seed_value)].loc[structure_ids]
        return weighted_rmse_from_clusters(
            block.squared_sum.to_numpy(np.float64),
            block["count"].to_numpy(np.float64),
            sampled,
        )

    full = np.arange(len(structure_ids))
    points = {
        (condition, seed_value): metric(condition, seed_value, full)
        for condition in CONDITIONS
        for seed_value in SEEDS
    }
    random = np.random.default_rng(seed)
    draw_rows: list[dict[str, object]] = []
    for draw in range(draws):
        sampled = random.integers(0, len(structure_ids), size=len(structure_ids))
        for left, right in COMPARISONS:
            per_seed = [
                metric(left, seed_value, sampled) - metric(right, seed_value, sampled)
                for seed_value in SEEDS
            ]
            draw_rows.append(
                {
                    "endpoint": "force_drift_rmse",
                    "comparison": f"{left}_minus_{right}",
                    "draw": draw,
                    "difference": float(np.mean(per_seed)),
                }
            )
    draws_frame = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, object]] = []
    for left, right in COMPARISONS:
        comparison = f"{left}_minus_{right}"
        distribution = draws_frame.loc[
            draws_frame.comparison == comparison, "difference"
        ].to_numpy(np.float64)
        per_seed_points = {
            seed_value: points[(left, seed_value)] - points[(right, seed_value)]
            for seed_value in SEEDS
        }
        summary_rows.append(
            {
                "endpoint": "force_drift_rmse",
                "comparison": comparison,
                "point_difference": float(np.mean(list(per_seed_points.values()))),
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
                "unit": "eV/angstrom",
                "clusters": len(structure_ids),
                "draws": draws,
                "bootstrap_seed": seed,
                **{
                    f"seed{seed_value}_difference": value
                    for seed_value, value in per_seed_points.items()
                },
            }
        )
    return pd.DataFrame(summary_rows), draws_frame


def projection_analysis(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for condition in ("reference_direct", "reference_aligned"):
        for seed_value in SEEDS:
            block = predictions[
                (predictions.condition == condition) & (predictions.seed == seed_value)
            ].sort_values("id")
            if len(block) != 2048 or block.id.duplicated().any():
                raise AssertionError("projection cohort is incomplete")
            target_direction = (
                block.target_ev_atom - block.pretrained_ev_atom
            ).to_numpy(np.float64)
            for calibration, column in (
                ("native", "prediction_native_ev_atom"),
                ("calibrated", "prediction_calibrated_ev_atom"),
            ):
                update = (block[column] - block.pretrained_ev_atom).to_numpy(np.float64)
                metrics = projection_metrics(target_direction, update)
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed_value,
                        "calibration": calibration,
                        "n": len(block),
                        **metrics,
                    }
                )
    per_seed = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    metric_columns = [
        "beta",
        "update_norm_ratio",
        "orthogonal_norm_ratio",
        "pearson",
        "cosine",
        "projection_explained_fraction",
        "residual_rms_ev_atom",
    ]
    for (condition, calibration), block in per_seed.groupby(
        ["condition", "calibration"], sort=True
    ):
        row: dict[str, object] = {
            "condition": condition,
            "calibration": calibration,
            "seeds": len(block),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_std"] = float(block[metric].std(ddof=1))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def raw_postcalibration_table(
    eqv3: pd.DataFrame,
    sevennet: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    specifications = (
        ("EqV3", eqv3, "overall2048", "relative401", "raw", "calibrated"),
        ("SevenNet", sevennet, "overall2048", "common401", "raw", "calibrated"),
    )
    for model, frame, absolute_cohort, relative_cohort, raw_name, calibrated_name in specifications:
        for route in ("direct", "aligned"):
            for seed_value in SEEDS:
                absolute = frame[
                    (frame.route == route)
                    & (frame.seed == seed_value)
                    & (frame.cohort == absolute_cohort)
                ].set_index("calibration")
                relative = frame[
                    (frame.route == route)
                    & (frame.seed == seed_value)
                    & (frame.cohort == relative_cohort)
                ].set_index("calibration")
                rows.append(
                    {
                        "model": model,
                        "route": route,
                        "seed": seed_value,
                        "absolute_cohort": absolute_cohort,
                        "relative_cohort": relative_cohort,
                        "raw_mae_mev_atom": float(absolute.loc[raw_name, "mae_mev"]),
                        "postcal_mae_mev_atom": float(
                            absolute.loc[calibrated_name, "mae_mev"]
                        ),
                        "raw_ec_mae_mev_atom": float(
                            relative.loc[raw_name, "ec_mae_mev"]
                        ),
                        "postcal_ec_mae_mev_atom": float(
                            relative.loc[calibrated_name, "ec_mae_mev"]
                        ),
                    }
                )
    for condition in CONDITIONS:
        for seed_value in SEEDS:
            block = synthetic[
                (synthetic.condition == condition) & (synthetic.seed == seed_value)
            ].set_index("calibration")
            rows.append(
                {
                    "model": "SevenNet-synthetic",
                    "route": condition,
                    "seed": seed_value,
                    "absolute_cohort": "synthetic_test2048",
                    "relative_cohort": "synthetic_22groups_52structures",
                    "raw_mae_mev_atom": float(block.loc["native", "synthetic_mae_mev"]),
                    "postcal_mae_mev_atom": float(
                        block.loc["calibrated", "synthetic_mae_mev"]
                    ),
                    "raw_ec_mae_mev_atom": float(block.loc["native", "drift_ec_mae_mev"]),
                    "postcal_ec_mae_mev_atom": float(
                        block.loc["calibrated", "drift_ec_mae_mev"]
                    ),
                }
            )
    per_seed = pd.DataFrame(rows)
    cancellation_error = (
        per_seed.raw_ec_mae_mev_atom - per_seed.postcal_ec_mae_mev_atom
    ).abs().max()
    if cancellation_error > 1e-10:
        raise AssertionError(f"composition calibration changed EC-MAE: {cancellation_error}")
    summary_rows: list[dict[str, object]] = []
    metrics = [
        "raw_mae_mev_atom",
        "postcal_mae_mev_atom",
        "raw_ec_mae_mev_atom",
        "postcal_ec_mae_mev_atom",
    ]
    for (model, route), block in per_seed.groupby(["model", "route"], sort=True):
        row: dict[str, object] = {"model": model, "route": route, "seeds": len(block)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_std"] = float(block[metric].std(ddof=1))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-energy-root", type=Path, required=True)
    parser.add_argument("--synthetic-force-root", type=Path, required=True)
    parser.add_argument("--eqv3-metrics", type=Path, required=True)
    parser.add_argument("--sevennet-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    energy_root = args.synthetic_energy_root.resolve(strict=True)
    force_root = args.synthetic_force_root.resolve(strict=True)
    eqv3_path = args.eqv3_metrics.resolve(strict=True)
    sevennet_path = args.sevennet_metrics.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "figures").mkdir(parents=True, exist_ok=True)

    prediction_path = energy_root / "predictions/synthetic_test_predictions.csv"
    pair_path = energy_root / "metrics/synthetic_test_pairs.csv"
    metric_path = energy_root / "metrics/per_seed_metrics.csv"
    vector_path = force_root / "force_vectors.csv"
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    pairs = pd.read_csv(pair_path, keep_default_na=False)
    synthetic_metrics = pd.read_csv(metric_path, keep_default_na=False)
    vectors = pd.read_csv(vector_path, keep_default_na=False)
    eqv3 = pd.read_csv(eqv3_path, keep_default_na=False)
    sevennet = pd.read_csv(sevennet_path, keep_default_na=False)

    if len(predictions) != 9 * 2048 or predictions[["condition", "seed", "id"]].duplicated().any():
        raise AssertionError("synthetic prediction matrix is incomplete or duplicated")
    if len(vectors.condition.unique()) != 3 or vectors[["condition", "seed", "id", "atom_index"]].duplicated().any():
        raise AssertionError("force vector matrix is incomplete or duplicated")

    ec_summary, ec_draws = bootstrap_ec(
        predictions, pairs, draws=args.draws, seed=args.seed
    )
    force_summary, force_draws = bootstrap_force(
        vectors, draws=args.draws, seed=args.seed + 1
    )
    bootstrap_summary = pd.concat([ec_summary, force_summary], ignore_index=True)
    bootstrap_draws = pd.concat([ec_draws, force_draws], ignore_index=True)
    bootstrap_summary.to_csv(output / "synthetic_paired_bootstrap_summary.csv", index=False)
    bootstrap_draws.to_csv(output / "synthetic_paired_bootstrap_draws.csv", index=False)

    projection_per_seed, projection_summary = projection_analysis(predictions)
    projection_per_seed.to_csv(output / "synthetic_projection_per_seed.csv", index=False)
    projection_summary.to_csv(output / "synthetic_projection_summary.csv", index=False)

    supplement_per_seed, supplement_summary = raw_postcalibration_table(
        eqv3, sevennet, synthetic_metrics
    )
    supplement_per_seed.to_csv(output / "raw_postcalibration_per_seed.csv", index=False)
    supplement_summary.to_csv(output / "raw_postcalibration_summary.csv", index=False)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    for endpoint, axis in zip(("ec_drift", "force_drift_rmse"), axes, strict=True):
        block = bootstrap_summary[bootstrap_summary.endpoint == endpoint]
        x = np.arange(len(block))
        y = block.point_difference.to_numpy(np.float64)
        low = y - block.ci95_low.to_numpy(np.float64)
        high = block.ci95_high.to_numpy(np.float64) - y
        axis.errorbar(x, y, yerr=np.vstack([low, high]), fmt="o", capsize=4)
        axis.axhline(0.0, color="black", linewidth=0.8)
        label_map = {
            "reference_direct_minus_reference_aligned": "direct − aligned",
            "reference_aligned_minus_zero_direct": "aligned − zero",
        }
        short_labels = [label_map[value] for value in block.comparison]
        axis.set_xticks(x, short_labels, rotation=0)
        axis.set_ylabel(f"paired difference ({block.unit.iloc[0]})")
        axis.set_title(endpoint.replace("_", " "))
    figure.tight_layout()
    figure.savefig(output / "figures/synthetic_paired_bootstrap.png", dpi=220)
    plt.close(figure)

    native_projection = projection_summary[projection_summary.calibration == "native"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for axis, metric, label in zip(
        axes,
        ("beta", "orthogonal_norm_ratio", "pearson"),
        ("projection β", "orthogonal norm / ||h||", "correlation(u, h)"),
        strict=True,
    ):
        means = native_projection[f"{metric}_mean"].to_numpy(np.float64)
        standard_deviations = native_projection[f"{metric}_std"].to_numpy(np.float64)
        axis.errorbar(
            np.arange(len(native_projection)), means, yerr=standard_deviations, fmt="o", capsize=4
        )
        axis.set_xticks(
            np.arange(len(native_projection)),
            native_projection.condition.str.replace("reference_", ""),
        )
        axis.set_ylabel(label)
    figure.tight_layout()
    figure.savefig(output / "figures/synthetic_composition_projection.png", dpi=220)
    plt.close(figure)

    inputs = [prediction_path, pair_path, metric_path, vector_path, eqv3_path, sevennet_path]
    manifest = pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path)} for path in inputs]
    )
    manifest.to_csv(output / "input_manifest.csv", index=False)
    audit = {
        "status": "PASS",
        "checks": {
            "bootstrap_draws": args.draws,
            "ec_bootstrap_seed": args.seed,
            "force_bootstrap_seed": args.seed + 1,
            "ec_cluster_unit": "whole exact-composition group",
            "ec_clusters": int(pairs.group_id.nunique()),
            "force_cluster_unit": "whole structure with all atoms/components",
            "force_clusters": int(vectors.id.nunique()),
            "seed_handling": "three seeds remain paired; endpoint is their arithmetic mean",
            "projection_primary": "native predictions before post-hoc calibration",
            "projection_definition": "uncentered beta=(u dot h)/(h dot h); Pearson is centered",
            "postcalibration_ec_cancellation_max_mev_atom": float(
                (
                    supplement_per_seed.raw_ec_mae_mev_atom
                    - supplement_per_seed.postcal_ec_mae_mev_atom
                ).abs().max()
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    ec_direct = bootstrap_summary[
        (bootstrap_summary.endpoint == "ec_drift")
        & (bootstrap_summary.comparison == "reference_direct_minus_reference_aligned")
    ].iloc[0]
    ec_background = bootstrap_summary[
        (bootstrap_summary.endpoint == "ec_drift")
        & (bootstrap_summary.comparison == "reference_aligned_minus_zero_direct")
    ].iloc[0]
    force_direct = bootstrap_summary[
        (bootstrap_summary.endpoint == "force_drift_rmse")
        & (bootstrap_summary.comparison == "reference_direct_minus_reference_aligned")
    ].iloc[0]
    force_background = bootstrap_summary[
        (bootstrap_summary.endpoint == "force_drift_rmse")
        & (bootstrap_summary.comparison == "reference_aligned_minus_zero_direct")
    ].iloc[0]
    report = [
        "# Final no-training post-processing",
        "",
        "All analyses use frozen predictions and forces. No model was trained, selected, or refit.",
        "",
        "## Paired cluster bootstrap",
        "",
        (
            f"EC drift direct − aligned: {ec_direct.point_difference:.3f} meV/atom "
            f"(95% whole-group bootstrap CI {ec_direct.ci95_low:.3f} to "
            f"{ec_direct.ci95_high:.3f}; 22 groups, 10,000 draws)."
        ),
        (
            f"EC drift aligned − zero: {ec_background.point_difference:.3f} meV/atom "
            f"(95% CI {ec_background.ci95_low:.3f} to {ec_background.ci95_high:.3f})."
        ),
        (
            f"Force drift direct − aligned: {force_direct.point_difference:.5f} eV/angstrom "
            f"(95% whole-structure bootstrap CI {force_direct.ci95_low:.5f} to "
            f"{force_direct.ci95_high:.5f}; 12 structures, 10,000 draws)."
        ),
        (
            f"Force drift aligned − zero: {force_background.point_difference:.5f} eV/angstrom "
            f"(95% CI {force_background.ci95_low:.5f} to "
            f"{force_background.ci95_high:.5f})."
        ),
        "",
        (
            "A confidence interval overlapping zero is reported as no detected excess over the "
            "zero-shift background, not as proof of equivalence."
        ),
        "",
        "## Composition-direction projection",
        "",
        "Primary values use native predictions before post-hoc composition calibration.",
        "",
        "| Condition | beta | ||u_perp|| / ||h|| | corr(u,h) | residual RMS (meV/atom) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in native_projection.itertuples(index=False):
        report.append(
            f"| {row.condition} | {row.beta_mean:.6f} ± {row.beta_std:.6f} | "
            f"{row.orthogonal_norm_ratio_mean:.6f} ± "
            f"{row.orthogonal_norm_ratio_std:.6f} | "
            f"{row.pearson_mean:.6f} ± {row.pearson_std:.6f} | "
            f"{1000 * row.residual_rms_ev_atom_mean:.3f} ± "
            f"{1000 * row.residual_rms_ev_atom_std:.3f} |"
        )
    report.extend(
        [
            "",
            (
                "The orthogonal component means not explained by the known reference vector; it "
                "may include other composition-dependent directions as well as structure-dependent "
                "change and is therefore not labelled a pure PES component."
            ),
            "",
            "## Raw and post-calibrated results",
            "",
            (
                "The complete per-seed and mean±SD tables are saved as CSV. EC-MAE is numerically "
                "unchanged by the pure composition post-calibration (maximum difference recorded "
                "in audit)."
            ),
            "",
            "## Reproducibility reference",
            "",
            (
                "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
                "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. "
                "arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065"
            ),
        ]
    )
    (output / "POSTPROCESSING_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
