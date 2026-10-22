from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from reference_allocation.required_vs_induced import correction_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve(strict=True)
    pair_path = root / "within_group_corrections.csv"
    summary_path = root / "summary_with_group_bootstrap.csv"
    pairs = pd.read_csv(pair_path, keep_default_na=False)
    summary = pd.read_csv(summary_path, keep_default_na=False)

    keys = ["model", "seed", "route"]
    if pairs.duplicated(keys + ["group_id", "id_i", "id_j"]).any():
        raise AssertionError("duplicate model/seed/route/pair rows")
    rows: list[dict[str, object]] = []
    for (model, seed), block in pairs.groupby(["model", "seed"], sort=True):
        identity = block[block.route == "direct"]
        if len(identity) == 0:
            raise AssertionError("missing direct rows for pretrained baseline")
        baseline = correction_summary(
            identity.r_target.to_numpy(np.float64),
            np.zeros(len(identity), dtype=np.float64),
        )
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "route": "pretrained",
                "excess_mae_mev_atom": 1000 * baseline["excess_mae"],
                "norm_ratio": baseline["norm_ratio"],
                "pearson": np.nan,
                "projection_alpha": baseline["projection_alpha"],
            }
        )
    columns = [
        "model",
        "seed",
        "route",
        "excess_mae_mev_atom",
        "norm_ratio",
        "pearson",
        "projection_alpha",
    ]
    rows.extend(summary[columns].to_dict("records"))
    comparison = pd.DataFrame(rows)[columns]
    comparison.to_csv(root / "required_vs_induced_metric_comparison.csv", index=False)

    import matplotlib.pyplot as plt

    metrics = (
        ("excess_mae_mev_atom", "excess MAE (meV/atom)"),
        ("norm_ratio", "norm ratio"),
        ("pearson", "Pearson correlation"),
        ("projection_alpha", "projection alpha"),
    )
    routes = ("pretrained", "direct", "aligned")
    colors = {"pretrained": "#666666", "direct": "#D55E00", "aligned": "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for model_index, model in enumerate(("sevennet", "eqv3")):
            for route_index, route in enumerate(routes):
                values = comparison.loc[
                    (comparison.model == model) & (comparison.route == route), metric
                ].to_numpy(np.float64)
                finite = values[np.isfinite(values)]
                position = model_index * 4 + route_index
                if finite.size == 0:
                    axis.text(position, 0.0, "N/A", ha="center", va="bottom", fontsize=7)
                    continue
                mean = float(finite.mean())
                standard_deviation = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                axis.errorbar(
                    position,
                    mean,
                    yerr=standard_deviation,
                    marker="o",
                    color=colors[route],
                    capsize=3,
                    linestyle="none",
                    label=route if model_index == 0 else None,
                )
        axis.set_xticks((1, 5), ("SevenNet", "EqV3"))
        axis.set_ylabel(label)
        axis.axhline(0.0, color="#BBBBBB", linewidth=0.7)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    destination = root / "figures/required_vs_induced_metrics.png"
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
