from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reference_allocation.required_vs_induced import correction_summary

SEEDS = (20260528, 20260529, 20260530)
ROUTES = ("direct", "aligned")
DIMENSIONAL = {"excess_mae", "target_rms", "induced_rms"}


def bootstrap_summary(
    frame: pd.DataFrame,
    route: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    groups = sorted(frame.group_id.unique())
    rows = {group: frame.index[frame.group_id == group].to_numpy() for group in groups}
    random = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {}
    point = correction_summary(frame.r_target.to_numpy(), frame[f"r_{route}"].to_numpy())
    for _ in range(draws):
        selected = random.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([rows[groups[index]] for index in selected])
        result = correction_summary(
            frame.r_target.to_numpy()[indices], frame[f"r_{route}"].to_numpy()[indices]
        )
        for metric, value in result.items():
            if metric != "n_pairs":
                distributions.setdefault(metric, []).append(float(value))
    return {
        metric: (
            float(point[metric]),
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        )
        for metric, values in distributions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqv3-root", type=Path, required=True)
    parser.add_argument("--sevennet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    roots = {
        "eqv3": args.eqv3_root.resolve(strict=True),
        "sevennet": args.sevennet_root.resolve(strict=True),
    }
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    (output / "figures").mkdir(parents=True, exist_ok=True)

    all_pairs: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    identity: pd.DataFrame | None = None
    for model, root in roots.items():
        for seed_value in SEEDS:
            for route in ROUTES:
                path = root / "metrics" / f"required_vs_induced_seed{seed_value}_{route}.csv"
                frame = pd.read_csv(path, keep_default_na=False)
                keys = frame[["group_id", "id_i", "id_j"]]
                if identity is None:
                    identity = keys
                elif not identity.equals(keys):
                    raise AssertionError("models/routes do not use the same frozen pair list")
                frame["model"] = model
                frame["seed"] = seed_value
                frame["route"] = route
                all_pairs.append(frame)
                interval = bootstrap_summary(
                    frame, route, draws=args.draws, seed=args.seed + seed_value
                )
                record: dict[str, object] = {
                    "model": model,
                    "seed": seed_value,
                    "route": route,
                    "groups": int(frame.group_id.nunique()),
                    "pairs": len(frame),
                }
                for metric, (point, low, high) in interval.items():
                    scale = 1000.0 if metric in DIMENSIONAL else 1.0
                    suffix = "_mev_atom" if metric in DIMENSIONAL else ""
                    record[f"{metric}{suffix}"] = point * scale
                    record[f"{metric}{suffix}_ci95_low"] = low * scale
                    record[f"{metric}{suffix}_ci95_high"] = high * scale
                rows.append(record)
    combined = pd.concat(all_pairs, ignore_index=True)
    combined.to_csv(output / "within_group_corrections.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_with_group_bootstrap.csv", index=False)

    plot_status = "not_attempted"
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=False, sharey=False)
        for axis, (model, route) in zip(
            axes.ravel(),
            (("sevennet", "direct"), ("sevennet", "aligned"),
             ("eqv3", "direct"), ("eqv3", "aligned")),
            strict=True,
        ):
            block = combined[(combined.model == model) & (combined.route == route)]
            for seed_value in SEEDS:
                subset = block[block.seed == seed_value]
                axis.scatter(
                    1000 * subset.r_target,
                    1000 * subset[f"r_{route}"],
                    s=8,
                    alpha=0.35,
                    label=str(seed_value),
                )
            limits = np.array([axis.get_xlim(), axis.get_ylim()])
            lower, upper = float(limits.min()), float(limits.max())
            axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_title(f"{model} — {route}")
            axis.set_xlabel("required correction (meV/atom)")
            axis.set_ylabel("induced correction (meV/atom)")
        axes[0, 0].legend(frameon=False, fontsize=7)
        figure.tight_layout()
        figure.savefig(output / "figures/required_vs_induced.png", dpi=220)
        plt.close(figure)
        plot_status = "written"
    except ImportError:
        plot_status = "matplotlib_unavailable"
    audit = {
        "status": "PASS",
        "checks": {
            "models": sorted(roots),
            "routes": list(ROUTES),
            "seeds": list(SEEDS),
            "groups": int(identity.group_id.nunique()) if identity is not None else 0,
            "pairs": len(identity) if identity is not None else 0,
            "bootstrap_draws": args.draws,
            "bootstrap_seed_base": args.seed,
            "pair_list_shared_by_all_models_routes": True,
            "plot_status": plot_status,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
