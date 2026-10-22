from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = source / "per_seed_metrics.csv"
    cleaning_path = source / "cleaning_report.json"
    audit_path = source / "consistency_audit.json"
    metrics = pd.read_csv(metrics_path, keep_default_na=False)
    cleaning = json.loads(cleaning_path.read_text())
    consistency = json.loads(audit_path.read_text())
    block = metrics[(metrics.cohort == "r2scan") & (metrics.scope == "dedup")].copy()
    block = block[block.method.str.match(r"^(U|A)_202605(28|29|30)$")]
    if len(block) != 6:
        raise AssertionError("expected three frozen U and three frozen A r2SCAN rows")
    expected = cleaning["r2scan"]["scopes"]["dedup"]
    for column, key in (("groups", "groups"), ("structures", "structures"), ("pairs", "pairs")):
        if set(block[column].astype(int)) != {int(expected[key])}:
            raise AssertionError(f"r2SCAN {column} differs between metrics and cleaning report")
    if consistency.get("n_failed") != 0:
        raise AssertionError("upstream consistency audit reports failures")
    output_rows: list[dict[str, object]] = []
    for row in block.itertuples(index=False):
        source_route, seed_text = row.method.split("_")
        output_rows.append(
            {
                "model": "SevenNet",
                "protocol": "MP_to_r2SCAN",
                "seed": int(seed_text),
                "route": "direct" if source_route == "U" else "aligned",
                "scope": "dedup",
                "groups": int(row.groups),
                "structures": int(row.structures),
                "pairs": int(row.pairs),
                "absolute_mae_full_scope_mev_atom": float(row.abs_mae_full_scope_mev),
                "absolute_mae_multigroup_mev_atom": float(row.abs_mae_multigroup_mev),
                "ec_mae_mev_atom": float(row.ec_mae_mev),
                "pair_delta_mae_mev_atom": float(row.pair_mae_mev),
                "top1_accuracy": float(row.exact_hit_rate),
                "major_misselection_rate": float(row.wrong_pick_rate_gt_10mev),
            }
        )
    summary = pd.DataFrame(output_rows).sort_values(["route", "seed"])
    summary.to_csv(output / "summary.csv", index=False)
    aggregate = summary.groupby("route").agg(
        {
            "absolute_mae_full_scope_mev_atom": ["mean", "std"],
            "ec_mae_mev_atom": ["mean", "std"],
            "pair_delta_mae_mev_atom": ["mean", "std"],
            "top1_accuracy": ["mean", "std"],
            "major_misselection_rate": ["mean", "std"],
        }
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate.csv", index=False)
    report = {
        "status": "PASS",
        "source_hashes": {
            "per_seed_metrics.csv": sha256(metrics_path),
            "cleaning_report.json": sha256(cleaning_path),
            "consistency_audit.json": sha256(audit_path),
        },
        "checks": {
            "upstream_consistency_failed": consistency.get("n_failed"),
            "dedup_groups": int(expected["groups"]),
            "dedup_structures": int(expected["structures"]),
            "dedup_pairs": int(expected["pairs"]),
            "new_training_performed": False,
        },
        "stale_report_warning": (
            "UNIFIED_ANALYSIS_REPORT.md states 85 groups/209 structures, while current machine-readable "
            "cleaning_report and per_seed_metrics agree on 90 groups/214 structures. This summary uses "
            "the current hashed machine-readable artifacts and does not copy the stale prose counts."
        ),
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
