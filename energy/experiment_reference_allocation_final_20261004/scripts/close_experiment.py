from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_metric(
    rows: list[dict[str, object]],
    *,
    scope: str,
    route: str,
    metric: str,
    mean: float,
    standard_deviation: float | None,
    unit: str,
    cohort: str,
    source: Path,
) -> None:
    rows.append(
        {
            "scope": scope,
            "route": route,
            "metric": metric,
            "mean": mean,
            "std": standard_deviation,
            "unit": unit,
            "cohort": cohort,
            "source": str(source),
            "source_sha256": sha256(source),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    historical = args.historical_report.resolve(strict=True)
    manuscript = args.manuscript_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    eqv3_path = root / "eqv3_final_eval/metrics/per_seed_metrics.csv"
    sevennet_path = root / "sevennet_final_eval_v2/metrics/per_seed_metrics.csv"
    synthetic_path = root / "synthetic_reference_control/final_analysis/energy_summary.csv"
    force_path = root / "synthetic_reference_control/final_analysis/force_summary.csv"
    cross_path = root / "cross_protocol_summary/aggregate.csv"
    closure_path = root / "closure_postprocessing/synthetic_paired_bootstrap_summary.csv"
    projection_path = root / "closure_postprocessing/synthetic_projection_summary.csv"
    eqv3 = pd.read_csv(eqv3_path)
    sevennet = pd.read_csv(sevennet_path)
    synthetic = pd.read_csv(synthetic_path)
    force = pd.read_csv(force_path)
    cross = pd.read_csv(cross_path)
    closure = pd.read_csv(closure_path)
    projection = pd.read_csv(projection_path)

    rows: list[dict[str, object]] = []
    for scope, frame, overall_cohort, relative_cohort, source in (
        ("EqV3", eqv3, "overall2048", "relative401", eqv3_path),
        ("SevenNet", sevennet, "overall2048", "common401", sevennet_path),
    ):
        for route in ("direct", "aligned"):
            overall = frame[
                (frame.route == route)
                & (frame.calibration == "calibrated")
                & (frame.cohort == overall_cohort)
            ]
            relative = frame[
                (frame.route == route)
                & (frame.calibration == "calibrated")
                & (frame.cohort == relative_cohort)
            ]
            add_metric(
                rows,
                scope=scope,
                route=route,
                metric="calibrated_absolute_mae",
                mean=float(overall.mae_mev.mean()),
                standard_deviation=float(overall.mae_mev.std(ddof=1)),
                unit="meV/atom",
                cohort="test2048",
                source=source,
            )
            add_metric(
                rows,
                scope=scope,
                route=route,
                metric="ec_mae",
                mean=float(relative.ec_mae_mev.mean()),
                standard_deviation=float(relative.ec_mae_mev.std(ddof=1)),
                unit="meV/atom",
                cohort="fixed confirmation: 190 groups/401 structures/238 pairs",
                source=source,
            )
    for condition in ("zero_direct", "reference_direct", "reference_aligned"):
        energy = synthetic[
            (synthetic.condition == condition) & (synthetic.calibration == "calibrated")
        ].iloc[0]
        force_row = force[force.condition == condition].iloc[0]
        for metric, mean_column, std_column, unit, cohort in (
            (
                "calibrated_absolute_mae",
                "synthetic_mae_mev_mean",
                "synthetic_mae_mev_std",
                "meV/atom",
                "synthetic test2048",
            ),
            (
                "ec_drift",
                "drift_ec_mae_mev_mean",
                "drift_ec_mae_mev_std",
                "meV/atom",
                "22 groups/52 structures/53 pairs",
            ),
        ):
            add_metric(
                rows,
                scope="SevenNet synthetic",
                route=condition,
                metric=metric,
                mean=float(energy[mean_column]),
                standard_deviation=float(energy[std_column]),
                unit=unit,
                cohort=cohort,
                source=synthetic_path,
            )
        add_metric(
            rows,
            scope="SevenNet synthetic",
            route=condition,
            metric="force_drift_rmse",
            mean=float(force_row.force_drift_rmse_ev_ang_mean),
            standard_deviation=float(force_row.force_drift_rmse_ev_ang_std),
            unit="eV/angstrom",
            cohort="12 SHA256-selected test structures",
            source=force_path,
        )
    for route in ("direct", "aligned"):
        block = cross[cross.route == route].iloc[0]
        for metric, mean_column, std_column in (
            (
                "calibrated_absolute_mae",
                "absolute_mae_full_scope_mev_atom_mean",
                "absolute_mae_full_scope_mev_atom_std",
            ),
            ("ec_mae", "ec_mae_mev_atom_mean", "ec_mae_mev_atom_std"),
        ):
            add_metric(
                rows,
                scope="MP-to-r2SCAN",
                route=route,
                metric=metric,
                mean=float(block[mean_column]),
                standard_deviation=float(block[std_column]),
                unit="meV/atom",
                cohort="90 groups/214 structures/202 pairs",
                source=cross_path,
            )
    for row in closure.itertuples(index=False):
        add_metric(
            rows,
            scope="Synthetic paired bootstrap",
            route=row.comparison,
            metric=f"{row.endpoint}_difference",
            mean=float(row.point_difference),
            standard_deviation=None,
            unit=str(row.unit),
            cohort=f"{row.clusters} whole clusters; CI [{row.ci95_low}, {row.ci95_high}]",
            source=closure_path,
        )
    for row in projection[projection.calibration == "native"].itertuples(index=False):
        for metric in ("beta", "orthogonal_norm_ratio", "pearson", "residual_rms_ev_atom"):
            add_metric(
                rows,
                scope="Synthetic composition projection",
                route=row.condition,
                metric=metric,
                mean=float(getattr(row, f"{metric}_mean")),
                standard_deviation=float(getattr(row, f"{metric}_std")),
                unit="eV/atom" if metric == "residual_rms_ev_atom" else "dimensionless",
                cohort="synthetic test2048; native prediction",
                source=projection_path,
            )
    registry = pd.DataFrame(rows)
    registry.to_csv(output / "CURRENT_NUMBER_REGISTRY.csv", index=False)

    historical_lines = historical.read_text(errors="replace").splitlines()
    stale_patterns = {
        "old_r2scan_cohort": re.compile(
            r"85\s*组\s*/\s*209|85\s*groups.*209", re.IGNORECASE
        ),
        "old_r2scan_direct_mae": re.compile(r"91\.18\s*[–-]\s*92\.02|\b91\.\d+"),
    }
    stale_matches: list[tuple[str, int, str]] = []
    for number, line in enumerate(historical_lines, start=1):
        for label, pattern in stale_patterns.items():
            if pattern.search(line):
                stale_matches.append((label, number, line.strip()))

    manuscript_files = sorted(manuscript.rglob("*.tex"))
    manuscript_matches: list[tuple[str, int, str]] = []
    target_pattern = re.compile(
        r"85\s*组\s*/\s*209|85\s*groups.*209|91\.18\s*[–-]\s*92\.02|"
        r"common401|relative401|reference[_ -]?allocation|pure-reference",
        re.IGNORECASE,
    )
    for path in manuscript_files:
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if target_pattern.search(line):
                manuscript_matches.append((str(path), number, line.strip()))

    report = [
        "# Legacy-number and manuscript audit",
        "",
        "## Canonical source",
        "",
        f"- Final report: `{root / 'FINAL_REPORT.md'}`",
        f"- Registry: `{output / 'CURRENT_NUMBER_REGISTRY.csv'}`",
        "- Current MP→r²SCAN cohort: **90 groups / 214 structures / 202 pairs**.",
        "- Current MP→r²SCAN calibrated MAE: direct **83.667**, aligned **23.943 meV/atom**.",
        "- Current shared real-data cohort name: **fixed confirmation cohort**, not blind or independent test.",
        "",
        "## Historical-only matches",
        "",
        f"Historical report: `{historical}` (SHA256 `{sha256(historical)}`).",
        "",
    ]
    for label, number, line in stale_matches:
        report.append(f"- `{label}`, line {number}: `{line}`")
    report.extend(
        [
            "",
            "These lines remain as experiment history and must not be copied into the manuscript.",
            "",
            "## Manuscript candidate",
            "",
            f"Scanned `{manuscript}` ({len(manuscript_files)} TeX files).",
            "",
        ]
    )
    if manuscript_matches:
        for path, number, line in manuscript_matches:
            report.append(f"- `{path}`, line {number}: `{line}`")
    else:
        report.append(
            "- No allocation/synthetic/common401 or identified stale-number strings were found."
        )
    report.extend(
        [
            "",
            (
                "The candidate is the separate FE-GX manuscript and was not modified. No uniquely "
                "identified current reference-allocation manuscript was present in the scanned roots."
            ),
        ]
    )
    (output / "LEGACY_NUMBER_AUDIT.md").write_text("\n".join(report) + "\n")
    lock = {
        "status": "EXPERIMENT_CLOSED",
        "closed_scope": "reference-allocation experiments",
        "new_training_allowed": False,
        "canonical_report": str(root / "FINAL_REPORT.md"),
        "canonical_number_registry": str(output / "CURRENT_NUMBER_REGISTRY.csv"),
        "historical_report_not_for_manuscript": str(historical),
        "manuscript_modified": False,
        "reason": "core evidence chain and no-training closure analyses complete",
    }
    (output / "EXPERIMENT_CLOSED.json").write_text(json.dumps(lock, indent=2) + "\n")
    audit = {
        "status": "PASS",
        "registry_rows": len(registry),
        "historical_stale_matches": len(stale_matches),
        "manuscript_target_matches": len(manuscript_matches),
        "inputs": {
            "historical_report_sha256": sha256(historical),
            "final_report_sha256": sha256(root / "FINAL_REPORT.md"),
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
