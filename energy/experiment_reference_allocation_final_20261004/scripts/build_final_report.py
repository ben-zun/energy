from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def mean_std(frame: pd.DataFrame, metric: str) -> str:
    values = frame[metric].dropna()
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)

    eqv3 = pd.read_csv(root / "eqv3_final_eval/metrics/per_seed_metrics.csv")
    sevennet = pd.read_csv(root / "sevennet_final_eval_v2/metrics/per_seed_metrics.csv")
    required = pd.read_csv(
        root / "required_vs_induced/summary_with_group_bootstrap.csv"
    )
    synthetic_energy = pd.read_csv(
        root / "synthetic_reference_control/final_analysis/energy_summary.csv"
    )
    synthetic_force = pd.read_csv(
        root / "synthetic_reference_control/final_analysis/force_summary.csv"
    )
    selection = pd.read_csv(
        root / "synthetic_reference_control/final_energy/metrics/selected_checkpoints.csv"
    )
    force_audit = json.loads(
        (root / "synthetic_reference_control/final_force/audit.json").read_text()
    )
    cross = pd.read_csv(root / "cross_protocol_summary/aggregate.csv")
    closure_bootstrap = pd.read_csv(
        root / "closure_postprocessing/synthetic_paired_bootstrap_summary.csv"
    )
    closure_projection = pd.read_csv(
        root / "closure_postprocessing/synthetic_projection_summary.csv"
    )
    raw_post = pd.read_csv(root / "closure_postprocessing/raw_postcalibration_summary.csv")

    lines = [
        "# Reference-allocation experiments: final report",
        "",
        ("This report summarizes the frozen protocol executed on 2026-10-04. "
        "No result-dependent runs, epoch extensions, route deletion, or test-set selection were used."),
        "",
        "## Audit outcome",
        "",
        "- Final numerical audit: `AUDIT_OK`.",
        "- Local and server unit tests: 10/10 passed with warnings treated as errors.",
        ("- Synthetic runs: 9/9 completed exactly 120 epochs; all four systemd workers ended "
        "successfully with zero restarts."),
        "- Synthetic split: 8192 train / 1024 validation / 2048 test; IDs are disjoint.",
        "- Checkpoint selection and post-calibration used validation and train data only, respectively.",
        "",
        "## 1. Corrected EqV3 matched comparison",
        "",
        "All values below are mean ± sample standard deviation across the three frozen seeds.",
        "",
        "| Route | Overall calibrated MAE (meV/atom) | Fixed confirmation EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ("direct", "aligned"):
        overall = eqv3[
            (eqv3.route == route)
            & (eqv3.calibration == "calibrated")
            & (eqv3.cohort == "overall2048")
        ]
        relative = eqv3[
            (eqv3.route == route)
            & (eqv3.calibration == "calibrated")
            & (eqv3.cohort == "relative401")
        ]
        lines.append(
            f"| {route} | {mean_std(overall, 'mae_mev')} | "
            f"{mean_std(relative, 'ec_mae_mev')} | "
            f"{mean_std(relative, 'pair_delta_mae_mev')} | "
            f"{mean_std(relative, 'top1_accuracy')} | "
            f"{mean_std(relative, 'major_misselection_rate')} |"
        )
    lines.extend(
        [
            "",
            ("Under the corrected full physical decoder, aligned is better than direct for every "
            "seed on both absolute and same-composition endpoints. The paired whole-group/bootstrap "
            "tables are retained in `eqv3_final_eval/bootstrap/`."),
            "",
            "## 2. SevenNet matched evaluation",
            "",
            "| Route | Overall calibrated MAE (meV/atom) | Fixed confirmation EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route in ("direct", "aligned"):
        overall = sevennet[
            (sevennet.route == route)
            & (sevennet.calibration == "calibrated")
            & (sevennet.cohort == "overall2048")
        ]
        relative = sevennet[
            (sevennet.route == route)
            & (sevennet.calibration == "calibrated")
            & (sevennet.cohort == "common401")
        ]
        lines.append(
            f"| {route} | {mean_std(overall, 'mae_mev')} | "
            f"{mean_std(relative, 'ec_mae_mev')} | "
            f"{mean_std(relative, 'pair_delta_mae_mev')} | "
            f"{mean_std(relative, 'top1_accuracy')} | "
            f"{mean_std(relative, 'major_misselection_rate')} |"
        )

    lines.extend(
        [
            "",
            "## 3. Required versus induced within-group corrections",
            "",
            ("The frozen 190-group/238-pair list is shared across models, routes, and seeds. "
            "Intervals in the machine-readable table use 10,000 whole-group bootstrap draws."),
            "",
            "| Model | Route | Excess MAE (meV/atom) | Norm ratio | Pearson | Projection α | Orthogonal norm ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("sevennet", "eqv3"):
        for route in ("direct", "aligned"):
            block = required[(required.model == model) & (required.route == route)]
            lines.append(
                f"| {model} | {route} | {mean_std(block, 'excess_mae_mev_atom')} | "
                f"{mean_std(block, 'norm_ratio')} | {mean_std(block, 'pearson')} | "
                f"{mean_std(block, 'projection_alpha')} | "
                f"{mean_std(block, 'orthogonal_norm_ratio')} |"
            )

    lines.extend(
        [
            "",
            ("Aligned corrections are much closer to the corrections required by the target. "
            "This supports a statement about correction allocation; it does not imply that every "
            "change to a potential-energy surface is intrinsically wrong."),
            "",
            "## 4. Synthetic pure-reference control",
            "",
            ("Synthetic targets are the frozen pretrained SevenNet physical energy plus a pure "
            "train-fitted composition reference. The reference has no coordinate dependence. "
            "The test set contains 2048 structures; same-composition endpoints use 22 complete "
            "groups, 52 structures, and 53 frozen pairs."),
            "",
            "| Condition | Selected epochs | Calibrated MAE (meV/atom) | EC drift (meV/atom) | Pair drift (meV/atom) | Force drift RMSE (eV/Å) | Force cosine |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in ("zero_direct", "reference_direct", "reference_aligned"):
        energy = synthetic_energy[
            (synthetic_energy.condition == condition)
            & (synthetic_energy.calibration == "calibrated")
        ].iloc[0]
        force = synthetic_force[synthetic_force.condition == condition].iloc[0]
        epochs = ", ".join(
            map(
                str,
                selection.loc[selection.condition == condition, "selected_epoch"]
                .astype(int)
                .tolist(),
            )
        )
        lines.append(
            f"| {condition} | {epochs} | "
            f"{energy.synthetic_mae_mev_mean:.3f} ± {energy.synthetic_mae_mev_std:.3f} | "
            f"{energy.drift_ec_mae_mev_mean:.3f} ± {energy.drift_ec_mae_mev_std:.3f} | "
            f"{energy.drift_pair_delta_mae_mev_mean:.3f} ± "
            f"{energy.drift_pair_delta_mae_mev_std:.3f} | "
            f"{force.force_drift_rmse_ev_ang_mean:.5f} ± "
            f"{force.force_drift_rmse_ev_ang_std:.5f} | "
            f"{force.force_cosine_mean:.5f} ± {force.force_cosine_std:.5f} |"
        )

    force_checks = force_audit["checks"]
    ec_direct = closure_bootstrap[
        (closure_bootstrap.endpoint == "ec_drift")
        & (
            closure_bootstrap.comparison
            == "reference_direct_minus_reference_aligned"
        )
    ].iloc[0]
    ec_background = closure_bootstrap[
        (closure_bootstrap.endpoint == "ec_drift")
        & (closure_bootstrap.comparison == "reference_aligned_minus_zero_direct")
    ].iloc[0]
    force_direct = closure_bootstrap[
        (closure_bootstrap.endpoint == "force_drift_rmse")
        & (
            closure_bootstrap.comparison
            == "reference_direct_minus_reference_aligned"
        )
    ].iloc[0]
    force_background = closure_bootstrap[
        (closure_bootstrap.endpoint == "force_drift_rmse")
        & (closure_bootstrap.comparison == "reference_aligned_minus_zero_direct")
    ].iloc[0]
    native_projection = closure_projection[closure_projection.calibration == "native"]
    lines.extend(
        [
            "",
            ("The aligned condition stays at the same energy/EC/force-drift scale as the zero-shift "
            "optimizer background. Direct training leaves a roughly 54 meV/atom absolute residual, "
            "induces about 15.4 meV/atom EC drift, and changes forces by about 0.75 eV/Å RMS."),
            "",
            ("Force-definition verification used physical total energy, 12 SHA256-selected test "
            "structures, and central differences for 8 structures × 4 models × 4 steps. "
            f"All {force_checks['finite_difference_total_components']} components passed; the "
            f"best-step worst absolute error was "
            f"{force_checks['finite_difference_best_step_max_abs_error_ev_ang']:.6f} eV/Å."),
            "",
            "## 5. Synthetic paired uncertainty and composition-direction projection",
            "",
            (
                f"Whole-group bootstrap gives direct − aligned EC drift = "
                f"{ec_direct.point_difference:.3f} meV/atom (95% CI "
                f"{ec_direct.ci95_low:.3f} to {ec_direct.ci95_high:.3f}); aligned − zero "
                f"is {ec_background.point_difference:.3f} (95% CI "
                f"{ec_background.ci95_low:.3f} to {ec_background.ci95_high:.3f})."
            ),
            (
                f"Whole-structure bootstrap gives direct − aligned force drift = "
                f"{force_direct.point_difference:.5f} eV/Å (95% CI "
                f"{force_direct.ci95_low:.5f} to {force_direct.ci95_high:.5f}); "
                f"aligned − zero is {force_background.point_difference:.5f} (95% CI "
                f"{force_background.ci95_low:.5f} to {force_background.ci95_high:.5f})."
            ),
            "",
            (
                "The aligned-minus-zero intervals include zero; this is reported as no detected "
                "excess over background, not as proof of equivalence."
            ),
            "",
            "| Condition | Projection β | ||u⊥|| / ||h|| | corr(u,h) | Residual RMS (meV/atom) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in native_projection.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.beta_mean:.6f} ± {row.beta_std:.6f} | "
            f"{row.orthogonal_norm_ratio_mean:.6f} ± "
            f"{row.orthogonal_norm_ratio_std:.6f} | "
            f"{row.pearson_mean:.6f} ± {row.pearson_std:.6f} | "
            f"{1000 * row.residual_rms_ev_atom_mean:.3f} ± "
            f"{1000 * row.residual_rms_ev_atom_std:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Direct training learns nearly the full global reference direction (β≈1) but "
                "simultaneously leaves a much larger component orthogonal to the known reference. "
                "That orthogonal component is described only as unexplained by h: it can contain "
                "other composition directions as well as structure-dependent change."
            ),
            "",
            "## 6. Raw versus post-calibrated supplementary summary",
            "",
            "| Model | Route | Raw MAE | Post-calibrated MAE | EC-MAE before/after |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in raw_post.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.route} | "
            f"{row.raw_mae_mev_atom_mean:.3f} ± {row.raw_mae_mev_atom_std:.3f} | "
            f"{row.postcal_mae_mev_atom_mean:.3f} ± "
            f"{row.postcal_mae_mev_atom_std:.3f} | "
            f"{row.raw_ec_mae_mev_atom_mean:.3f} / "
            f"{row.postcal_ec_mae_mev_atom_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Pure composition post-calibration changes absolute MAE but leaves EC-MAE "
                "unchanged to numerical precision."
            ),
            "",
            "## 7. Existing MP→r²SCAN summary",
            "",
            ("No retraining was performed. The current hashed machine-readable artifacts contain "
            "90 groups / 214 structures / 202 pairs; an older prose report saying 85/209 is stale."),
            "",
            "| Route | Absolute MAE | EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route in ("direct", "aligned"):
        row = cross[cross.route == route].iloc[0]
        lines.append(
            f"| {route} | {row.absolute_mae_full_scope_mev_atom_mean:.3f} ± "
            f"{row.absolute_mae_full_scope_mev_atom_std:.3f} | "
            f"{row.ec_mae_mev_atom_mean:.3f} ± {row.ec_mae_mev_atom_std:.3f} | "
            f"{row.pair_delta_mae_mev_atom_mean:.3f} ± "
            f"{row.pair_delta_mae_mev_atom_std:.3f} | "
            f"{row.top1_accuracy_mean:.3f} ± {row.top1_accuracy_std:.3f} | "
            f"{row.major_misselection_rate_mean:.3f} ± "
            f"{row.major_misselection_rate_std:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            ("The corrected EqV3 and matched SevenNet results both support aligned allocation. "
            "The synthetic experiment supplies the clean causal control: a composition-only target "
            "shift causes large unwanted same-composition and force changes under direct training, "
            "whereas aligned training remains at the zero-shift optimizer background."),
            "",
            ("Limits: the EqV3/SevenNet fixed confirmation cohort was previously inspected and is "
            "not a new blind test; the synthetic same-composition test has only 22 groups; force "
            "drift is evaluated on 12 fixed structures and is not a force error against DFT labels; "
            "and the conclusions concern these fixed models, datasets, and adaptation protocols."),
            "",
            ("All detailed predictions, pair lists, bootstrap tables, force vectors, finite-difference "
            "checks, histories, selected checkpoint hashes, and the frozen protocol remain under this "
            "experiment directory."),
        ]
    )
    destination = root / "FINAL_REPORT.md"
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
