from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read
from sevenn.calculator import SevenNetCalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ids(ids: list[str], count: int) -> list[str]:
    return sorted(ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:count]


def calculator_for_checkpoint(checkpoint: Path | None, device: str) -> SevenNetCalculator:
    calculator = SevenNetCalculator("7net-0", device=device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        calculator.model.load_state_dict(payload["model_state"], strict=True)
        calculator.model.to(device).eval()
    return calculator


def energy_force(calculator: SevenNetCalculator, structure_path: str) -> tuple[float, np.ndarray]:
    atoms = read(structure_path)
    atoms.calc = calculator
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces(), dtype=np.float64)


def finite_difference(
    calculator: SevenNetCalculator,
    structure_path: str,
    atom_index: int,
    axis: int,
    displacement: float,
) -> float:
    atoms = read(structure_path)
    energies: list[float] = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        positions = displaced.get_positions()
        positions[atom_index, axis] += sign * displacement
        displaced.set_positions(positions)
        displaced.calc = calculator
        energies.append(float(displaced.get_potential_energy()))
    return -(energies[0] - energies[1]) / (2.0 * displacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    energy = args.energy_root.resolve(strict=True)
    selection_path = args.selection.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(selection_path, keep_default_na=False)
    expected = {(condition, seed) for condition in ("zero_direct", "reference_direct", "reference_aligned")
                for seed in (20260528, 20260529, 20260530)}
    observed = set(zip(selection.condition, selection.seed.astype(int), strict=True))
    if observed != expected or len(selection) != 9:
        raise AssertionError("force evaluation requires all nine frozen synthetic selections")
    metadata_path = (
        energy
        / "runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision"
        / "fair_finetune_293/graph_cache/medium/graph_metadata.csv"
    )
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    cohort = metadata[metadata.split == "test"].copy()
    if cohort.id.duplicated().any() or len(cohort) != 2048:
        raise AssertionError("unexpected synthetic test cohort")
    if not cohort.cif_path.map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("one or more frozen synthetic test CIFs are unavailable")
    probe_ids = stable_ids(cohort.id.tolist(), 12)
    finite_difference_ids = probe_ids[:8]
    probe = cohort.set_index("id").loc[probe_ids]
    base_calculator = calculator_for_checkpoint(None, args.device)
    base: dict[str, tuple[float, np.ndarray]] = {
        structure_id: energy_force(base_calculator, row.cif_path)
        for structure_id, row in probe.iterrows()
    }

    vector_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    finite_rows: list[dict[str, object]] = []
    calculators: dict[str, SevenNetCalculator] = {"pretrained": base_calculator}
    first_seed_checkpoints: dict[str, Path] = {}
    for row in selection.itertuples(index=False):
        checkpoint = Path(row.checkpoint).resolve(strict=True)
        if sha256(checkpoint) != str(row.checkpoint_sha256):
            raise AssertionError(f"selected checkpoint hash changed: {checkpoint}")
        calculator = calculator_for_checkpoint(checkpoint, args.device)
        if int(row.seed) == 20260528:
            calculators[str(row.condition)] = calculator
            first_seed_checkpoints[str(row.condition)] = checkpoint
        differences: list[np.ndarray] = []
        base_vectors: list[np.ndarray] = []
        adapted_vectors: list[np.ndarray] = []
        for structure_id, structure in probe.iterrows():
            _, base_force = base[structure_id]
            _, adapted_force = energy_force(calculator, structure.cif_path)
            if base_force.shape != adapted_force.shape:
                raise AssertionError("base/adapted force shapes differ")
            differences.append(adapted_force - base_force)
            base_vectors.append(base_force)
            adapted_vectors.append(adapted_force)
            for atom_index in range(len(base_force)):
                vector_rows.append(
                    {
                        "id": structure_id,
                        "condition": row.condition,
                        "seed": int(row.seed),
                        "selected_epoch": int(row.selected_epoch),
                        "atom_index": atom_index,
                        **{
                            f"base_f{axis}": float(base_force[atom_index, dimension])
                            for axis, dimension in zip(("x", "y", "z"), range(3), strict=True)
                        },
                        **{
                            f"adapted_f{axis}": float(adapted_force[atom_index, dimension])
                            for axis, dimension in zip(("x", "y", "z"), range(3), strict=True)
                        },
                    }
                )
        difference = np.concatenate(differences)
        base_vector = np.concatenate(base_vectors).ravel()
        adapted_vector = np.concatenate(adapted_vectors).ravel()
        summary_rows.append(
            {
                "condition": row.condition,
                "seed": int(row.seed),
                "selected_epoch": int(row.selected_epoch),
                "n_structures": len(probe),
                "n_atoms": int(sum(len(values) for values in base_vectors)),
                "force_drift_rmse_ev_ang": float(np.sqrt(np.mean(difference**2))),
                "force_drift_mae_ev_ang": float(np.mean(np.abs(difference))),
                "base_force_rms_ev_ang": float(np.sqrt(np.mean(base_vector**2))),
                "adapted_force_rms_ev_ang": float(np.sqrt(np.mean(adapted_vector**2))),
                "force_cosine": float(
                    np.dot(base_vector, adapted_vector)
                    / (np.linalg.norm(base_vector) * np.linalg.norm(adapted_vector) + 1e-30)
                ),
                "checkpoint": str(checkpoint),
            }
        )

    # Frozen before seeing these results.  Float32 total energies suffer from
    # cancellation at very small displacements, so span a decade and accept
    # the best converged central difference for each scalar force component.
    steps = (1e-2, 5e-3, 2e-3, 1e-3)
    for model_name, calculator in calculators.items():
        for index, structure_id in enumerate(finite_difference_ids):
            structure = probe.loc[structure_id]
            _, analytic = energy_force(calculator, structure.cif_path)
            atom_index = 0
            axis = index % 3
            analytic_value = float(analytic[atom_index, axis])
            for step in steps:
                numeric = finite_difference(
                    calculator,
                    structure.cif_path,
                    atom_index,
                    axis,
                    step,
                )
                finite_rows.append(
                    {
                        "model": model_name,
                        "seed": "pretrained" if model_name == "pretrained" else 20260528,
                        "id": structure_id,
                        "atom_index": atom_index,
                        "axis": axis,
                        "step_angstrom": step,
                        "analytic_force_ev_ang": analytic_value,
                        "finite_difference_force_ev_ang": numeric,
                        "absolute_error_ev_ang": abs(numeric - analytic_value),
                        "relative_error": abs(numeric - analytic_value) / (abs(analytic_value) + 1e-12),
                        "reference_term_force_contribution_ev_ang": 0.0,
                    }
                )

    vector_frame = pd.DataFrame(vector_rows)
    summary_frame = pd.DataFrame(summary_rows)
    finite_frame = pd.DataFrame(finite_rows)
    vector_frame.to_csv(output / "force_vectors.csv", index=False)
    summary_frame.to_csv(output / "force_drift_summary.csv", index=False)
    finite_frame.to_csv(output / "finite_difference_checks.csv", index=False)
    convergence = finite_frame.groupby(["model", "id"]).absolute_error_ev_ang.agg(["min", "max"])
    best_rows = finite_frame.loc[
        finite_frame.groupby(["model", "id"]).absolute_error_ev_ang.idxmin()
    ].copy()
    absolute_tolerance_ev_ang = 2e-2
    relative_tolerance = 1e-1
    best_rows["allowed_error_ev_ang"] = np.maximum(
        absolute_tolerance_ev_ang,
        relative_tolerance * best_rows.analytic_force_ev_ang.abs(),
    )
    best_rows["within_tolerance"] = (
        best_rows.absolute_error_ev_ang <= best_rows.allowed_error_ev_ang
    )
    finite_difference_pass = bool(best_rows.within_tolerance.all())
    finite_values_pass = bool(np.isfinite(finite_frame.absolute_error_ev_ang).all())
    audit = {
        "status": "PASS" if finite_values_pass and finite_difference_pass else "FAIL",
        "checks": {
            "probe_selection": "12 smallest SHA256(id); independent of labels and outcomes",
            "probe_population": "frozen synthetic medium-split test IDs (n=2048)",
            "metadata_path": str(metadata_path),
            "force_probe_structures": len(probe_ids),
            "finite_difference_structures": len(finite_difference_ids),
            "finite_difference_steps_angstrom": list(steps),
            "models_finite_differenced": sorted(calculators),
            "finite_difference_max_abs_error_ev_ang": float(convergence["max"].max()),
            "finite_difference_best_step_max_abs_error_ev_ang": float(convergence["min"].max()),
            "finite_difference_acceptance": (
                "for every model/structure scalar component, the best tested step must have "
                "absolute_error <= max(0.02 eV/angstrom, 0.10*abs(analytic_force))"
            ),
            "finite_difference_absolute_tolerance_ev_ang": absolute_tolerance_ev_ang,
            "finite_difference_relative_tolerance": relative_tolerance,
            "finite_difference_passed_components": int(best_rows.within_tolerance.sum()),
            "finite_difference_total_components": len(best_rows),
            "energy_for_force": "SevenNet physical total energy from calculator; composition reference is coordinate independent",
            "first_seed_checkpoints": {key: str(value) for key, value in first_seed_checkpoints.items()},
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise AssertionError(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
