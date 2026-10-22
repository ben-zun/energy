from __future__ import annotations

import numpy as np
import pandas as pd


def fold_intercept_into_element_coefficients(mu: np.ndarray, bias: float) -> np.ndarray:
    """Convert d(x)=-(x·mu+bias) to d(x)=x·delta_mu using sum(x)=1."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (118,) or not np.isfinite(mu).all() or not np.isfinite(bias):
        raise ValueError("expected 118 finite element coefficients and a finite bias")
    return -(mu + float(bias))


def composition_shift_per_atom(fractions: np.ndarray, delta_mu: np.ndarray) -> np.ndarray:
    fractions = np.asarray(fractions, dtype=np.float64)
    delta_mu = np.asarray(delta_mu, dtype=np.float64)
    if fractions.ndim != 2 or fractions.shape[1] != 118 or delta_mu.shape != (118,):
        raise ValueError("fractions must be n×118 and delta_mu must have length 118")
    if not np.isfinite(fractions).all() or not np.isfinite(delta_mu).all():
        raise ValueError("nonfinite composition input")
    if np.any(fractions < -1e-12):
        raise ValueError("negative atomic fraction")
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-10, rtol=0)
    return fractions @ delta_mu


def build_synthetic_targets(
    pretrained_per_atom: np.ndarray,
    fractions: np.ndarray,
    delta_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pretrained = np.asarray(pretrained_per_atom, dtype=np.float64)
    if pretrained.ndim != 1 or pretrained.shape[0] != fractions.shape[0]:
        raise ValueError("pretrained predictions and fractions are not row-aligned")
    if not np.isfinite(pretrained).all():
        raise ValueError("nonfinite pretrained prediction")
    shift = composition_shift_per_atom(fractions, delta_mu)
    return pretrained.copy(), pretrained + shift


def audit_synthetic_targets(
    *,
    pretrained_per_atom: np.ndarray,
    target_reference_per_atom: np.ndarray,
    fractions: np.ndarray,
    natoms: np.ndarray,
    delta_mu: np.ndarray,
    composition_keys: np.ndarray,
) -> dict[str, float | int]:
    pretrained = np.asarray(pretrained_per_atom, dtype=np.float64)
    target = np.asarray(target_reference_per_atom, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    natoms = np.asarray(natoms, dtype=np.float64)
    keys = np.asarray(composition_keys).astype(str)
    shift = composition_shift_per_atom(fractions, delta_mu)
    counts = fractions * natoms[:, None]
    integer_count_error = float(np.max(np.abs(counts - np.rint(counts))))
    total_from_per_atom = natoms * (target - pretrained)
    total_from_counts = np.rint(counts) @ np.asarray(delta_mu, dtype=np.float64)
    total_identity_error = float(np.max(np.abs(total_from_per_atom - total_from_counts)))
    shift_frame = pd.DataFrame({"shift": shift, "group": keys})
    group_spread = shift_frame.groupby("group")["shift"].agg(
        lambda values: values.max() - values.min()
    )
    max_group_spread = float(group_spread.max())
    if integer_count_error > 1e-6:
        raise AssertionError(f"noninteger composition counts: {integer_count_error}")
    if total_identity_error > 1e-8:
        raise AssertionError(f"total/per-atom synthetic shift mismatch: {total_identity_error}")
    if max_group_spread > 1e-10:
        raise AssertionError(f"same-composition shift mismatch: {max_group_spread}")
    return {
        "n": len(pretrained),
        "integer_count_max_abs": integer_count_error,
        "total_shift_identity_max_abs_ev": total_identity_error,
        "same_composition_shift_max_spread_ev_atom": max_group_spread,
        "oracle_energy_max_abs_ev_atom": float(np.max(np.abs(target - pretrained - shift))),
    }
