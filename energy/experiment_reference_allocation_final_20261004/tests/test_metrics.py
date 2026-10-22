from __future__ import annotations

import unittest

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
from reference_allocation.synthetic import (
    audit_synthetic_targets,
    build_synthetic_targets,
    fold_intercept_into_element_coefficients,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "id": ["a2", "b1", "a1", "b2"],
                "composition_key": ["A2", "B2", "A2", "B2"],
                "truth": [0.02, 0.20, 0.00, 0.10],
                "prediction": [0.03, 0.12, 0.01, 0.22],
            }
        )
        self.pairs = build_pairs(self.frame)

    def test_pair_list_is_deterministic_under_row_shuffle(self) -> None:
        shuffled = self.frame.sample(frac=1.0, random_state=9)
        pd.testing.assert_frame_equal(self.pairs, build_pairs(shuffled))

    def test_composition_only_offsets_cancel(self) -> None:
        before = same_composition_metrics(self.frame, self.pairs)
        shifted = self.frame.copy()
        shifted["prediction"] += shifted.composition_key.map({"A2": 4.0, "B2": -3.0})
        after = same_composition_metrics(shifted, self.pairs)
        for key in ("ec_mae", "pair_delta_mae", "pair_accuracy", "top1_accuracy"):
            self.assertAlmostEqual(before[key], after[key], places=12)

    def test_absolute_metrics_sign_and_quantiles(self) -> None:
        result = absolute_metrics([0.0, 1.0], [0.2, 0.6])
        self.assertAlmostEqual(result["mae"], 0.3)
        self.assertAlmostEqual(result["bias"], -0.1)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.1))

    def test_required_vs_induced_exact_match(self) -> None:
        frame = self.frame.rename(columns={"truth": "target"}).copy()
        frame["pretrained"] = [0.01, 0.18, 0.01, 0.12]
        frame["direct"] = frame.target
        frame["aligned"] = frame.pretrained
        pairs = pair_corrections(frame, self.pairs)
        exact = correction_summary(pairs.r_target.to_numpy(), pairs.r_direct.to_numpy())
        self.assertAlmostEqual(exact["excess_mae"], 0.0)
        self.assertAlmostEqual(exact["projection_alpha"], 1.0)
        self.assertAlmostEqual(exact["orthogonal_norm_ratio"], 0.0)

    def test_duplicate_ids_fail_closed(self) -> None:
        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_pairs(duplicate)

    def test_synthetic_reference_shift_total_and_group_invariants(self) -> None:
        fractions = np.zeros((3, 118), dtype=float)
        fractions[0, :2] = [0.5, 0.5]
        fractions[1, :2] = [0.5, 0.5]
        fractions[2, :2] = [0.25, 0.75]
        natoms = np.array([2, 4, 4])
        mu = np.arange(118, dtype=float) / 100.0
        delta_mu = fold_intercept_into_element_coefficients(mu, 0.2)
        pretrained = np.array([-1.0, -0.9, -0.7])
        zero, reference = build_synthetic_targets(pretrained, fractions, delta_mu)
        np.testing.assert_array_equal(zero, pretrained)
        audit = audit_synthetic_targets(
            pretrained_per_atom=pretrained,
            target_reference_per_atom=reference,
            fractions=fractions,
            natoms=natoms,
            delta_mu=delta_mu,
            composition_keys=np.array(["same", "same", "other"]),
        )
        self.assertLessEqual(audit["total_shift_identity_max_abs_ev"], 1e-12)
        self.assertLessEqual(audit["same_composition_shift_max_spread_ev_atom"], 1e-12)


if __name__ == "__main__":
    unittest.main()
