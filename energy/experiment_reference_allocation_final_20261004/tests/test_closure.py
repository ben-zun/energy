import unittest

import numpy as np
from reference_allocation.closure import (
    projection_metrics,
    weighted_mae_from_clusters,
    weighted_rmse_from_clusters,
)


class ClosureTests(unittest.TestCase):
    def test_projection_separates_parallel_and_orthogonal_parts(self):
        target = np.array([1.0, 0.0, -1.0])
        update = 2.0 * target + np.array([1.0, -2.0, 1.0])
        result = projection_metrics(target, update)
        self.assertAlmostEqual(result["beta"], 2.0)
        self.assertAlmostEqual(result["orthogonal_norm_ratio"], np.sqrt(3.0))
        self.assertAlmostEqual(result["projection_explained_fraction"], 4.0 / 7.0)

    def test_cluster_statistics_retain_cluster_multiplicity(self):
        absolute_sums = np.array([2.0, 9.0])
        squared_sums = np.array([2.0, 27.0])
        counts = np.array([2.0, 3.0])
        sampled = np.array([0, 0, 1])
        self.assertAlmostEqual(
            weighted_mae_from_clusters(absolute_sums, counts, sampled), 13.0 / 7.0
        )
        self.assertAlmostEqual(
            weighted_rmse_from_clusters(squared_sums, counts, sampled), np.sqrt(31.0 / 7.0)
        )

    def test_projection_rejects_zero_target(self):
        with self.assertRaises(ValueError):
            projection_metrics(np.zeros(3), np.ones(3))


if __name__ == "__main__":
    unittest.main()
