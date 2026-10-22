from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = next(
    (candidate for candidate in (ROOT / "upstream_snapshots" / "eqv3", ROOT / "inputs" / "eqv3")
     if candidate.is_dir()),
    ROOT / "upstream_snapshots" / "eqv3",
)


@unittest.skipUnless(INPUTS.is_dir(), "real EqV3 snapshot not present")
class EqV3RegressionTests(unittest.TestCase):
    def test_real_snapshot_reproduces_locked_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "finalize_eqv3.py"),
                    "--input-root", str(INPUTS),
                    "--output-root", str(output),
                    "--bootstrap-draws", "200",
                ],
                check=True,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            metrics = pd.read_csv(output / "metrics" / "per_seed_metrics.csv")
            archived = pd.read_csv(INPUTS / "original_per_seed_metrics.csv")
            archived["route"] = archived.route.map({"clean": "aligned", "direct": "direct"})
            archived["calibration"] = archived.calibration.replace({"native": "raw"})
            archived["cohort"] = archived.scope.map(
                {"overall2048": "overall2048", "relative_clean401": "relative401"}
            )
            joined = metrics.merge(
                archived[["seed", "route", "calibration", "cohort", "mae_mev", "ec_mae_mev"]],
                on=["seed", "route", "calibration", "cohort"],
                suffixes=("_new", "_archived"),
                validate="one_to_one",
            )
            self.assertEqual(len(joined), 24)
            self.assertTrue(
                ((joined.mae_mev_new - joined.mae_mev_archived).abs() < 1e-9).all()
            )
            relative = joined[joined.cohort == "relative401"]
            self.assertTrue(
                ((relative.ec_mae_mev_new - relative.ec_mae_mev_archived).abs() < 1e-9).all()
            )
            self.assertTrue((metrics.loc[metrics.cohort == "relative401", "n_groups"] == 190).all())


if __name__ == "__main__":
    unittest.main()
