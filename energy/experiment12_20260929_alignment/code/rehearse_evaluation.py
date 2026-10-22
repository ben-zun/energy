"""Rehearsal: exercise the confirmation-evaluation path on early checkpoints.

Runs while the long training is still going, so it validates the riskiest code
(model loading, confirmation inference, per-epoch calibration reuse, paired
bootstrap) before the final analysis depends on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alignment_common import build_model, centered, ec_mae, run_dir  # noqa: E402
from evaluate_alignment import cluster_interval, confirmation_setup, evaluate_checkpoint  # noqa: E402


def main():
    device = torch.device('cuda:0')
    graphs, manifest, fractions, offset = confirmation_setup()
    truth = manifest.y_true.to_numpy()
    groups = manifest.exact_group.to_numpy()
    overall = manifest.overall_cohort.to_numpy(bool)
    relative = manifest.relative_cohort.to_numpy(bool)
    print('confirmation: %d structures | overall %d | relative %d | relative groups %d'
          % (len(manifest), overall.sum(), relative.sum(), len(np.unique(groups[relative]))))

    model = build_model(device)
    tensors = (graphs, manifest, fractions, offset)
    results = {}
    for route in ('U', 'A'):
        native, calibrated = evaluate_checkpoint(route, 20260528, 5, model, tensors, device)
        results[route] = calibrated
        ec_cal = ec_mae(truth[relative], calibrated[relative], groups[relative])[0]
        ec_nat = ec_mae(truth[relative], native[relative], groups[relative])[0]
        print('%s seed 20260528 epoch 5: overall_abs(cal)=%.2f relative_abs(cal)=%.2f '
              'relative_EC=%.3f |EC_cal-EC_nat|=%.2e'
              % (route, 1000 * np.abs(calibrated - truth)[overall].mean(),
                 1000 * np.abs(calibrated - truth)[relative].mean(), ec_cal, abs(ec_cal - ec_nat)))

    error_u = 1000 * np.abs(centered(results['U'] - truth, groups))[relative]
    error_a = 1000 * np.abs(centered(results['A'] - truth, groups))[relative]
    point, low, high = cluster_interval(error_u - error_a, groups[relative])
    print('paired U-A on relative EC error: %+.3f meV [%+.3f, %+.3f] (bootstrap path OK)'
          % (point, low, high))
    print('U_20260528 checkpoints so far:', len(list(run_dir('U', 20260528).glob('epoch*.pt'))))


if __name__ == '__main__':
    main()
