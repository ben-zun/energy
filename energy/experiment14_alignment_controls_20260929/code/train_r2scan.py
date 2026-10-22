"""Run one (route, seed) of the cross-protocol r2SCAN experiment.

Reuses the experiment14 matched trainer unchanged; only three things are swapped:
  * the graph cache -> the r2SCAN split built by r2scan_prepare.py
  * the routed reference d(z) -> re-fitted on the r2SCAN train split
  * the output root -> r2scan10k/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
EXP12_CODE = ENERGY / 'experiment12_20260929_alignment/code'
R2 = EXP14 / 'r2scan10k'
sys.path.insert(0, str(EXP14 / 'code'))
sys.path.insert(0, str(EXP12_CODE))

import alignment_common as ac  # noqa: E402
import train_controls as tc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('U', 'A'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--verify-init-only', action='store_true')
    args = parser.parse_args()

    ac.GRAPH_CACHE = R2 / 'graph_cache'
    tc.OUT = R2

    fit = np.load(R2 / 'r2scan_reference_fit.npz')
    mu, bias = fit['w'], float(fit['b'])

    def load_element_reference(path=None):
        return mu, bias

    tc.load_element_reference = load_element_reference

    split = json.loads((R2 / 'split_report.json').read_text())
    sys.argv = [
        'train_controls.py', '--route', args.route, '--seed', str(args.seed),
        '--lr', str(args.lr), '--tag', 'r2scan', '--device', args.device,
        '--epochs', str(args.epochs),
        '--expect-train', str(split['train']['structures']),
        '--expect-val', str(split['val']['structures']),
        '--out-root', str(R2),
    ]
    if args.verify_init_only:
        sys.argv.append('--verify-init-only')
    print(f'r2SCAN route={args.route} seed={args.seed} lr={args.lr} '
          f'epochs={args.epochs} device={args.device}', flush=True)
    tc.main()


if __name__ == '__main__':
    main()
