"""Analysis C: did adding d(z) improve the starting point, and what did training do?

Answers the audit question directly, from archived artefacts only:
   C1  epoch-0 (no parameter update) native error for every route
   C2  the trajectory, to see whether training preserves or destroys the start
   C3  a side-by-side of "start" and "end" so the inversion is explicit

No training.  C1 is re-derived here rather than quoted, using the same objects the
gate used: the frozen per-atom output e0 from the calibration archive, and the
composition-only offset fitted on the current initialisation / train split / labels.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
UNIFIED = EXP14 / 'unified'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(UNIFIED))
    args = parser.parse_args()
    out = Path(args.out)

    coeffs = np.load(EXP14 / 'eqv3_fitted_offset_coefficients.npz')
    w, b = coeffs['w'], float(coeffs['b'])

    calib = (ENERGY / 'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration'
             / 'calibrated_predictions.csv')
    frame = pd.read_csv(calib)
    from pymatgen.core import Composition, Element

    def fractions(formula):
        c = Composition(str(formula))
        return np.array([c.get_atomic_fraction(Element.from_Z(z)) for z in range(1, 119)])

    x = np.vstack([fractions(f) for f in frame.formula])
    e0 = frame.model_energy_per_atom.to_numpy(float)
    y = frame.y_true.to_numpy(float)
    split = frame.split.to_numpy()
    delta = x @ w + b

    rows = []
    for name, pred in (('direct', e0), ('clean', e0 + delta)):
        for s in ('train', 'val', 'test'):
            m = split == s
            err = pred[m] - y[m]
            rows.append(dict(route=name, split=s, structures=int(m.sum()),
                             mae_mev=1000 * float(np.abs(err).mean()),
                             rmse_mev=1000 * float(np.sqrt((err ** 2).mean())),
                             bias_mev=1000 * float(err.mean())))
    epoch0 = pd.DataFrame(rows)
    epoch0.to_csv(out / 'epoch0_native_error.csv', index=False)

    # ---- C2 trajectory ------------------------------------------------------
    traj = []
    for path in sorted(glob.glob(str(EXP14 / 'eqv3_final40/*/seed*/final40*history.csv'))):
        d = pd.read_csv(path)
        route = path.split('/')[-3]
        seed = int(path.split('/')[-2].replace('seed', ''))
        for _, r in d.iterrows():
            traj.append(dict(route=route, seed=seed, epoch=int(r.epoch),
                             train_mae_mev=float(r.train_mae_mev_atom),
                             val_mae_mev=float(r.val_mae_mev_atom)))
    traj = pd.DataFrame(traj)
    traj.to_csv(out / 'training_trajectory.csv', index=False)

    # ---- C3 start vs end ----------------------------------------------------
    start = epoch0[epoch0.split.eq('val')].set_index('route').mae_mev
    early = traj[traj.epoch.eq(1)].groupby('route').val_mae_mev.mean()
    best = traj.groupby('route').val_mae_mev.min()
    last = traj[traj.epoch.eq(traj.epoch.max())].groupby('route').val_mae_mev.mean()
    compare = pd.DataFrame({
        'epoch0_mev': start,
        'epoch1_mean_mev': early,
        'best_mev': best,
        'final_epoch_mean_mev': last,
    })
    best_idx = traj.groupby('route').val_mae_mev.idxmin()
    compare['best_epoch'] = traj.loc[best_idx].set_index('route').epoch
    compare['best_seed'] = traj.loc[best_idx].set_index('route').seed
    compare['ratio_epoch0_over_best'] = compare.epoch0_mev / compare.best_mev
    compare.to_csv(out / 'alignment_start_vs_end.csv')

    print('=== C1. epoch-0 native error (no parameter update) ===')
    print(epoch0.round(2).to_string(index=False))
    print('\n=== C3. start vs end, validation split (meV/atom) ===')
    print(compare.round(2).to_string())
    print('\nwrote epoch0_native_error.csv, training_trajectory.csv, '
          'alignment_start_vs_end.csv')


if __name__ == '__main__':
    main()
