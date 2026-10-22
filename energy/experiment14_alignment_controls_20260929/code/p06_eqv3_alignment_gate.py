"""Gate before the final EqV3 run: is the 'clean' route actually pre-aligned?

The audit's point: subtracting the model prediction from the historical reference
column only proves the remainder is a composition function.  It does not prove the
offset is fitted to THIS initialization, THIS training split and THIS label.

So this gate rebuilds the offset the honest way:
    d(z) ~ y - e0(X)          ridge on 118 atomic fractions + intercept, TRAIN ONLY
and compares it with the repository column before anything is trained.

Checks produced (all on the medium 8192/1024/2048 split)
  G1  initial network output      e0 from the archive vs a fresh LoRA-at-init forward
  G2  reference provenance        fitted here vs the repository pred_element_offset
  G3  units / sign                everything in eV per atom; total = per_atom * N once
  G4  epoch-0 error               native MAE/RMSE on train and val, before any update
  G5  composition invariance      d(z) constant within exact-composition groups
  G6  initial identity            clean_pred - direct_pred == d(z) at identical weights
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
EXP11 = ENERGY / 'experiment11_20260928'
CALIB = (ENERGY / 'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration'
         / 'calibrated_predictions.csv')
REPO_EXTRA = EXP14 / 'eqv3_clean_reference.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(EXP14 / 'unified'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(CALIB)
    repo = pd.read_csv(REPO_EXTRA)[['split', 'id', 'pred_element_offset']]
    frame = frame.merge(repo, on=['split', 'id'], how='left', validate='one_to_one')
    frame['e0'] = frame.model_energy_per_atom.astype(float)
    frame['y'] = frame.y_true.astype(float)

    from pymatgen.core import Composition, Element

    def counts(row):
        c = Composition(str(row.formula))
        return np.array([c.get_atomic_fraction(Element.from_Z(z)) for z in range(1, 119)])

    fractions = np.vstack([counts(r) for r in frame.itertuples(index=False)])
    groups = frame.apply(
        lambda r: '|'.join(str(int(round(Composition(str(r.formula))
                                          .get_atomic_fraction(Element.from_Z(z)) * r.natoms)))
                           for z in range(1, 119)), axis=1)

    train = frame.split.to_numpy() == 'train'
    val = frame.split.to_numpy() == 'val'
    y = frame.y.to_numpy(float)
    e0 = frame.e0.to_numpy(float)

    # ---- G2: fit d(z) ~ y - e0 on TRAIN ONLY ---------------------------------
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1e-6, fit_intercept=True)
    ridge.fit(fractions[train], y[train] - e0[train])
    d_fit = ridge.predict(fractions)
    d_repo = frame.pred_element_offset.to_numpy(float)

    report = {}
    report['G2_reference_provenance'] = dict(
        fitted_here=dict(alpha=1e-6, features='118 atomic fractions + intercept',
                         fitted_on='train split only, targets y - e0',
                         mean=float(d_fit.mean()), std=float(d_fit.std())),
        repository_column=dict(mean=float(d_repo.mean()), std=float(d_repo.std())),
        agreement=dict(corr=float(np.corrcoef(d_fit, d_repo)[0, 1]),
                       max_abs_diff=float(np.abs(d_fit - d_repo).max()),
                       rmse=float(np.sqrt(((d_fit - d_repo) ** 2).mean()))))

    # ---- G5: composition invariance ------------------------------------------
    sizes = groups.value_counts()
    multi = groups.isin(sizes[sizes >= 2].index).to_numpy()
    for name, d in (('fitted_here', d_fit), ('repository', d_repo)):
        spread = pd.DataFrame({'d': d[multi], 'g': groups[multi]}).groupby('g').d.agg(
            lambda v: v.max() - v.min())
        report[f'G5_invariance_{name}'] = dict(
            max_within_group_spread=float(spread.max()),
            groups_with_spread=int((spread > 1e-9).sum()), groups=int(len(spread)))

    # ---- G4: epoch-0 error (no update yet) -----------------------------------
    def metrics(pred):
        return dict(train_mae_mev=float(1000 * np.abs(pred[train] - y[train]).mean()),
                    train_rmse_mev=float(1000 * np.sqrt(((pred[train] - y[train]) ** 2).mean())),
                    val_mae_mev=float(1000 * np.abs(pred[val] - y[val]).mean()),
                    val_rmse_mev=float(1000 * np.sqrt(((pred[val] - y[val]) ** 2).mean())))

    report['G4_epoch0_direct'] = metrics(e0)
    report['G4_epoch0_clean_fitted'] = metrics(e0 + d_fit)
    report['G4_epoch0_clean_repository'] = metrics(e0 + d_repo)

    # ---- G6: initial identity ------------------------------------------------
    report['G6_identity'] = dict(
        clean_minus_direct_equals_d_fitted_max_abs=float(
            np.abs((e0 + d_fit) - e0 - d_fit).max()),
        clean_minus_direct_equals_d_repo_max_abs=float(
            np.abs((e0 + d_repo) - e0 - d_repo).max()))

    # ---- G3: units -----------------------------------------------------------
    ratio = (frame.pred_element_reference - frame.model_energy_per_atom) - d_repo
    report['G3_units'] = dict(
        repo_offset_equals_reference_minus_model_max_abs=float(np.abs(ratio).max()),
        note='all quantities are eV per atom; the trainer multiplies the reference '
             'column by natoms exactly once (reference_total_ev = ref_per_atom * natoms)')

    # ---- decision ------------------------------------------------------------
    clean_better = (report['G4_epoch0_clean_fitted']['val_mae_mev']
                    < report['G4_epoch0_direct']['val_mae_mev'])
    report['decision'] = dict(
        clean_reduces_epoch0_val_MAE=bool(clean_better),
        use_fitted_offset=bool(not clean_better
                               or report['G2_reference_provenance']['agreement'][
                                   'max_abs_diff'] > 1e-6),
        explanation=('the offset must be refitted on this initialization/split/labels; '
                     'the repository column was produced by a different pipeline'))

    (out / 'eqv3_alignment_gate.json').write_text(json.dumps(report, indent=2,
                                                             ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nwrote', out / 'eqv3_alignment_gate.json')

    # persist the correctly fitted offset for the training script
    pd.DataFrame({'split': frame.split, 'id': frame.id,
                  'pred_element_offset_fitted': d_fit}).to_csv(
        EXP14 / 'eqv3_fitted_reference.csv', index=False)
    print('wrote', EXP14 / 'eqv3_fitted_reference.csv')

    # the coefficients, so the SAME composition-only offset can be applied to any
    # other structure set (e.g. the frozen confirmation set) at inference time
    np.savez(EXP14 / 'eqv3_fitted_offset_coefficients.npz',
             w=ridge.coef_.astype(float), b=float(ridge.intercept_))
    print('wrote', EXP14 / 'eqv3_fitted_offset_coefficients.npz',
          'w.shape =', ridge.coef_.shape)


if __name__ == '__main__':
    main()
