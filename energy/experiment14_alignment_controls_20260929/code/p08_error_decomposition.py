"""Existing-prediction analysis for the reframed paper.

Three things, all from predictions already on disk -- no training.

A. Error decomposition (exact for MSE, never for MAE)
       r_gi = pred_gi - y_gi                       per-structure error
       r_gi = rbar_g + (r_gi - rbar_g)
       MSE  = (1/M) sum_g n_g rbar_g^2  +  (1/M) sum_g sum_i (r_gi - rbar_g)^2
            = between-group bias        +  within-group relative-energy error
   The identity is verified numerically for every method.

B. What the target needs vs what the model does
       t   = centreg(y)     - centreg(base)
       c_m = centreg(pred_m) - centreg(base)
   with centreg the exact-composition group centring.  Reported: ||c||/||t||,
   cos(c, t), and the decomposition
       ||c - t||^2 = ||t||^2 + ||c||^2 - 2 c.t
   so that "changed too much / too little / wrong direction" is read off rather
   than assumed.  Nothing here presumes which route over- or under-corrects.

Note on scaling: every composition-only term cancels under group centring, so
these numbers are invariant to the train-only composition ridge the pipeline
applies.  No per-method scale factor is used anywhere in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
UNIFIED = EXP14 / 'unified'
OUT = UNIFIED


def centred(values, groups):
    """Remove each exact-composition group's mean."""
    frame = pd.DataFrame({'v': values, 'g': groups})
    return (frame.v - frame.groupby('g').v.transform('mean')).to_numpy(float)


def decomposition(pred, truth, groups, mask):
    rows = np.flatnonzero(mask)
    r = (pred - truth)[rows]
    g = groups[rows]
    frame = pd.DataFrame({'r': r, 'g': g})
    agg = frame.groupby('g').r.agg(['mean', 'size'])
    sse_total = float((r ** 2).sum())
    sse_between = float((agg['mean'] ** 2 * agg['size']).sum())
    within = frame.r - frame.groupby('g').r.transform('mean')
    sse_within = float((within ** 2).sum())
    m = len(rows)
    return dict(
        structures=int(m), groups=int(len(agg)),
        sse_total=sse_total, sse_between=sse_between, sse_within=sse_within,
        identity_residual=float(abs(sse_total - sse_between - sse_within)),
        mse_total=sse_total / m, mse_between=sse_between / m, mse_within=sse_within / m,
        rmse_total_mev=1000 * float(np.sqrt(sse_total / m)),
        rmse_between_mev=1000 * float(np.sqrt(sse_between / m)),
        rmse_within_mev=1000 * float(np.sqrt(sse_within / m)),
        share_between=sse_between / sse_total if sse_total else np.nan,
        share_within=sse_within / sse_total if sse_total else np.nan,
        mae_mev=1000 * float(np.abs(r).mean()),
        ec_mae_mev=1000 * float(np.abs(within).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scopes', default='dedup,raw')
    args = parser.parse_args()

    manifest = pd.read_csv(UNIFIED / 'evaluation_manifest.csv', keep_default_na=False)
    truth = manifest.y_true.to_numpy(float)
    groups = manifest.exact_group.to_numpy()
    is_rel = manifest.relative_cohort.to_numpy(bool)
    keep = manifest.keep.to_numpy(bool)
    conflict = manifest.label_conflict.fillna(False).to_numpy(bool)
    overall = manifest.overall_cohort.to_numpy(bool)
    scopes = {'raw': is_rel, 'dedup': is_rel & keep,
              'dedup_no_conflict': is_rel & keep & (~conflict),
              'overall': overall, 'overall_dedup': overall & keep}

    # ---- collect every method's confirmation predictions --------------------
    predictions = {}
    mb = pd.read_csv(UNIFIED / 'predictions_matbench.csv')
    for (method, seed), block in mb.groupby(['method', 'seed']):
        key = method if method in ('base', 'composition') else f'{method}'
        predictions.setdefault(key, {})[seed] = (
            block.drop_duplicates('id').set_index('id').loc[manifest.id]
            .prediction.to_numpy(float))

    eq = UNIFIED / 'predictions_eqv3_clean.csv'
    if eq.exists():
        frame = pd.read_csv(eq)
        for (method, seed), block in frame.groupby(['method', 'seed']):
            predictions.setdefault(f'eqv3_{method}', {})[seed] = (
                block.drop_duplicates('id').set_index('id').loc[manifest.id]
                .prediction.to_numpy(float))

    f40 = UNIFIED / 'predictions_eqv3_final40.csv'
    if f40.exists():
        frame = pd.read_csv(f40)
        for (method, seed, sel), block in frame.groupby(
                ['method', 'seed', 'checkpoint_selection']):
            predictions.setdefault(f'eqv3_40_{method}_{sel}', {})[seed] = (
                block.drop_duplicates('id').set_index('id').loc[manifest.id]
                .prediction.to_numpy(float))

    # ---- A. decomposition ---------------------------------------------------
    dec_rows = []
    for scope in args.scopes.split(','):
        mask = scopes[scope]
        for method, by_seed in sorted(predictions.items()):
            for seed, pred in sorted(by_seed.items()):
                row = decomposition(pred, truth, groups, mask)
                row.update(method=method, seed=seed, scope=scope)
                dec_rows.append(row)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / 'error_decomposition.csv', index=False)
    bad = dec[dec.identity_residual > 1e-6]
    if len(bad):
        raise AssertionError(f'decomposition identity failed for {len(bad)} rows')

    # ---- B. target vs actual change ----------------------------------------
    change_rows = []
    for scope in args.scopes.split(','):
        mask = scopes[scope]
        for base_method in ('base', 'eqv3_residual'):
            if base_method not in predictions:
                continue
            seeds = sorted(set(predictions[base_method]) & set(
                s for m in predictions.values() for s in m))
            for seed in seeds:
                base_pred = predictions[base_method][seed]
                y_c = centred(truth, groups)
                b_c = centred(base_pred, groups)
                t = (y_c - b_c)[mask]
                norm_t = float(np.linalg.norm(t))
                if norm_t == 0:
                    continue
                for method, by_seed in sorted(predictions.items()):
                    if seed not in by_seed:
                        continue
                    c = (centred(by_seed[seed], groups) - b_c)[mask]
                    norm_c = float(np.linalg.norm(c))
                    dot = float(c @ t)
                    diff = float(np.linalg.norm(c - t) ** 2)
                    change_rows.append(dict(
                        scope=scope, reference=base_method, method=method, seed=seed,
                        n_structures=int(mask.sum()),
                        rms_t_mev=1000 * norm_t / np.sqrt(mask.sum()),
                        rms_c_mev=1000 * norm_c / np.sqrt(mask.sum()),
                        norm_ratio=norm_c / norm_t,
                        cos_ct=dot / (norm_c * norm_t) if norm_c else np.nan,
                        identity_residual=abs(diff - (norm_t ** 2 + norm_c ** 2 - 2 * dot)),
                        scale_ref=norm_t ** 2,
                        rms_ct_mev=1000 * np.sqrt(diff / mask.sum()),
                        components=json.dumps(dict(
                            norm_t2=norm_t ** 2, norm_c2=norm_c ** 2, two_dot=2 * dot))))
    change = pd.DataFrame(change_rows)
    change.to_csv(OUT / 'target_vs_actual_change.csv', index=False)
    # both sides are in eV^2; compare against the natural scale of the identity
    bad = change[change.identity_residual > 1e-8 * change.scale_ref.clip(lower=1e-12)]
    if len(bad):
        raise AssertionError(
            f'norm identity failed for {len(bad)} rows; worst '
            f'{bad.identity_residual.max():.3e} vs scale {bad.scale_ref.min():.3e}')

    for scope in args.scopes.split(','):
        print(f'\n================ {scope} ================')
        sub = dec[dec.scope.eq(scope)].groupby('method')[
            ['rmse_total_mev', 'rmse_between_mev', 'rmse_within_mev', 'share_between',
             'share_within', 'mae_mev', 'ec_mae_mev']].mean()
        print('--- A. MSE decomposition (3-seed mean) ---')
        print(sub.round(3).to_string())
        for reference in ('base', 'eqv3_residual'):
            ch = change[(change.scope.eq(scope)) & (change.reference.eq(reference))]
            if len(ch) == 0:
                continue
            print(f'--- B. target vs actual change, reference = {reference} ---')
            print(ch.groupby('method')[
                ['rms_t_mev', 'rms_c_mev', 'norm_ratio', 'cos_ct', 'rms_ct_mev']
            ].mean().round(3).to_string())
    print('\nwrote', OUT / 'error_decomposition.csv', 'and target_vs_actual_change.csv')


if __name__ == '__main__':
    main()
