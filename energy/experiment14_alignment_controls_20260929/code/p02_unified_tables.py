"""P0-2 / P0-3 / P0-5 and batch-2 (1): one script, one definition, all tables.

Everything is recomputed from the P0-1 evaluation manifest and the saved
per-structure predictions.  No model is retrained and no metric is taken from an
earlier script.

Scopes (from P0-1)
  raw                200 groups / 422 structures
  dedup              190 groups / 401 structures   <- primary
  dedup_no_conflict  dedup minus label-conflict groups
plus spread filters (non-degenerate subsets) for the selection-consequence table.

Every paired number carries its statistic definition:
  diff_of_group_means        mean_g(x_g) - mean_g(y_g)          (paired resample)
  diff_of_group_medians      median_g(x_g) - median_g(y_g)      (paired resample)
  diff_of_group_quantiles    Q0.9(x_g) - Q0.9(y_g)              descriptive contrast
  quantile_of_group_diffs    Q0.9(x_g - y_g)                    paired, per-group
  rate_difference            mean_g(hit_x) - mean_g(hit_y)      paired resample
For binary hit/acceptance metrics the bootstrap is near-degenerate, so the exact
discordant counts (n10 / n01, McNemar style) are reported alongside.
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
R2RUN = EXP14 / 'r2scan10k'
CONF = ENERGY / 'experiment11_20260928/confirmation'
EXP12 = ENERGY / 'experiment12_20260929_alignment'

TIE_TAUS = (0.0, 1.0, 2.0, 5.0, 10.0)
MATERIAL_TAU = 10.0
DRAWS, BOOT_SEED = 5000, 20260930
SPREAD_CUTS = (5.0, 10.0, 20.0)
LR_GRID = (1e-4, 3e-4, 1e-3)

MATBENCH_METHODS = ['base', 'composition', 'adapt'] + \
    [f'{r}@{lr:g}' for r in ('U', 'A') for lr in LR_GRID] + ['T@0.0003']
EQV3_METHODS = ['residual', 'direct']
R2SCAN_METHODS = ['frozen_raw', 'frozen_aligned', 'U_20260528', 'U_20260529', 'U_20260530',
                  'A_20260528', 'A_20260529', 'A_20260530']


# --------------------------------------------------------------------------- #
# per-group metrics
# --------------------------------------------------------------------------- #
def per_group(truth, prediction, groups, mask, seed, method, scope):
    rows = np.flatnonzero(mask)
    frame = pd.DataFrame({'y': truth[rows], 'p': prediction[rows], 'g': groups[rows]})
    out = []
    for name, block in frame.groupby('g', sort=True):
        if len(block) < 2:
            continue
        y, yhat = block.y.to_numpy(), block.p.to_numpy()
        centred_err = (yhat - yhat.mean()) - (y - y.mean())
        chosen = int(np.argmin(yhat))
        cost = float(y[chosen] - y.min()) * 1000.0
        pair_err = []
        for a in range(len(y)):
            for b in range(a + 1, len(y)):
                pair_err.append(abs((yhat[a] - yhat[b]) - (y[a] - y[b])) * 1000.0)
        record = dict(scope=scope, seed=seed, method=method, group=name, n=len(y),
                      y_spread_mev=1000 * float(y.max() - y.min()),
                      ec_abs_sum_mev=float(np.abs(centred_err).sum()) * 1000.0,
                      pair_err_sum_mev=float(np.sum(pair_err)), pair_count=len(pair_err),
                      exact_hit=int(cost <= 1e-9), cost_mev=cost)
        for tau in TIE_TAUS:
            record[f'acceptable_{tau:g}'] = int(cost <= tau)
        out.append(record)
    return pd.DataFrame(out)


def seed_metrics(group_frame, abs_mae_mev, **meta):
    """The scope IS the relative cohort, so absolute MAE is computed on its structures."""
    n = group_frame.n.sum()
    row = dict(**meta,
               groups=len(group_frame), structures=int(n),
               pairs=int(group_frame.pair_count.sum()),
               abs_mae_mev=float(abs_mae_mev),
               ec_mae_mev=float(group_frame.ec_abs_sum_mev.sum() / n),
               pair_mae_mev=float(group_frame.pair_err_sum_mev.sum() / group_frame.pair_count.sum()),
               exact_hit_rate=float(group_frame.exact_hit.mean()),
               cost_mean_mev=float(group_frame.cost_mev.mean()),
               cost_median_mev=float(group_frame.cost_mev.median()),
               cost_p90_mev=float(np.quantile(group_frame.cost_mev, .90)),
               wrong_pick_rate_gt_10mev=float((group_frame.cost_mev > MATERIAL_TAU).mean()))
    for tau in TIE_TAUS:
        row[f'acceptable_{tau:g}_rate'] = float(group_frame[f'acceptable_{tau:g}'].mean())
    return row


def bootstrap_groups(group_frame, draws=DRAWS, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(group_frame), size=(draws, len(group_frame)))
    n = group_frame.n.to_numpy()[pick]
    return dict(
        ec_mae=group_frame.ec_abs_sum_mev.to_numpy()[pick].sum(axis=1) / n.sum(axis=1),
        pair_mae=(group_frame.pair_err_sum_mev.to_numpy()[pick].sum(axis=1)
                  / group_frame.pair_count.to_numpy()[pick].sum(axis=1)),
        exact_hit=group_frame.exact_hit.to_numpy()[pick].mean(axis=1),
        cost_mean=group_frame.cost_mev.to_numpy()[pick].mean(axis=1),
        cost_median=np.median(group_frame.cost_mev.to_numpy()[pick], axis=1),
        cost_p90=np.quantile(group_frame.cost_mev.to_numpy()[pick], .90, axis=1),
        wrong10=(group_frame.cost_mev.to_numpy()[pick] > MATERIAL_TAU).mean(axis=1),
        **{f'acceptable_{t:g}': group_frame[f'acceptable_{t:g}'].to_numpy()[pick].mean(axis=1)
           for t in TIE_TAUS})


def ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


# --------------------------------------------------------------------------- #
# paired contrasts
# --------------------------------------------------------------------------- #
def paired_contrast(frame_a, frame_b, label_a, label_b, contrast):
    """frame_* are per-group tables with the same group order."""
    if not (frame_a.group.to_numpy() == frame_b.group.to_numpy()).all():
        raise ValueError('group order differs between the two methods')
    rng = np.random.default_rng(BOOT_SEED)
    pick = rng.integers(len(frame_a), size=(DRAWS, len(frame_a)))
    ca, cb = frame_a.cost_mev.to_numpy(), frame_b.cost_mev.to_numpy()
    na, nb = frame_a.n.to_numpy(), frame_b.n.to_numpy()
    eca, ecb = frame_a.ec_abs_sum_mev.to_numpy(), frame_b.ec_abs_sum_mev.to_numpy()
    pa, pb = frame_a.pair_err_sum_mev.to_numpy(), frame_b.pair_err_sum_mev.to_numpy()
    pca, pcb = frame_a.pair_count.to_numpy(), frame_b.pair_count.to_numpy()
    ha, hb = frame_a.exact_hit.to_numpy(), frame_b.exact_hit.to_numpy()
    wa, wb = (ca > MATERIAL_TAU).astype(float), (cb > MATERIAL_TAU).astype(float)

    def mean_diff(x, y):
        return float(np.mean(x) - np.mean(y)), (x[pick].mean(axis=1) - y[pick].mean(axis=1))

    rows = []
    specs = [
        ('ec_mae_mev', 'diff_of_group_means',
         lambda: mean_diff(eca / na, ecb / nb)),
        ('pair_mae_mev', 'diff_of_group_means',
         lambda: mean_diff(pa / pca, pb / pcb)),
        ('exact_hit_rate', 'rate_difference', lambda: mean_diff(ha, hb)),
        ('wrong_pick_rate_gt_10mev', 'rate_difference', lambda: mean_diff(wa, wb)),
        ('acceptable_rate_tau1', 'rate_difference',
         lambda: mean_diff(frame_a['acceptable_1'].to_numpy(),
                           frame_b['acceptable_1'].to_numpy())),
        ('cost_mean_mev', 'diff_of_group_means', lambda: mean_diff(ca, cb)),
        ('cost_median_mev', 'diff_of_group_medians',
         lambda: (float(np.median(ca) - np.median(cb)),
                  np.median(ca[pick], axis=1) - np.median(cb[pick], axis=1))),
        ('cost_p90_mev', 'diff_of_group_quantiles',
         lambda: (float(np.quantile(ca, .90) - np.quantile(cb, .90)),
                  np.quantile(ca[pick], .90, axis=1) - np.quantile(cb[pick], .90, axis=1))),
        ('cost_p90_mev', 'quantile_of_group_diffs',
         lambda: (float(np.quantile(ca - cb, .90)),
                  np.quantile(ca[pick] - cb[pick], .90, axis=1))),
    ]
    for metric, statistic, fn in specs:
        point, dist = fn()
        lo, hi = ci(dist)
        rows.append(dict(contrast=contrast, metric=metric, statistic_definition=statistic,
                         value_a=label_a, value_b=label_b, difference=point,
                         ci95_low=lo, ci95_high=hi, groups=len(frame_a),
                         excludes_zero=bool(lo > 0 or hi < 0)))
    # exact discordant counts for the binary metrics
    return rows, (int(((ha == 0) & (hb == 1)).sum()), int(((ha == 1) & (hb == 0)).sum()),
                  int(((wa == 1) & (wb == 0)).sum()), int(((wa == 0) & (wb == 1)).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(UNIFIED))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(out / 'evaluation_manifest.csv', keep_default_na=False)
    truth = manifest.y_true.to_numpy(float)
    groups = manifest.exact_group.to_numpy()
    is_rel = manifest.relative_cohort.to_numpy(bool)
    keep = manifest.keep.to_numpy(bool)
    conflict = manifest.label_conflict.fillna(False).to_numpy(bool)

    scopes = {}
    for scope in ('raw', 'dedup', 'dedup_no_conflict'):
        mask = is_rel.copy()
        if scope != 'raw':
            mask &= keep
        if scope == 'dedup_no_conflict':
            mask &= ~conflict
        scopes[scope] = mask
    spread_cache = {}

    predictions = pd.read_csv(out / 'predictions_matbench.csv')
    per_group_frames, seed_rows = {}, []
    for (method, seed), block in predictions.groupby(['method', 'seed'], sort=False):
        block = block.set_index('id').loc[manifest.id]
        prediction = block.prediction.to_numpy(float)
        for scope, mask in scopes.items():
            abs_mae = 1000 * float(np.abs(prediction[mask] - truth[mask]).mean())
            frame = per_group(truth, prediction, groups, mask, seed, method, scope)
            per_group_frames[(method, seed, scope)] = frame
            seed_rows.append(seed_metrics(frame, abs_mae, method=method, seed=seed, scope=scope))

    # ---- non-degenerate subsets -------------------------------------------
    for cut in SPREAD_CUTS:
        for (method, seed), block in predictions.groupby(['method', 'seed'], sort=False):
            base = per_group_frames[(method, seed, 'dedup')]
            sub = base[base.y_spread_mev > cut].reset_index(drop=True)
            if len(sub) < 5:
                continue
            per_group_frames[(method, seed, f'spread_gt_{cut:g}')] = sub
            seed_rows.append(seed_metrics(
                sub, np.nan, method=method, seed=seed, scope=f'spread_gt_{cut:g}'))

    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(out / 'per_seed_metrics.csv', index=False)

    per_group_all = pd.concat(per_group_frames.values(), ignore_index=True)
    per_group_all.to_csv(out / 'per_group_metrics.csv', index=False)

    # ---- main text table: primary scope, 3-seed mean +- sd -----------------
    primary = per_seed[per_seed.scope.eq('dedup')]
    metric_cols = ['abs_mae_mev', 'ec_mae_mev', 'pair_mae_mev',
                   'exact_hit_rate', 'acceptable_1_rate', 'cost_mean_mev', 'cost_median_mev',
                   'cost_p90_mev', 'wrong_pick_rate_gt_10mev']
    main_table = primary.groupby('method')[metric_cols].agg(['mean', 'std']).round(4)
    main_table.columns = [f'{a}_{b}' for a, b in main_table.columns]
    main_table.to_csv(out / 'main_text_table.csv')

    # ---- paired contrasts on the primary scope ----------------------------
    paired_rows, discord_rows = [], []
    contrasts = [
        ('U@0.0001', 'base'), ('U@0.0003', 'base'), ('U@0.001', 'base'),
        ('A@0.0001', 'base'), ('A@0.0003', 'base'), ('A@0.001', 'base'),
        ('T@0.0003', 'A@0.0003'), ('T@0.0003', 'base'),
        ('composition', 'base'), ('adapt', 'base'),
        ('U@0.0001', 'A@0.0001'), ('U@0.0003', 'A@0.0003'), ('U@0.001', 'A@0.001'),
    ]
    for scope in ('raw', 'dedup', 'dedup_no_conflict', 'spread_gt_10'):
        for a, b in contrasts:
            for seed in sorted(predictions.seed.unique()):
                key_a, key_b = (a, seed, scope), (b, seed, scope)
                if key_a not in per_group_frames or key_b not in per_group_frames:
                    continue
                rows, disc = paired_contrast(per_group_frames[key_a], per_group_frames[key_b],
                                             a, b, f'{a} - {b}')
                for row in rows:
                    row.update(scope=scope, seed=seed)
                paired_rows.extend(rows)
                discord_rows.append(dict(scope=scope, seed=seed, contrast=f'{a} - {b}',
                                         n10_exact_hit_a_wrong_b_right=disc[0],
                                         n01_exact_hit_a_right_b_wrong=disc[1],
                                         n10_wrong10_a_wrong_b_right=disc[2],
                                         n01_wrong10_a_right_b_wrong=disc[3],
                                         groups=len(per_group_frames[key_a])))
    pd.DataFrame(paired_rows).to_csv(out / 'paired_metrics.csv', index=False)
    pd.DataFrame(discord_rows).to_csv(out / 'discordant_counts.csv', index=False)

    # ---- tie tolerance curve ----------------------------------------------
    tie_rows = []
    for scope in ('raw', 'dedup', 'spread_gt_10'):
        for (method, seed, sc), frame in per_group_frames.items():
            if sc != scope:
                continue
            for tau in TIE_TAUS:
                tie_rows.append(dict(scope=scope, method=method, seed=seed, tau_mev=tau,
                                     acceptable_rate=float(frame[f'acceptable_{tau:g}'].mean()),
                                     groups=len(frame)))
    pd.DataFrame(tie_rows).to_csv(out / 'tie_tolerance_curve.csv', index=False)

    # ---- P0-5 validation-based learning-rate selection ---------------------
    lr_rows = []
    for route in ('U', 'A'):
        for seed in (20260528, 20260529, 20260530):
            candidates = []
            for lr in LR_GRID:
                if lr == 3e-4:
                    run = EXP12 / 'runs' / f'{route}_{seed}'
                else:
                    run = EXP14 / 'runs' / f'controls_{route}_{seed}_lr{lr:g}'
                history = pd.read_csv(run / 'history.csv')
                best = history.loc[history.val_calibrated_mae.idxmin()]
                candidates.append(dict(route=route, seed=seed, lr=lr,
                                       val_selected_epoch=int(best.epoch),
                                       val_calibrated_mae=float(best.val_calibrated_mae)))
            chosen = min(candidates, key=lambda c: c['val_calibrated_mae'])
            for c in candidates:
                method = f'{route}@{c["lr"]:g}'
                frame = per_group_frames.get((method, seed, 'dedup'))
                c.update(selected_by_val=bool(c is chosen),
                         selected_lr=chosen['lr'],
                         test_ec_mae_mev=(float(frame.ec_abs_sum_mev.sum() / frame.n.sum())
                                          if frame is not None else np.nan),
                         test_wrong_pick_rate=((frame.cost_mev > MATERIAL_TAU).mean()
                                               if frame is not None else np.nan))
                lr_rows.append(c)
    lr_table = pd.DataFrame(lr_rows)
    lr_table.to_csv(out / 'lr_selection_by_validation.csv', index=False)

    # ---- representative examples (rule-based, balanced) -------------------
    example_rows = []
    for seed in (20260528,):
        u = per_group_frames[('U@0.001', seed, 'spread_gt_10')].set_index('group')
        a = per_group_frames[('A@0.001', seed, 'spread_gt_10')].set_index('group')
        both_ok = u[(u.cost_mev <= 0) & (a.cost_mev <= 0)]
        picks = (list(u.cost_mev.nlargest(3).index)
                 + list(a.cost_mev.nlargest(3).index)
                 + list(both_ok.assign(gap=(both_ok.index.map(
                     lambda g: abs(u.loc[g, 'y_spread_mev'] - u.y_spread_mev.median()))))
                     .nsmallest(3).index))
        for group in dict.fromkeys(picks):
            example_rows.append(dict(
                seed=seed, group=group, n=int(u.loc[group, 'n']),
                y_spread_mev=round(float(u.loc[group, 'y_spread_mev']), 3),
                cost_U_mev=round(float(u.loc[group, 'cost_mev']), 3),
                cost_A_mev=round(float(a.loc[group, 'cost_mev']), 3),
                hit_U=int(u.loc[group, 'exact_hit']), hit_A=int(a.loc[group, 'exact_hit']),
                selection_rule='top-3 by cost_U, top-3 by cost_A, 3 median-spread ties'))
    pd.DataFrame(example_rows).to_csv(out / 'representative_groups.csv', index=False)

    print('=== per-seed metrics, scope=dedup (primary) ===')
    show = ['method', 'seed', 'abs_mae_mev', 'ec_mae_mev', 'exact_hit_rate',
            'acceptable_1_rate', 'cost_p90_mev', 'wrong_pick_rate_gt_10mev']
    print(per_seed[per_seed.scope.eq('dedup')][show].round(3).to_string(index=False))
    print('\n=== 验证集选出的学习率（P0-5）===')
    print(lr_table[['route', 'seed', 'lr', 'val_calibrated_mae', 'selected_by_val',
                    'test_ec_mae_mev', 'test_wrong_pick_rate']].round(4).to_string(index=False))
    print('\nwrote', out)


if __name__ == '__main__':
    main()
