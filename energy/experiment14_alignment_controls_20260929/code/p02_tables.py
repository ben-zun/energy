"""P0-2 / P0-3 / P0-5 and batch-2 (1): one script, one definition, all cohorts.

Cohorts, all with the identical metric definitions and the identical P0-1 cleaning
rule:
  matbench   the frozen confirmation, base / composition / T / U@lr / A@lr
  eqv3       the same confirmation structures, residual (aligned) vs direct (unaligned)
  r2scan     the cross-protocol eval split, frozen / U / A

Nothing is retrained and no number is taken from an earlier script.

Every paired number carries an explicit statistic_definition:
  diff_of_group_means       mean_g(x_g) - mean_g(y_g)         (paired resample)
  diff_of_group_medians     median_g(x_g) - median_g(y_g)     (paired resample)
  diff_of_group_quantiles   Q0.9(x_g) - Q0.9(y_g)             descriptive contrast
  quantile_of_group_diffs   Q0.9(x_g - y_g)                   paired, per-group
  rate_difference           mean_g(hit_x) - mean_g(hit_y)     (paired resample)
Binary metrics additionally report exact discordant counts (n10 / n01), because the
bootstrap of a near-degenerate rate difference is otherwise uninformative.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
UNIFIED = EXP14 / 'unified'
EXP12 = ENERGY / 'experiment12_20260929_alignment'

TIE_TAUS = (0.0, 1.0, 2.0, 5.0, 10.0)
MATERIAL_TAU = 10.0
DRAWS, BOOT_SEED = 5000, 20260930
SPREAD_CUTS = (5.0, 10.0, 20.0)
LR_GRID = (1e-4, 3e-4, 1e-3)


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
        pair_err = [abs((yhat[a] - yhat[b]) - (y[a] - y[b])) * 1000.0
                    for a in range(len(y)) for b in range(a + 1, len(y))]
        record = dict(scope=scope, seed=seed, method=method, group=name, n=len(y),
                      y_spread_mev=1000 * float(y.max() - y.min()),
                      ec_abs_sum_mev=float(np.abs(centred_err).sum()) * 1000.0,
                      pair_err_sum_mev=float(np.sum(pair_err)), pair_count=len(pair_err),
                      exact_hit=int(cost <= 1e-9), cost_mev=cost,
                      abs_error_mev=float(np.abs(yhat - y).mean()) * 1000.0)
        for tau in TIE_TAUS:
            record[f'acceptable_{tau:g}'] = int(cost <= tau)
        out.append(record)
    return pd.DataFrame(out)


def seed_metrics(frame, abs_mae_scope_mev, **meta):
    n = frame.n.sum()
    row = dict(**meta, groups=len(frame), structures=int(n),
               pairs=int(frame.pair_count.sum()),
               abs_mae_full_scope_mev=float(abs_mae_scope_mev),
               abs_mae_multigroup_mev=float((frame.abs_error_mev * frame.n).sum() / n),
               ec_mae_mev=float(frame.ec_abs_sum_mev.sum() / n),
               pair_mae_mev=float(frame.pair_err_sum_mev.sum() / frame.pair_count.sum()),
               exact_hit_rate=float(frame.exact_hit.mean()),
               cost_mean_mev=float(frame.cost_mev.mean()),
               cost_median_mev=float(frame.cost_mev.median()),
               cost_p90_mev=float(np.quantile(frame.cost_mev, .90)),
               wrong_pick_rate_gt_10mev=float((frame.cost_mev > MATERIAL_TAU).mean()))
    for tau in TIE_TAUS:
        row[f'acceptable_{tau:g}_rate'] = float(frame[f'acceptable_{tau:g}'].mean())
    return row


def ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def paired_contrast(fa, fb, label_a, label_b, contrast):
    if not (fa.group.to_numpy() == fb.group.to_numpy()).all():
        raise ValueError(f'group order differs in {contrast}')
    rng = np.random.default_rng(BOOT_SEED)
    pick = rng.integers(len(fa), size=(DRAWS, len(fa)))
    ca, cb = fa.cost_mev.to_numpy(), fb.cost_mev.to_numpy()
    na, nb = fa.n.to_numpy(), fb.n.to_numpy()
    eca, ecb = fa.ec_abs_sum_mev.to_numpy(), fb.ec_abs_sum_mev.to_numpy()
    pa, pb = fa.pair_err_sum_mev.to_numpy(), fb.pair_err_sum_mev.to_numpy()
    pca, pcb = fa.pair_count.to_numpy(), fb.pair_count.to_numpy()
    aa, ab = fa.abs_error_mev.to_numpy(), fb.abs_error_mev.to_numpy()
    ha, hb = fa.exact_hit.to_numpy(), fb.exact_hit.to_numpy()
    wa, wb = (ca > MATERIAL_TAU).astype(float), (cb > MATERIAL_TAU).astype(float)

    def mean_diff(x, y):
        return float(np.mean(x) - np.mean(y)), x[pick].mean(axis=1) - y[pick].mean(axis=1)

    def weighted_diff(x, wx, y, wy):
        point = float((x * wx).sum() / wx.sum() - (y * wy).sum() / wy.sum())
        xb, wxb, yb, wyb = x[pick], wx[pick], y[pick], wy[pick]
        dist = xb * wxb
        dist = dist.sum(axis=1) / wxb.sum(axis=1) - (yb * wyb).sum(axis=1) / wyb.sum(axis=1)
        return point, dist

    specs = [
        ('abs_mae_multigroup_mev', 'weighted_diff_of_group_means',
         lambda: weighted_diff(aa, na, ab, nb)),
        # x must be the per-group MEAN; the weight then reproduces Sum/Sum(n)
        ('ec_mae_mev', 'weighted_diff_of_group_means',
         lambda: weighted_diff(eca / na, na, ecb / nb, nb)),
        ('pair_mae_mev', 'weighted_diff_of_group_means',
         lambda: weighted_diff(pa / pca, pca, pb / pcb, pcb)),
        ('exact_hit_rate', 'rate_difference', lambda: mean_diff(ha, hb)),
        ('wrong_pick_rate_gt_10mev', 'rate_difference', lambda: mean_diff(wa, wb)),
        ('acceptable_rate_tau1', 'rate_difference',
         lambda: mean_diff(fa['acceptable_1'].to_numpy(), fb['acceptable_1'].to_numpy())),
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
    rows = []
    for metric, statistic, fn in specs:
        point, dist = fn()
        lo, hi = ci(dist)
        rows.append(dict(contrast=contrast, metric=metric, statistic_definition=statistic,
                         value_a=label_a, value_b=label_b, difference=point,
                         ci95_low=lo, ci95_high=hi, groups=len(fa),
                         excludes_zero=bool(lo > 0 or hi < 0)))
    disc = dict(n10_exact_hit_a_wrong_b_right=int(((ha == 0) & (hb == 1)).sum()),
                n01_exact_hit_a_right_b_wrong=int(((ha == 1) & (hb == 0)).sum()),
                n10_wrong10_a_wrong_b_right=int(((wa == 1) & (wb == 0)).sum()),
                n01_wrong10_a_right_b_wrong=int(((wa == 0) & (wb == 1)).sum()),
                groups=len(fa))
    return rows, disc


def run_cohort(name, manifest, predictions, contrasts):
    truth = manifest.y_true.to_numpy(float)
    groups = manifest.exact_group.to_numpy()
    is_rel = manifest.relative_cohort.to_numpy(bool)
    keep = manifest.keep.to_numpy(bool)
    conflict = manifest.label_conflict.fillna(False).to_numpy(bool)

    scopes = {'raw': is_rel.copy(),
              'dedup': is_rel & keep,
              'dedup_no_conflict': is_rel & keep & (~conflict)}

    frames, seed_rows = {}, []
    for (method, seed), block in predictions.groupby(['method', 'seed'], sort=False):
        block = block.drop_duplicates('id').set_index('id')
        missing = manifest.id[~manifest.id.isin(block.index)]
        if len(missing):
            raise KeyError(f'{name}/{method}/{seed}: {len(missing)} ids missing')
        prediction = block.loc[manifest.id].prediction.to_numpy(float)
        for scope, mask in scopes.items():
            frame = per_group(truth, prediction, groups, mask, seed, method, scope)
            frame['cohort'] = name
            frames[(method, seed, scope)] = frame
            seed_rows.append(seed_metrics(
                frame, 1000 * float(np.abs(prediction[mask] - truth[mask]).mean()),
                cohort=name, method=method, seed=seed, scope=scope))
    for cut in SPREAD_CUTS:
        for (method, seed) in {(m, s) for (m, s, _) in frames}:
            base = frames.get((method, seed, 'dedup'))
            if base is None or len(base) == 0:
                continue
            sub = base[base.y_spread_mev > cut].reset_index(drop=True)
            if len(sub) < 5:
                continue
            sub = sub.copy()
            sub['scope'] = f'spread_gt_{cut:g}'      # slicing keeps the parent's label
            frames[(method, seed, f'spread_gt_{cut:g}')] = sub
            seed_rows.append(seed_metrics(sub, np.nan, cohort=name, method=method,
                                          seed=seed, scope=f'spread_gt_{cut:g}'))

    paired_rows, discord_rows, tie_rows = [], [], []
    for scope in ('raw', 'dedup', 'dedup_no_conflict', 'spread_gt_10'):
        for a, b, seed_only in contrasts:
            seeds = sorted({s for (m, s, sc) in frames if m == a and sc == scope})
            for seed in seeds:
                if seed_only is not None and seed != seed_only:
                    continue
                ka, kb = (a, seed, scope), (b, seed, scope)
                if kb not in frames:
                    continue
                rows, disc = paired_contrast(frames[ka], frames[kb], a, b, f'{a} - {b}')
                for row in rows:
                    row.update(cohort=name, scope=scope, seed=seed)
                paired_rows.extend(rows)
                discord_rows.append(dict(cohort=name, scope=scope, seed=seed,
                                         contrast=f'{a} - {b}', **disc))
    for scope in ('raw', 'dedup', 'spread_gt_10'):
        for (method, seed, sc), frame in frames.items():
            if sc != scope or len(frame) == 0:
                continue
            for tau in TIE_TAUS:
                tie_rows.append(dict(cohort=name, scope=scope, method=method, seed=seed,
                                     tau_mev=tau,
                                     acceptable_rate=float(frame[f'acceptable_{tau:g}'].mean()),
                                     groups=len(frame)))
    return (pd.DataFrame(seed_rows), pd.concat(frames.values(), ignore_index=True),
            pd.DataFrame(paired_rows), pd.DataFrame(discord_rows), pd.DataFrame(tie_rows),
            frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(UNIFIED))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_seed, all_group, all_paired, all_disc, all_tie = [], [], [], [], []
    cohort_frames = {}

    mb_manifest = pd.read_csv(out / 'evaluation_manifest.csv', keep_default_na=False)
    mb_pred = pd.read_csv(out / 'predictions_matbench.csv')
    mb_contrasts = [(f'{r}@{lr:g}', 'base', None) for r in ('U', 'A') for lr in LR_GRID]
    mb_contrasts += [('T@0.0003', 'A@0.0003', None), ('T@0.0003', 'base', None),
                     ('composition', 'base', None)]
    mb_contrasts += [(f'U@{lr:g}', f'A@{lr:g}', None) for lr in LR_GRID]
    parts = run_cohort('matbench', mb_manifest, mb_pred, mb_contrasts)
    cohort_frames['matbench'] = parts[5]
    for store, part in zip((all_seed, all_group, all_paired, all_disc, all_tie), parts[:5]):
        store.append(part)

    eqv3_path = out / 'predictions_eqv3_clean.csv'
    if eqv3_path.exists():
        parts = run_cohort('eqv3', mb_manifest, pd.read_csv(eqv3_path),
                           [('direct', 'residual', None), ('direct', 'clean', None),
                            ('residual', 'clean', None)])
        cohort_frames['eqv3'] = parts[5]
        for store, part in zip((all_seed, all_group, all_paired, all_disc, all_tie), parts[:5]):
            store.append(part)
    else:
        print('WARNING: predictions_eqv3.csv missing -> eqv3 cohort skipped')

    r2_path = EXP14 / 'r2scan10k_full/predictions_r2scan_full.csv'
    if r2_path.exists():
        r2_manifest = pd.read_csv(out / 'evaluation_manifest_r2scan.csv', keep_default_na=False)
        r2_manifest['relative_cohort'] = True
        r2_contrasts = [(f'U_{s}', f'A_{s}', s) for s in (20260528, 20260529, 20260530)]
        r2_contrasts += [('frozen_aligned', 'frozen_raw', None)]
        parts = run_cohort('r2scan', r2_manifest, pd.read_csv(r2_path), r2_contrasts)
        cohort_frames['r2scan'] = parts[5]
        for store, part in zip((all_seed, all_group, all_paired, all_disc, all_tie), parts[:5]):
            store.append(part)
    else:
        print('WARNING: predictions_r2scan.csv missing -> r2scan cohort skipped')

    per_seed = pd.concat(all_seed, ignore_index=True)
    per_seed.to_csv(out / 'per_seed_metrics.csv', index=False)
    groups_all = pd.concat(all_group, ignore_index=True)
    groups_all.to_csv(out / 'per_group_metrics.csv', index=False)
    pd.concat(all_paired, ignore_index=True).to_csv(out / 'paired_metrics.csv', index=False)
    pd.concat(all_disc, ignore_index=True).to_csv(out / 'discordant_counts.csv', index=False)
    pd.concat(all_tie, ignore_index=True).to_csv(out / 'tie_tolerance_curve.csv', index=False)

    metric_cols = ['abs_mae_full_scope_mev', 'abs_mae_multigroup_mev', 'ec_mae_mev', 'pair_mae_mev', 'exact_hit_rate',
                   'acceptable_1_rate', 'cost_mean_mev', 'cost_median_mev',
                   'cost_p90_mev', 'wrong_pick_rate_gt_10mev']
    primary = per_seed[per_seed.scope.eq('dedup')]
    main_table = primary.groupby(['cohort', 'method'])[metric_cols].agg(['mean', 'std']).round(4)
    main_table.columns = [f'{a}_{b}' for a, b in main_table.columns]
    main_table.to_csv(out / 'main_text_table.csv')

    # ---- P0-5: learning rate chosen by VALIDATION, not by test EC ----------
    lr_rows = []
    for route in ('U', 'A'):
        for seed in (20260528, 20260529, 20260530):
            candidates = []
            for lr in LR_GRID:
                run = (EXP12 / 'runs' / f'{route}_{seed}' if lr == 3e-4
                       else EXP14 / 'runs' / f'controls_{route}_{seed}_lr{lr:g}')
                history = pd.read_csv(run / 'history.csv')
                best = history.loc[history.val_calibrated_mae.idxmin()]
                frame = per_seed[(per_seed.cohort.eq('matbench'))
                                 & (per_seed.method.eq(f'{route}@{lr:g}'))
                                 & (per_seed.seed.eq(seed))
                                 & (per_seed.scope.eq('dedup'))]
                candidates.append(dict(
                    route=route, seed=seed, lr=lr,
                    val_selected_epoch=int(best.epoch),
                    val_calibrated_mae=float(best.val_calibrated_mae),
                    test_ec_mae_mev=float(frame.ec_mae_mev.iloc[0]) if len(frame) else np.nan,
                    test_wrong_pick_rate=(float(frame.wrong_pick_rate_gt_10mev.iloc[0])
                                          if len(frame) else np.nan)))
            chosen = min(candidates, key=lambda c: c['val_calibrated_mae'])
            for c in candidates:
                c['selected_by_validation'] = bool(c['lr'] == chosen['lr'])
                c['lr_chosen_by_validation'] = chosen['lr']
                lr_rows.append(c)
    lr_table = pd.DataFrame(lr_rows)
    lr_table.to_csv(out / 'lr_selection_by_validation.csv', index=False)

    # ---- representative groups, balanced explicit rule ---------------------
    example_rows = []
    if 'matbench' in cohort_frames:
        for seed in (20260528,):
            u = cohort_frames['matbench'].get(('U@0.001', seed, 'spread_gt_10'))
            a = cohort_frames['matbench'].get(('A@0.001', seed, 'spread_gt_10'))
            if u is None or a is None:
                continue
            u, a = u.set_index('group'), a.set_index('group')
            both_ok = u[(u.cost_mev <= 0) & (a.cost_mev <= 0)].copy()
            both_ok['dist'] = (both_ok.y_spread_mev - u.y_spread_mev.median()).abs()
            picks = (list(u.cost_mev.nlargest(3).index) + list(a.cost_mev.nlargest(3).index)
                     + list(both_ok.nsmallest(3, 'dist').index))
            for group in dict.fromkeys(picks):
                example_rows.append(dict(
                    seed=seed, group=group, n=int(u.loc[group, 'n']),
                    y_spread_mev=round(float(u.loc[group, 'y_spread_mev']), 3),
                    cost_U_mev=round(float(u.loc[group, 'cost_mev']), 3),
                    cost_A_mev=round(float(a.loc[group, 'cost_mev']), 3),
                    hit_U=int(u.loc[group, 'exact_hit']), hit_A=int(a.loc[group, 'exact_hit']),
                    selection_rule='top-3 by cost_U, top-3 by cost_A, '
                                   '3 ties nearest median spread'))
    pd.DataFrame(example_rows).to_csv(out / 'representative_groups.csv', index=False)

    print('\n=== per-seed metrics, scope=dedup ===')
    show = ['cohort', 'method', 'seed', 'abs_mae_full_scope_mev', 'abs_mae_multigroup_mev', 'ec_mae_mev', 'exact_hit_rate',
            'acceptable_1_rate', 'cost_p90_mev', 'wrong_pick_rate_gt_10mev']
    print(per_seed[per_seed.scope.eq('dedup')][show].round(3).to_string(index=False))
    print('\n=== P0-5: learning rate selected by VALIDATION ===')
    print(lr_table[['route', 'seed', 'lr', 'val_calibrated_mae', 'selected_by_validation',
                    'test_ec_mae_mev', 'test_wrong_pick_rate']].round(4).to_string(index=False))
    print('\nwrote', out)


if __name__ == '__main__':
    main()
