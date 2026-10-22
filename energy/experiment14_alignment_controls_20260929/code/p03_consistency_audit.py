"""Independent consistency audit of every number the unified pipeline produced.

Each check recomputes a quantity a SECOND way (or checks an identity that must
hold), and fails loudly instead of silently passing.  Nothing here trusts the
pipeline's own aggregation.

Checks
  A1  per-group rows aggregate to the per-seed row (structures, EC-MAE, pair MAE,
      abs MAE, hit rate, cost mean/median/p90, wrong-pick rate)
  A2  scope nesting: dedup g <= raw g, dedup_no_conflict g <= dedup g, and the
      spread subsets are literal subsets of dedup
  A3  composition == base on every within-group centred quantity (algebraic identity)
  A4  paired point estimates equal the difference of the per-seed values
  A5  bootstrap CI brackets the point estimate for mean/rate statistics
  A6  discordant counts are consistent with the group counts
  A7  the manifest's keep / exclusion_reason flags are self-consistent and the
      cleaned group sizes match the cleaning report
  A8  no method silently inherits another method's predictions (spot check that
      U and A differ, and that base != composition on absolute MAE)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
UNIFIED = ENERGY / 'experiment14_alignment_controls_20260929/unified'
TOL = 1e-6

problems = []
checks = []


def check(name, ok, detail=''):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    if not ok:
        problems.append(f'{name}: {detail}')


def main():
    per_seed = pd.read_csv(UNIFIED / 'per_seed_metrics.csv')
    per_group = pd.read_csv(UNIFIED / 'per_group_metrics.csv')
    paired = pd.read_csv(UNIFIED / 'paired_metrics.csv')
    discord = pd.read_csv(UNIFIED / 'discordant_counts.csv')

    # ---------------- A1 per-group -> per-seed --------------------------
    worst = 0.0
    worst_where = ''
    for _, row in per_seed.iterrows():
        sub = per_group[(per_group.cohort.eq(row.cohort)) & (per_group.method.eq(row.method))
                        & (per_group.seed.eq(row.seed)) & (per_group.scope.eq(row.scope))]
        if len(sub) == 0:
            continue
        n = sub.n.sum()
        recomputed = {
            'structures': int(n),
            'ec_mae_mev': sub.ec_abs_sum_mev.sum() / n,
            'pair_mae_mev': sub.pair_err_sum_mev.sum() / sub.pair_count.sum(),
            'abs_mae_multigroup_mev': float((sub.abs_error_mev * sub.n).sum() / n),
            'exact_hit_rate': sub.exact_hit.mean(),
            'cost_mean_mev': sub.cost_mev.mean(),
            'cost_median_mev': sub.cost_mev.median(),
            'cost_p90_mev': np.quantile(sub.cost_mev, .90),
            'wrong_pick_rate_gt_10mev': (sub.cost_mev > 10.0).mean(),
        }
        for key, value in recomputed.items():
            recorded = row[key]
            if not np.isfinite(recorded):
                continue
            deviation = abs(float(value) - float(recorded))
            if deviation > worst:
                worst = deviation
                worst_where = (f'{row.cohort}/{row.method}/seed{row.seed}/{row.scope}/{key} '
                               f'recomputed={float(value):.6f} recorded={float(recorded):.6f}')
    check('A1 per-group aggregates reproduce per-seed rows', worst < 1e-6,
          f'max abs deviation {worst:.3e} at {worst_where}')

    # ---------------- A2 scope nesting ----------------------------------
    ok = True
    detail = []
    for (cohort, method, seed), _ in per_seed.groupby(['cohort', 'method', 'seed']):
        sets = {}
        for scope in ('raw', 'dedup', 'dedup_no_conflict', 'spread_gt_5', 'spread_gt_10',
                      'spread_gt_20'):
            sub = per_group[(per_group.cohort.eq(cohort)) & (per_group.method.eq(method))
                            & (per_group.seed.eq(seed)) & (per_group.scope.eq(scope))]
            sets[scope] = set(sub.group)
        if not sets['dedup'] <= sets['raw']:
            ok = False
            detail.append(f'{cohort}/{method}/{seed}: dedup not subset of raw')
        if not sets['dedup_no_conflict'] <= sets['dedup']:
            ok = False
            detail.append(f'{cohort}/{method}/{seed}: no_conflict not subset of dedup')
        for scope in ('spread_gt_5', 'spread_gt_10', 'spread_gt_20'):
            if sets[scope] and not sets[scope] <= sets['dedup']:
                ok = False
                detail.append(f'{cohort}/{method}/{seed}: {scope} not subset of dedup')
    check('A2 scopes nest correctly', ok, '; '.join(detail[:4]))

    # ---------------- A3 composition == base on centred stats -----------
    ok = True
    detail = []
    for scope in ('raw', 'dedup'):
        for seed in sorted(per_group.seed.unique()):
            b = per_group[(per_group.cohort.eq('matbench')) & (per_group.method.eq('base'))
                          & (per_group.seed.eq(seed)) & (per_group.scope.eq(scope))]
            c = per_group[(per_group.cohort.eq('matbench')) & (per_group.method.eq('composition'))
                          & (per_group.seed.eq(seed)) & (per_group.scope.eq(scope))]
            if len(b) == 0 or len(c) == 0:
                continue
            b, c = b.sort_values('group'), c.sort_values('group')
            for column in ('ec_abs_sum_mev', 'pair_err_sum_mev', 'cost_mev', 'exact_hit'):
                dev = float(np.abs(b[column].to_numpy() - c[column].to_numpy()).max())
                if dev > 1e-9:
                    ok = False
                    detail.append(f'{scope}/seed{seed}/{column}: {dev:.2e}')
    check('A3 composition is identical to base under group centring', ok, '; '.join(detail[:4]))

    # ---------------- A4 paired point = difference of per-seed ----------
    ok = True
    detail = []
    for _, row in paired.iterrows():
        if row.statistic_definition not in ('diff_of_group_means',
                                            'weighted_diff_of_group_means'):
            continue
        a = per_seed[(per_seed.cohort.eq(row.cohort)) & (per_seed.method.eq(row.value_a))
                     & (per_seed.seed.eq(row.seed)) & (per_seed.scope.eq(row.scope))]
        b = per_seed[(per_seed.cohort.eq(row.cohort)) & (per_seed.method.eq(row.value_b))
                     & (per_seed.seed.eq(row.seed)) & (per_seed.scope.eq(row.scope))]
        if len(a) == 0 or len(b) == 0 or row.metric not in per_seed.columns:
            continue
        expected = float(a[row.metric].iloc[0] - b[row.metric].iloc[0])
        if abs(expected - row.difference) > 1e-6:
            ok = False
            detail.append(f'{row.contrast}/{row.metric}/{row.seed}: '
                          f'{expected:.6f} vs {row.difference:.6f}')
    check('A4 paired point estimates match the difference of per-seed values', ok,
          '; '.join(detail[:4]))

    # ---------------- A5 CI brackets the point estimate -----------------
    ok = True
    detail = []
    for _, row in paired.iterrows():
        if row.statistic_definition == 'quantile_of_group_diffs':
            continue                       # not a difference of per-seed aggregates
        if not (row.ci95_low - 1e-6 <= row.difference <= row.ci95_high + 1e-6):
            ok = False
            detail.append(f'{row.contrast}/{row.metric}/{row.seed}/'
                          f'{row.statistic_definition}')
    check('A5 bootstrap interval brackets the point estimate', ok, '; '.join(detail[:4]))

    # ---------------- A6 discordant counts ------------------------------
    ok = True
    detail = []
    for _, row in discord.iterrows():
        if not (0 <= row.n10_exact_hit_a_wrong_b_right <= row.groups
                and 0 <= row.n01_exact_hit_a_right_b_wrong <= row.groups):
            ok = False
            detail.append(f'{row.contrast}/{row.seed}: counts exceed group count')
        if row.n10_exact_hit_a_wrong_b_right + row.n01_exact_hit_a_right_b_wrong > row.groups:
            ok = False
            detail.append(f'{row.contrast}/{row.seed}: discordant pairs exceed groups')
    check('A6 discordant counts are internally consistent', ok, '; '.join(detail[:4]))

    # ---------------- A7 manifest self-consistency ----------------------
    manifest = pd.read_csv(UNIFIED / 'evaluation_manifest.csv', keep_default_na=False)
    report = json.loads((UNIFIED / 'cleaning_report.json').read_text())
    surv = manifest.groupby('exact_group').keep.sum()
    usable = surv[surv >= 2]
    # note: keep.sum() counts structures whose group may still have <2 survivors,
    # so it is NOT the dedup structure count -- the usable groups are.
    ok = (int(usable.sum()) == report['matbench']['scopes']['dedup']['structures']
          and int(len(usable)) == report['matbench']['scopes']['dedup']['groups']
          and int(manifest.exclusion_reason.str.startswith('R2_').sum())
          == report['matbench']['R2_strict_equivalence'])
    check('A7 cleaning report matches the manifest', ok,
          f'usable_structures={int(usable.sum())} usable_groups={int(len(usable))} '
          f'report={report["matbench"]["scopes"]["dedup"]}')
    kept_with_reason = int((manifest.keep & (manifest.exclusion_reason != '')).sum())
    check('A7b no structure is kept with an exclusion reason', kept_with_reason == 0,
          f'{kept_with_reason} violations')

    # ---------------- A8 methods are not accidentally identical ---------
    for scope in ('dedup',):
        u = per_group[(per_group.cohort.eq('matbench')) & (per_group.method.eq('U@0.001'))
                      & (per_group.scope.eq(scope))].sort_values('group').reset_index(drop=True)
        a = per_group[(per_group.cohort.eq('matbench')) & (per_group.method.eq('A@0.001'))
                      & (per_group.scope.eq(scope))].sort_values('group').reset_index(drop=True)
        check('A8 U and A predictions are not identical',
              bool(len(u) == len(a) and not np.allclose(u.cost_mev, a.cost_mev)),
              'cost vectors differ' if len(u) == len(a) else 'different group counts')
    b = per_seed[(per_seed.cohort.eq('matbench')) & (per_seed.method.eq('base'))
                 & (per_seed.scope.eq('dedup'))].abs_mae_full_scope_mev.mean()
    c = per_seed[(per_seed.cohort.eq('matbench')) & (per_seed.method.eq('composition'))
                 & (per_seed.scope.eq('dedup'))].abs_mae_full_scope_mev.mean()
    check('A8b composition changes the absolute MAE but not the relative metrics',
          bool(abs(b - c) > 10), f'base={b:.3f} composition={c:.3f}')

    summary = dict(checks=checks, n_checks=len(checks),
                   n_failed=len(problems), problems=problems)
    (UNIFIED / 'consistency_audit.json').write_text(json.dumps(summary, indent=2,
                                                               ensure_ascii=False))
    for entry in checks:
        print(('PASS ' if entry['passed'] else 'FAIL ') + entry['check']
              + (f"   [{entry['detail']}]" if entry['detail'] else ''))
    print(f'\n{len(checks) - len(problems)}/{len(checks)} checks passed')
    if problems:
        print('\nPROBLEMS:')
        for p in problems:
            print(' -', p)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
