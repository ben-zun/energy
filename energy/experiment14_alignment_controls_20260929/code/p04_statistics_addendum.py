"""Audit-response statistics: exact tests, cross-seed means, and explicit settings.

Addresses three findings from the external audit:
  S1  the binary-metric claims must be accompanied by an exact paired test
      (two-sided McNemar), which for the "wrong pick > 10 meV" endpoint does NOT
      reach 0.05 in seed 20260529;
  S2  a single seed's group-level interval must not be attached to the three-seed
      mean; the cross-seed mean needs its own interval, and the three seeds share
      the SAME evaluation groups so group uncertainty is not independent evidence;
  S3  the bootstrap settings (draws, interval type, seed) and the multiplicity
      situation must be stated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ENERGY = Path('/home/yxzhang/energy')
UNIFIED = ENERGY / 'experiment14_alignment_controls_20260929/unified'
DRAWS, BOOT_SEED = 5000, 20260930
SEEDS = (20260528, 20260529, 20260530)

PRIMARY = [('U@0.001', 'A@0.001'), ('T@0.0003', 'A@0.0003'), ('A@0.001', 'base'),
           ('composition', 'base')]
METRICS = ['ec_mae_mev', 'pair_mae_mev', 'exact_hit_rate', 'acceptable_rate_tau1',
           'cost_mean_mev', 'cost_p90_mev', 'wrong_pick_rate_gt_10mev']


def mcnemar_exact(n10, n01):
    if n10 + n01 == 0:
        return 1.0
    return float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)


def main():
    paired = pd.read_csv(UNIFIED / 'paired_metrics.csv')
    discord = pd.read_csv(UNIFIED / 'discordant_counts.csv')

    out = {'settings': dict(draws=DRAWS, interval='percentile 2.5/97.5',
                            boot_seed=BOOT_SEED,
                            resample_unit='exact-composition group',
                            multiplicity='none applied; EC-MAE and wrong-pick rate are the '
                                         'pre-specified primary endpoints',
                            note='the three training seeds share one evaluation group set, '
                                 'so group-level uncertainty is common across seeds')}
    rows, tests = [], []

    for a, b in PRIMARY:
        contrast = f'{a} - {b}'
        for metric in METRICS:
            sub = paired[(paired.scope.eq('dedup')) & (paired.contrast.eq(contrast))
                         & (paired.metric.eq(metric))]
            if len(sub) == 0:
                continue
            per_seed = sub.sort_values('seed').difference.to_numpy()
            mean = float(np.mean(per_seed))
            sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float('nan')
            if len(per_seed) > 1 and sd > 0:
                half = float(stats.t.ppf(0.975, len(per_seed) - 1) * sd / np.sqrt(len(per_seed)))
            else:
                half = float('nan')
            rows.append(dict(
                contrast=contrast, metric=metric,
                statistic=sub.statistic_definition.iloc[0],
                per_seed_list=json.dumps([round(float(v), 4) for v in per_seed]),
                cross_seed_mean=mean, cross_seed_sd=sd,
                cross_seed_ci_low=mean - half if np.isfinite(half) else np.nan,
                cross_seed_ci_high=mean + half if np.isfinite(half) else np.nan,
                ci_basis=f't(0.975, df={len(per_seed) - 1}) over {len(per_seed)} training seeds',
                seed28_group_ci_low=float(sub[sub.seed.eq(SEEDS[0])].ci95_low.iloc[0]),
                seed28_group_ci_high=float(sub[sub.seed.eq(SEEDS[0])].ci95_high.iloc[0]),
                n_groups=int(sub.groups.iloc[0])))

    for a, b in PRIMARY:
        contrast = f'{a} - {b}'
        for label, cols in (('exact_hit', ('n10_exact_hit_a_wrong_b_right',
                                           'n01_exact_hit_a_right_b_wrong')),
                            ('wrong_pick_gt_10mev', ('n10_wrong10_a_wrong_b_right',
                                                     'n01_wrong10_a_right_b_wrong'))):
            sub = discord[(discord.scope.eq('dedup')) & (discord.contrast.eq(contrast))]
            for _, r in sub.sort_values('seed').iterrows():
                n10, n01 = int(r[cols[0]]), int(r[cols[1]])
                tests.append(dict(contrast=contrast, metric=label, seed=int(r.seed),
                                  n10=n10, n01=n01,
                                  discordant_total=n10 + n01,
                                  mcnemar_exact_two_sided=mcnemar_exact(n10, n01),
                                  groups=int(r.groups)))

    summary = pd.DataFrame(rows)
    tests = pd.DataFrame(tests)
    summary.to_csv(UNIFIED / 'cross_seed_summary.csv', index=False)
    tests.to_csv(UNIFIED / 'mcnemar_exact.csv', index=False)

    print('=== cross-seed mean vs single-seed group interval (dedup) ===')
    show = summary[summary.metric.isin(['ec_mae_mev', 'exact_hit_rate',
                                        'wrong_pick_rate_gt_10mev'])]
    print(show[['contrast', 'metric', 'per_seed_list', 'cross_seed_mean', 'cross_seed_sd',
                'cross_seed_ci_low', 'cross_seed_ci_high', 'seed28_group_ci_low',
                'seed28_group_ci_high']].round(4).to_string(index=False))
    print('\n=== two-sided exact McNemar ===')
    print(tests[tests.contrast.isin(['U@0.001 - A@0.001', 'T@0.0003 - A@0.0003',
                                     'A@0.001 - base'])].round(5).to_string(index=False))
    (UNIFIED / 'stats_settings.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print('\nwrote', UNIFIED / 'cross_seed_summary.csv', 'and mcnemar_exact.csv')


if __name__ == '__main__':
    main()
