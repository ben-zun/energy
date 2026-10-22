"""Step 4 evaluation: cross-protocol (r2SCAN) U vs A.

Matched U/A runs are scored on the frozen r2SCAN eval split with the same metric
family used for the Matbench analysis:

  absolute MAE              over the whole eval split
  EC-MAE                    within exact-composition groups (multi-structure groups)
  pair MAE                  |predicted (Ei-Ej) - true (Ei-Ej)| within groups
  argmin hit rate           fraction of groups whose predicted minimum is the true one
  acceptable at tau         fraction of groups whose pick is within tau of the minimum
  selection cost r_g        y(pick) - min(y), meV/atom

Baselines: the frozen checkpoint, before and after the routed reference d(z).
All intervals are group-level bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ENERGY = Path('/home/yxzhang/energy')
EXP12_CODE = ENERGY / 'experiment12_20260929_alignment/code'
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
R2 = EXP14 / 'r2scan10k'
sys.path.insert(0, str(EXP12_CODE))
sys.path.insert(0, str(EXP14 / 'code'))

from alignment_common import atomic_fractions, build_model, per_atom  # noqa: E402

SEEDS = (20260528, 20260529, 20260530)
TIE_TAUS = (0.0, 1.0, 2.0, 5.0, 10.0)
MATERIAL_TAU = 10.0
DRAWS, BOOT_SEED = 5000, 20260929
BATCH = 32


def load_split():
    cache = R2 / 'graph_cache'
    meta = pd.read_csv(cache / 'graph_metadata.csv').sort_values('row_index').reset_index(drop=True)
    graphs = {s: torch.load(cache / f'{s}_graphs.pt', map_location='cpu', weights_only=False)
              for s in ('train', 'val', 'test')}
    fit = np.load(R2 / 'r2scan_reference_fit.npz')
    return meta, graphs, fit


def infer(model, graphs, device):
    out = np.full(sum(len(graphs[s]) for s in graphs), np.nan)
    with torch.no_grad():
        for split in graphs:
            for batch in DataLoader(graphs[split], batch_size=BATCH):
                batch = batch.to(device)
                out[batch.manifest_row.cpu().numpy()] = per_atom(model, batch).cpu().numpy()
    if not np.isfinite(out).all():
        raise AssertionError('nonfinite predictions')
    return out


def group_metrics(truth, prediction, groups, keep):
    """Per-group metrics on the rows where keep is True (eval split)."""
    rows = np.flatnonzero(keep)
    frame = pd.DataFrame({'y': truth[rows], 'p': prediction[rows], 'g': groups[rows]})
    records = []
    for name, block in frame.groupby('g', sort=True):
        if len(block) < 2:
            continue
        y = block.y.to_numpy()
        yhat = block.p.to_numpy()
        err = (yhat - yhat.mean()) - (y - y.mean())
        chosen = int(np.argmin(yhat))
        cost = float(y[chosen] - y.min()) * 1000.0
        pair_err, signs = [], []
        for a in range(len(y)):
            for b in range(a + 1, len(y)):
                dy, dhat = y[a] - y[b], yhat[a] - yhat[b]
                pair_err.append(abs(dhat - dy) * 1000.0)
                signs.append((abs(dy) * 1000.0, np.sign(dhat) == np.sign(dy)))
        records.append(dict(group=name, n=len(y), ec_abs_sum=float(np.abs(err).sum()) * 1000.0,
                            pair_err_sum=float(np.sum(pair_err)), pair_count=len(pair_err),
                            exact_hit=int(cost <= 1e-9), cost_mev=cost,
                            **{f'acceptable_{t:g}': int(cost <= t) for t in TIE_TAUS},
                            **{f'order_ok_{t:g}': int(sum(1 for d, ok in signs if d > t and ok))
                               for t in TIE_TAUS},
                            **{f'order_n_{t:g}': int(sum(1 for d, _ in signs if d > t))
                               for t in TIE_TAUS}))
    return pd.DataFrame(records)


def summarise(frame, boot):
    def ci(values):
        v = np.asarray(values)[np.isfinite(values)]
        return float(np.quantile(v, .025)), float(np.quantile(v, .975))
    row = dict(groups=len(frame), structures=int(frame.n.sum()), pairs=int(frame.pair_count.sum()),
               ec_mae=float(frame.ec_abs_sum.sum() / frame.n.sum()),
               pair_mae=float(frame.pair_err_sum.sum() / frame.pair_count.sum()),
               exact_hit_rate=float(frame.exact_hit.mean()),
               acceptable_1_rate=float(frame['acceptable_1'].mean()),
               cost_mean=float(frame.cost_mev.mean()), cost_median=float(frame.cost_mev.median()),
               cost_p90=float(np.quantile(frame.cost_mev, .90)),
               wrong_pick_rate_gt_10mev=float((frame.cost_mev > MATERIAL_TAU).mean()))
    for key in ('ec_mae', 'pair_mae', 'exact_hit', 'acceptable_1', 'cost_mean', 'cost_p90',
                'wrong10'):
        row[f'{key}_lo'], row[f'{key}_hi'] = ci(boot[key])
    return row


def bootstrap(frame, draws=DRAWS, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(frame), size=(draws, len(frame)))
    return dict(
        ec_mae=frame.ec_abs_sum.to_numpy()[pick].sum(axis=1) / frame.n.to_numpy()[pick].sum(axis=1),
        pair_mae=(frame.pair_err_sum.to_numpy()[pick].sum(axis=1)
                  / frame.pair_count.to_numpy()[pick].sum(axis=1)),
        exact_hit=frame.exact_hit.to_numpy()[pick].mean(axis=1),
        acceptable_1=frame['acceptable_1'].to_numpy()[pick].mean(axis=1),
        cost_mean=frame.cost_mev.to_numpy()[pick].mean(axis=1),
        cost_p90=np.quantile(frame.cost_mev.to_numpy()[pick], .90, axis=1),
        wrong10=(frame.cost_mev.to_numpy()[pick] > MATERIAL_TAU).mean(axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tag', default='r2scan')
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    device = torch.device(args.device)

    meta, graphs, fit = load_split()
    truth = meta.y_true.to_numpy()
    fractions = atomic_fractions(meta)
    groups = meta.exact_group.to_numpy()
    is_test = meta.split.to_numpy() == 'test'
    is_train = meta.split.to_numpy() == 'train'
    print(f'eval split: {int(is_test.sum())} structures, '
          f'{pd.Series(groups[is_test]).value_counts().gt(1).sum()} multi-structure groups')

    raw = fit['raw']
    # same convention as alignment_common.alignment_offset: d(z) = -(x @ w + b)
    d = -(fractions @ fit['w'] + float(fit['b']))

    # train-only post-hoc ridge applied identically to every method
    def post_hoc(pred):
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1e-6, fit_intercept=True)
        ridge.fit(fractions[is_train], truth[is_train] - pred[is_train])
        return pred + ridge.predict(fractions)

    methods = {'frozen_raw': post_hoc(raw), 'frozen_aligned': post_hoc(raw + d)}

    for route in ('U', 'A'):
        for seed in SEEDS:
            run = R2 / 'runs' / f'{args.tag}_{route}_{seed}_lr{args.lr:g}'
            status = json.loads((run / 'status.json').read_text())
            best_epoch = int(status['best_epoch'])
            calib = pd.read_csv(run / 'calibration_history.csv')
            row = calib[calib.epoch.eq(best_epoch)].iloc[0]
            w = np.array([row[f'w_{i}'] for i in range(1, 119)], dtype=float)
            b = float(row['b'])
            model = build_model(device)
            payload = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(payload['model'], strict=True)
            model.to(device).eval()
            native = infer(model, graphs, device)
            if route == 'A':
                native = native + d          # route A's prediction includes the routed d(z)
            methods[f'{route}_{seed}'] = native + (fractions @ w + b)
            print(f'{route} seed={seed}: best_epoch={best_epoch} '
                  f'val_calibrated_mae={status["best_val_calibrated_mae"]:.4f}')

    rows, frames = [], {}
    for name, prediction in methods.items():
        overall = 1000 * float(np.abs(prediction[is_test] - truth[is_test]).mean())
        frame = group_metrics(truth, prediction, groups, is_test)
        frames[name] = frame
        rows.append(dict(method=name, eval_abs_mae_mev=overall, **summarise(frame, bootstrap(frame))))
    summary = pd.DataFrame(rows)
    summary.to_csv(R2 / 'results_r2scan_summary.csv', index=False)

    long_rows = []
    for name, prediction in methods.items():
        long_rows.append(pd.DataFrame({'id': meta.id, 'method': name, 'seed': 0,
                                       'prediction': prediction}))
    pd.concat(long_rows, ignore_index=True).to_csv(
        R2 / 'predictions_r2scan.csv', index=False)
    print(summary.round(4).to_string(index=False))

    # paired U - A
    paired = []
    for seed in SEEDS:
        fb = frames[f'U_{seed}'].sort_values('group').reset_index(drop=True)
        fa = frames[f'A_{seed}'].sort_values('group').reset_index(drop=True)
        if not (fb.group.to_numpy() == fa.group.to_numpy()).all():
            raise ValueError('group sets differ between U and A')
        rng = np.random.default_rng(BOOT_SEED)
        pick = rng.integers(len(fb), size=(DRAWS, len(fb)))
        cu, ca = fb.cost_mev.to_numpy()[pick], fa.cost_mev.to_numpy()[pick]
        defs = {
            'ec_mae_mev': (float(fb.ec_abs_sum.sum() / fb.n.sum() - fa.ec_abs_sum.sum() / fa.n.sum()),
                           fb.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fb.n.to_numpy()[pick].sum(axis=1)
                           - fa.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fa.n.to_numpy()[pick].sum(axis=1)),
            'exact_hit_rate': (float(fb.exact_hit.mean() - fa.exact_hit.mean()),
                               fb.exact_hit.to_numpy()[pick].mean(axis=1)
                               - fa.exact_hit.to_numpy()[pick].mean(axis=1)),
            'acceptable_1_rate': (float(fb['acceptable_1'].mean() - fa['acceptable_1'].mean()),
                                  fb['acceptable_1'].to_numpy()[pick].mean(axis=1)
                                  - fa['acceptable_1'].to_numpy()[pick].mean(axis=1)),
            'cost_mean_mev': (float(fb.cost_mev.mean() - fa.cost_mev.mean()),
                              cu.mean(axis=1) - ca.mean(axis=1)),
            'cost_p90_mev': (float(np.quantile(fb.cost_mev, .90) - np.quantile(fa.cost_mev, .90)),
                             np.quantile(cu, .90, axis=1) - np.quantile(ca, .90, axis=1)),
            'wrong_pick_rate_gt_10mev': (float((fb.cost_mev > MATERIAL_TAU).mean()
                                               - (fa.cost_mev > MATERIAL_TAU).mean()),
                                         (cu > MATERIAL_TAU).mean(axis=1)
                                         - (ca > MATERIAL_TAU).mean(axis=1)),
        }
        for metric, (point, dist) in defs.items():
            v = np.asarray(dist)[np.isfinite(dist)]
            lo, hi = float(np.quantile(v, .025)), float(np.quantile(v, .975))
            paired.append(dict(seed=seed, metric=metric, difference_u_minus_a=point,
                               ci95_low=lo, ci95_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    paired = pd.DataFrame(paired)
    paired.to_csv(R2 / 'results_r2scan_paired.csv', index=False)
    print('\nU - A paired:')
    print(paired.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
