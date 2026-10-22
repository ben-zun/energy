"""Evaluate the experiment14 controls on the frozen confirmation set.

Covers, all under the identical frozen protocol (train-only post-hoc ridge, the
same confirmation cohort, group-level bootstrap):

  T           trainable element reference, lr 3e-4      (experiment14)
  U, A        lr 1e-4 and 1e-3                          (experiment14)
  U, A        lr 3e-4, the original matched runs        (reused from experiment12)
  base/base+hT                                          (experiment11 artifacts)

and reports both the energy metrics and the structure-selection metrics, so the
learning-rate budget question is answered on the metric that carries material
meaning, not only on the validation MAE.
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
EXP11 = ENERGY / 'experiment11_20260928'
CONF = EXP11 / 'confirmation'
EXP12 = ENERGY / 'experiment12_20260929_alignment'
EXP12_CODE = EXP12 / 'code'
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
sys.path.insert(0, str(EXP12_CODE))
sys.path.insert(0, str(EXP14 / 'code'))

from alignment_common import atomic_fractions, build_model, group_labels, per_atom  # noqa: E402
from evaluate_alignment import confirmation_setup  # noqa: E402
from train_controls import ReferenceTerm  # noqa: E402

SEEDS = (20260528, 20260529, 20260530)
TIERS = [('T', 3e-4), ('U', 1e-4), ('A', 1e-4), ('U', 3e-4), ('A', 3e-4),
         ('U', 1e-3), ('A', 1e-3)]
TIE_TAUS = (0.0, 1.0, 2.0, 5.0, 10.0)
MATERIAL_TAU = 10.0
DRAWS, BOOT_SEED = 5000, 20260928
BATCH = 32


def exp14_run(route, seed, lr):
    return EXP14 / 'runs' / f'controls_{route}_{seed}_lr{lr:g}'


def exp12_run(route, seed):
    return EXP12 / 'runs' / f'{route}_{seed}'


def forward_confirmation(route, model, reference, graphs, offset, fractions, device):
    fractions_t = torch.tensor(fractions, dtype=torch.float32, device=device)
    offset_t = torch.tensor(offset, dtype=torch.float32, device=device)
    out = np.full(len(fractions), np.nan)
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=BATCH):
            batch = batch.to(device)
            energy = per_atom(model, batch)
            rows = batch.manifest_row.long()
            if route == 'U':
                prediction = energy
            elif route == 'A':
                prediction = energy + offset_t[rows]
            elif route == 'T':
                prediction = energy + reference(fractions_t[rows])
            else:
                raise ValueError(route)
            out[batch.manifest_row.cpu().numpy()] = prediction.cpu().numpy()
    if not np.isfinite(out).all():
        raise AssertionError('nonfinite confirmation predictions')
    return out


def load_run(route, seed, lr, model, graphs, offset, fractions, device):
    """Return (native prediction, calibrated prediction, provenance dict)."""
    if lr == 3e-4 and route in ('U', 'A'):
        run = exp12_run(route, seed)
        selected = pd.read_csv(EXP12 / 'results' / 'selected_checkpoints.csv')
        epoch = int(selected[(selected.comparison.eq('final_best')) & (selected.route.eq(route))
                             & (selected.seed.eq(seed))].epoch.iloc[0])
        payload = torch.load(run / f'epoch{epoch:03d}.pt', map_location='cpu', weights_only=False)
        origin = 'experiment12'
    else:
        run = exp14_run(route, seed, lr)
        status = json.loads((run / 'status.json').read_text())
        epoch = int(status['best_epoch'])
        payload = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        origin = 'experiment14'
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    reference = None
    if route == 'T':
        if 'reference' not in payload:
            raise KeyError('T checkpoint has no reference weights')
        reference = ReferenceTerm(np.zeros(118), 0.0)
        reference.load_state_dict(payload['reference'])
        reference.to(device).eval()
    native = forward_confirmation(route, model, reference, graphs, offset, fractions, device)
    calibration = pd.read_csv(run / 'calibration_history.csv')
    row = calibration[calibration.epoch.eq(epoch)].iloc[0]
    w = np.array([row[f'w_{i}'] for i in range(1, 119)], dtype=float)
    b = float(row['b'])
    calibrated = native + (fractions @ w + b)
    reference_drift = float('nan')
    if route == 'T':
        history = pd.read_csv(run / 'history.csv')
        reference_drift = float(history[history.epoch.eq(epoch)].reference_drift_norm.iloc[0])
    return native, calibrated, dict(origin=origin, epoch=epoch, reference_drift=reference_drift)


def selection_metrics(truth, prediction, groups, keep):
    rows = np.flatnonzero(keep)
    frame = pd.DataFrame({'y': truth[rows], 'p': prediction[rows], 'g': groups[rows]})
    records = []
    for name, block in frame.groupby('g', sort=True):
        if len(block) < 2:
            continue
        y, yhat = block.y.to_numpy(), block.p.to_numpy()
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


def ci(values):
    v = np.asarray(values)[np.isfinite(values)]
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)

    graphs, manifest, fractions, offset = confirmation_setup()
    truth = manifest.y_true.to_numpy()
    groups = manifest.exact_group.to_numpy()
    overall = manifest.overall_cohort.to_numpy(bool)
    relative = manifest.relative_cohort.to_numpy(bool)
    print(f'confirmation {len(manifest)} structures, {int(relative.sum())} in the relative cohort',
          flush=True)

    model = build_model(device)
    methods, frames = {}, {}
    # frozen baselines from the experiment11 artifacts
    for seed in SEEDS:
        sn = pd.read_csv(CONF / f'sevennet_{seed}.csv')
        methods[('base', seed)] = sn.base.to_numpy(float)
        methods[('composition', seed)] = sn.composition.to_numpy(float)

    provenance = []
    for route, lr in TIERS:
        for seed in SEEDS:
            native, calibrated, info = load_run(route, seed, lr, model, graphs, offset,
                                                fractions, device)
            methods[(f'{route}@{lr:g}', seed)] = calibrated
            provenance.append(dict(tier=f'{route}@{lr:g}', route=route, lr=lr, seed=seed, **info))
            print(f'{route}@{lr:g} seed={seed}: origin={info["origin"]} epoch={info["epoch"]}',
                  flush=True)
    pd.DataFrame(provenance).to_csv(EXP14 / 'results_provenance.csv', index=False)

    rows = []
    for (name, seed), prediction in methods.items():
        relative_abs = 1000 * float(np.abs(prediction[relative] - truth[relative]).mean())
        frame = selection_metrics(truth, prediction, groups, relative)
        frames[(name, seed)] = frame
        boot = bootstrap(frame)
        entry = dict(method=name, seed=seed,
                     overall_abs_mae=1000 * float(np.abs(prediction[overall] - truth[overall]).mean()),
                     relative_abs_mae=relative_abs,
                     ec_mae=float(frame.ec_abs_sum.sum() / frame.n.sum()),
                     pair_mae=float(frame.pair_err_sum.sum() / frame.pair_count.sum()),
                     exact_hit_rate=float(frame.exact_hit.mean()),
                     acceptable_1_rate=float(frame['acceptable_1'].mean()),
                     cost_mean=float(frame.cost_mev.mean()),
                     cost_p90=float(np.quantile(frame.cost_mev, .90)),
                     wrong_pick_rate_gt_10mev=float((frame.cost_mev > MATERIAL_TAU).mean()))
        for key in ('ec_mae', 'pair_mae', 'exact_hit', 'acceptable_1', 'cost_mean', 'cost_p90',
                    'wrong10'):
            entry[f'{key}_lo'], entry[f'{key}_hi'] = ci(boot[key])
        rows.append(entry)
    summary = pd.DataFrame(rows)
    summary.to_csv(EXP14 / 'results_controls_summary.csv', index=False)

    long_rows = []
    for (name, seed), prediction in methods.items():
        long_rows.append(pd.DataFrame({'id': manifest.id, 'method': name, 'seed': seed,
                                       'prediction': prediction}))
    (EXP14 / 'unified').mkdir(parents=True, exist_ok=True)
    pd.concat(long_rows, ignore_index=True).to_csv(
        EXP14 / 'unified' / 'predictions_matbench.csv', index=False)

    show = ['method', 'seed', 'overall_abs_mae', 'relative_abs_mae', 'ec_mae', 'exact_hit_rate',
            'acceptable_1_rate', 'cost_p90', 'wrong_pick_rate_gt_10mev']
    print('\n=== 控制组：冻结确认集 ===')
    print(summary[show].round(3).to_string(index=False))

    # paired contrasts against the aligned baseline A@3e-4
    paired = []
    for name in sorted({n for n, _ in methods}):
        if name in ('base', 'composition'):
            continue
        for seed in SEEDS:
            fb = frames[(name, seed)].sort_values('group').reset_index(drop=True)
            fa = frames[('A@0.0003', seed)].sort_values('group').reset_index(drop=True)
            if not (fb.group.to_numpy() == fa.group.to_numpy()).all():
                raise ValueError('group sets differ')
            rng = np.random.default_rng(BOOT_SEED)
            pick = rng.integers(len(fb), size=(DRAWS, len(fb)))
            cu, ca = fb.cost_mev.to_numpy()[pick], fa.cost_mev.to_numpy()[pick]
            for metric, point, dist in (
                    ('ec_mae_mev',
                     float(fb.ec_abs_sum.sum() / fb.n.sum() - fa.ec_abs_sum.sum() / fa.n.sum()),
                     fb.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fb.n.to_numpy()[pick].sum(axis=1)
                     - fa.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fa.n.to_numpy()[pick].sum(axis=1)),
                    ('wrong_pick_rate_gt_10mev',
                     float((fb.cost_mev > MATERIAL_TAU).mean() - (fa.cost_mev > MATERIAL_TAU).mean()),
                     (cu > MATERIAL_TAU).mean(axis=1) - (ca > MATERIAL_TAU).mean(axis=1)),
                    ('cost_p90_mev',
                     float(np.quantile(fb.cost_mev, .90) - np.quantile(fa.cost_mev, .90)),
                     np.quantile(cu, .90, axis=1) - np.quantile(ca, .90, axis=1))):
                lo, hi = ci(dist)
                paired.append(dict(method=name, seed=seed, metric=metric, difference=point,
                                   ci95_low=lo, ci95_high=hi,
                                   excludes_zero=bool(lo > 0 or hi < 0)))
    paired = pd.DataFrame(paired)
    paired.to_csv(EXP14 / 'results_controls_paired.csv', index=False)
    print('\n=== 相对 A@3e-4 的配对差（负值 = 该方法更好）===')
    print(paired.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
