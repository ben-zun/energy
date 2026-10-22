"""Batch-2 (2): Smooth L1 saturation fraction, and a check on the mechanism story.

For every epoch of every matched SevenNet run we compute the fraction of validation
structures whose residual exceeds the loss transition point beta:

    s_t = #{ i : |yhat_i - y_i| > beta } / N ,      beta = HUBER_BETA = 0.05 eV/atom

Crucially yhat here is the prediction that ACTUALLY ENTERS THE LOSS (the route's
native prediction saved per epoch), not the post-hoc calibrated prediction.

We also record, per epoch:
  val_native_mae, val_calibrated_mae   from history.csv
  grad_norm_first, grad_norm_mean      L2 norm of the gradient over the main network
  param_delta_norm                     see definition below

Definition of the parameter-update quantities
  param_delta_norm(t)  = || theta_end(t) - theta_start(t) ||_2 over the main network's
                         trainable parameters only (the routed reference is excluded;
                         for route T it is logged separately as reference_drift_norm).
  "total update"       = sum_t param_delta_norm(t): a path length, NOT a displacement.
  true displacement    = || theta(epoch_last) - theta(epoch_1) ||_2, read directly from
                         the saved checkpoints.  Both are reported so the difference
                         between "how far it moved" and "how much it travelled" is explicit.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
EXP12 = ENERGY / 'experiment12_20260929_alignment'
MEDIUM_META = (ENERGY / 'runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision'
               / 'fair_finetune_293/graph_cache/medium/graph_metadata.csv')
BETA = 0.05
LR_GRID = (1e-4, 3e-4, 1e-3)


def run_dir(route, seed, lr):
    if lr == 3e-4:
        return EXP12 / 'runs' / f'{route}_{seed}'
    return EXP14 / 'runs' / f'controls_{route}_{seed}_lr{lr:g}'


def epoch_paths(run):
    out = {}
    for path in sorted(glob.glob(str(run / 'val_epoch*.npz'))):
        epoch = int(re.search(r'val_epoch(\d+)\.npz', path).group(1))
        out[epoch] = Path(path)
    return out


def displacement(run, first, last):
    a = torch.load(run / f'epoch{first:03d}.pt', map_location='cpu', weights_only=False)
    b = torch.load(run / f'epoch{last:03d}.pt', map_location='cpu', weights_only=False)
    names = sorted(a['model'].keys())
    va = torch.cat([a['model'][n].reshape(-1).float() for n in names])
    vb = torch.cat([b['model'][n].reshape(-1).float() for n in names])
    return float((vb - va).norm())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(EXP14 / 'unified'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(MEDIUM_META)
    truth = meta.y_true.to_numpy(float)

    rows, summary = [], []
    for route in ('U', 'A'):
        for seed in (20260528, 20260529, 20260530):
            for lr in LR_GRID:
                run = run_dir(route, seed, lr)
                if not (run / 'history.csv').exists():
                    continue
                history = pd.read_csv(run / 'history.csv').set_index('epoch')
                paths = epoch_paths(run)
                if not paths:
                    continue
                per_epoch = []
                for epoch, path in paths.items():
                    payload = np.load(path)
                    native = payload['native'].astype(float)
                    rows_idx = payload['manifest_row']
                    residual = np.abs(native - truth[rows_idx])
                    saturation = float((residual > BETA).mean())
                    hist = history.loc[epoch] if epoch in history.index else None

                    def hcol(name):
                        # gradient / update columns exist only for the experiment14 runs
                        if hist is None or name not in hist.index:
                            return np.nan
                        return float(hist[name])

                    per_epoch.append(dict(
                        route=route, seed=seed, lr=lr, epoch=epoch,
                        saturation_fraction=saturation,
                        val_native_mae_mev=hcol('val_native_mae'),
                        val_calibrated_mae_mev=hcol('val_calibrated_mae'),
                        grad_norm_first=hcol('grad_norm_first'),
                        grad_norm_mean=hcol('grad_norm_mean'),
                        param_delta_norm=hcol('param_delta_norm')))
                frame = pd.DataFrame(per_epoch)
                rows.extend(per_epoch)
                epochs = sorted(paths)
                entry = dict(route=route, seed=seed, lr=lr, epochs=len(epochs),
                             saturation_epoch1=float(frame.saturation_fraction.iloc[0]),
                             saturation_epoch_last=float(frame.saturation_fraction.iloc[-1]),
                             saturation_median=float(frame.saturation_fraction.median()),
                             param_delta_sum=(float(frame.param_delta_norm.sum())
                                              if frame.param_delta_norm.notna().any() else np.nan),
                             grad_mean_median=float(frame.grad_norm_mean.median()),
                             grad_first_epoch1=float(frame.grad_norm_first.iloc[0]))
                try:
                    entry['true_displacement'] = displacement(run, epochs[0], epochs[-1])
                except Exception as exc:                              # noqa: BLE001
                    entry['true_displacement'] = np.nan
                    print(f'displacement failed for {run}: {exc}')
                summary.append(entry)

    curves = pd.DataFrame(rows)
    curves.to_csv(out / 'saturation_curve.csv', index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'saturation_summary.csv', index=False)

    print('=== 饱和比例与更新幅度（按路线/学习率，3 seed 均值）===')
    agg = summary.groupby(['route', 'lr']).agg(
        sat_epoch1=('saturation_epoch1', 'mean'),
        sat_last=('saturation_epoch_last', 'mean'),
        sat_median=('saturation_median', 'mean'),
        grad_median=('grad_mean_median', 'mean'),
        update_path_sum=('param_delta_sum', 'mean'),
        true_displacement=('true_displacement', 'mean')).round(4)
    print(agg.to_string())
    print('\n注：update_path_sum = Σ_t ||θ_end(t)−θ_start(t)||（路径长度）；'
          'true_displacement = ||θ_last − θ_first||（真实位移）。两者不同。')
    print('\nwrote', out)


if __name__ == '__main__':
    main()
