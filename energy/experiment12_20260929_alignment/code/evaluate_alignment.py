"""Three-class checkpoint comparison and frozen-confirmation evaluation for U vs A.

Class 1 (same budget)   : best-so-far checkpoint at budgets 20/40/60/80/120
Class 2 (final best)    : best checkpoint within the full 120-epoch budget
Class 3 (matched accuracy): thresholds reachable by both routes; first epoch reaching each

Checkpoint selection always uses the unified post-calibration validation MAE
(argmin, ties -> earlier epoch).  The confirmation set is evaluated only after
selection is frozen, and the post-hoc ridge is the one already fitted on the
medium train split for that epoch (no confirmation labels are ever fitted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from alignment_common import (BATCH_EVAL, BUDGETS, CONFIRMATION, OUT, ROUTES, SEEDS,
                              alignment_offset, atomic_fractions, build_model, centered,
                              ec_mae, load_element_reference, load_meta, route_prediction,
                              run_dir)

BOOT_SEED, DRAWS = 20260928, 5000
CLASS1 = 'same_budget'
CLASS2 = 'final_best'
CLASS3 = 'matched_accuracy'


def cluster_interval(values, groups, draws=DRAWS, seed=BOOT_SEED):
    frame = pd.DataFrame({'value': values, 'group': groups}).groupby('group').value.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(frame), size=(draws, len(frame)))
    boot = frame['sum'].to_numpy()[pick].sum(axis=1) / frame['count'].to_numpy()[pick].sum(axis=1)
    return float(np.mean(values)), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def load_histories():
    frames = []
    for route in ROUTES:
        for seed in SEEDS:
            path = run_dir(route, seed) / 'history.csv'
            if not path.exists():
                raise FileNotFoundError(f'missing history: {path}')
            frames.append(pd.read_csv(path))
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(['route', 'seed', 'epoch']).reset_index(drop=True)
    history.to_csv(OUT / 'results' / 'training_history.csv', index=False)
    return history


def select_checkpoints(history):
    rows = []
    for route in ROUTES:
        for seed in SEEDS:
            block = history[(history.route == route) & (history.seed == seed)].sort_values('epoch')
            best_final = block.loc[block.val_calibrated_mae.idxmin()]
            for budget in BUDGETS:
                upto = block[block.epoch <= budget]
                pick = upto.loc[upto.val_calibrated_mae.idxmin()]
                rows.append(dict(seed=seed, comparison=CLASS1, criterion=f'budget_{budget}',
                                 budget=budget, route=route, epoch=int(pick.epoch),
                                 optimizer_steps=int(pick.optimizer_steps),
                                 val_native_mae=pick.val_native_mae,
                                 val_calibrated_mae=pick.val_calibrated_mae,
                                 reason=f'best val calibrated MAE among epochs <= {budget}'))
            rows.append(dict(seed=seed, comparison=CLASS2, criterion='final_best', budget=120,
                             route=route, epoch=int(best_final.epoch),
                             optimizer_steps=int(best_final.optimizer_steps),
                             val_native_mae=best_final.val_native_mae,
                             val_calibrated_mae=best_final.val_calibrated_mae,
                             reason='best val calibrated MAE within 120 epochs'))
    # class 3: thresholds reachable by both routes within the same seed
    for seed in SEEDS:
        curves = {route: history[(history.route == route) & (history.seed == seed)].sort_values('epoch')
                  for route in ROUTES}
        candidates = []
        for route in ROUTES:
            block = curves[route]
            candidates.append(float(block[block.epoch == 20].val_calibrated_mae.iloc[0]))
            candidates.append(float(block.val_calibrated_mae.min()))
        for threshold in sorted(set(candidates)):
            reached = {}
            ok = True
            for route in ROUTES:
                block = curves[route]
                hit = block[block.val_calibrated_mae <= threshold]
                if hit.empty:
                    ok = False
                    break
                reached[route] = hit.iloc[0]
            if not ok:
                continue
            gap = abs(float(reached['U'].val_calibrated_mae) - float(reached['A'].val_calibrated_mae))
            for route in ROUTES:
                pick = reached[route]
                rows.append(dict(seed=seed, comparison=CLASS3,
                                 criterion=f'first_within_{threshold:.4f}', budget=None, route=route,
                                 epoch=int(pick.epoch), optimizer_steps=int(pick.optimizer_steps),
                                 val_native_mae=pick.val_native_mae,
                                 val_calibrated_mae=pick.val_calibrated_mae,
                                 threshold_mev=threshold, achieved_gap_mev=gap,
                                 reason=f'first epoch with val calibrated MAE <= {threshold:.4f}'))
    selected = pd.DataFrame(rows)
    selected.to_csv(OUT / 'results' / 'selected_checkpoints.csv', index=False)
    return selected


def confirmation_setup():
    graphs = torch.load(CONFIRMATION / 'sevennet_graphs.pt', map_location='cpu', weights_only=False)
    manifest = pd.read_csv(CONFIRMATION / 'manifest.csv', keep_default_na=False)
    mu, bias = load_element_reference()
    fractions = atomic_fractions(manifest)
    offset = alignment_offset(fractions, mu, bias)
    return graphs, manifest, fractions, offset


def evaluate_checkpoint(route, seed, epoch, model_cache, tensors, device):
    checkpoint = run_dir(route, seed) / f'epoch{epoch:03d}.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = model_cache
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    graphs, manifest, fractions, offset = tensors
    offset_t = torch.tensor(offset, dtype=torch.float32, device=device)
    native = np.full(len(manifest), np.nan)
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=BATCH_EVAL):
            batch = batch.to(device)
            native[batch.manifest_row.cpu().numpy()] = \
                route_prediction(route, model, batch, offset_t).cpu().numpy()
    calibration = pd.read_csv(run_dir(route, seed) / 'calibration_history.csv')
    row = calibration[(calibration.epoch == epoch)].iloc[0]
    weights = row[[f'w_{i}' for i in range(1, 119)]].to_numpy(float)
    calibrated = native + fractions @ weights + float(row['b'])
    return native, calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)

    history = load_histories()
    selected = select_checkpoints(history)
    graphs, manifest, fractions, offset = confirmation_setup()
    truth = manifest.y_true.to_numpy()
    groups = manifest.exact_group.to_numpy()
    overall = manifest.overall_cohort.to_numpy(bool)
    relative = manifest.relative_cohort.to_numpy(bool)

    model = build_model(device)
    tensors = (graphs, manifest, fractions, offset)

    cache = {}
    metric_rows = []
    for _, pick in selected.iterrows():
        key = (pick.route, int(pick.seed), int(pick.epoch))
        if key not in cache:
            cache[key] = evaluate_checkpoint(pick.route, int(pick.seed), int(pick.epoch),
                                             model, tensors, device)
        native, calibrated = cache[key]
        record = dict(seed=int(pick.seed), comparison=pick.comparison, criterion=pick.criterion,
                      route=pick.route, epoch=int(pick.epoch),
                      overall_abs_mae_calibrated=1000 * float(np.abs(calibrated - truth)[overall].mean()),
                      overall_abs_mae_native=1000 * float(np.abs(native - truth)[overall].mean()),
                      relative_abs_mae_calibrated=1000 * float(np.abs(calibrated - truth)[relative].mean()),
                      relative_abs_mae_native=1000 * float(np.abs(native - truth)[relative].mean()))
        ec_cal, _, _ = ec_mae(truth[relative], calibrated[relative], groups[relative])
        ec_nat, _, _ = ec_mae(truth[relative], native[relative], groups[relative])
        record['relative_ec_mae'] = ec_cal
        record['relative_ec_mae_native'] = ec_nat
        record['ec_mae_calibration_shift_mev'] = abs(ec_cal - ec_nat)
        metric_rows.append(record)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / 'results' / 'final_metrics.csv', index=False)

    # paired U-A differences on the confirmation cohorts, matched by (comparison, criterion, seed)
    paired = []
    keys = ['comparison', 'criterion', 'seed']
    for key, block in metrics.groupby(keys):
        if set(block.route) != set(ROUTES):
            continue
        u = block[block.route == 'U'].iloc[0]
        a = block[block.route == 'A'].iloc[0]
        _, cu = cache[('U', int(u.seed), int(u.epoch))]
        _, ca = cache[('A', int(a.seed), int(a.epoch))]
        definitions = (
            ('overall', 'overall_abs_mae_calibrated', overall,
             lambda values: 1000 * np.abs(values - truth)[overall], False),
            ('relative', 'relative_abs_mae_calibrated', relative,
             lambda values: 1000 * np.abs(values - truth)[relative], False),
            ('relative', 'relative_ec_mae', relative,
             lambda values: 1000 * np.abs(centered(values - truth, groups))[relative], True),
        )
        for cohort, metric, mask, scorer, _ in definitions:
            error_u, error_a = scorer(cu), scorer(ca)
            point, low, high = cluster_interval(error_u - error_a, groups[mask])
            paired.append(dict(zip(keys, key),
                               cohort=cohort, metric=metric,
                               mean_u_mev=float(error_u.mean()), mean_a_mev=float(error_a.mean()),
                               difference_u_minus_a_mev=point, ci95_low=low, ci95_high=high))
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / 'results' / 'paired_differences.csv', index=False)

    # are the last 20 epochs still descending?
    slope = []
    for route in ROUTES:
        for seed in SEEDS:
            block = history[(history.route == route) & (history.seed == seed)].sort_values('epoch')
            tail = block.tail(20)
            first = float(tail.val_calibrated_mae.iloc[:5].mean())
            last = float(tail.val_calibrated_mae.iloc[-5:].mean())
            slope.append(dict(route=route, seed=seed, tail20_head_mean=first, tail20_tail_mean=last,
                              change_mev=last - first,
                              still_descending=bool(last < first - 0.5)))
    pd.DataFrame(slope).to_csv(OUT / 'results' / 'tail20_trend.csv', index=False)
    print('selected checkpoints:', len(selected), '| confirmation evaluations:', len(cache))
    print('wrote results/{training_history,selected_checkpoints,final_metrics,paired_differences,tail20_trend}.csv')


if __name__ == '__main__':
    main()
