"""Matched U-vs-A long-budget training.

One script, two routes, identical everything except whether the frozen
element-reference offset d(z) is added to the prediction that enters the loss.

    --alignment none              -> route U:  y_hat = f_theta(X)/N
    --alignment element_reference -> route A:  y_hat = f_theta(X)/N + d(z)

120 epochs by default, no early stopping, best checkpoint by post-calibration
validation MAE (ties -> earlier epoch).  The post-hoc ridge is fit on the train
split only and never participates in backpropagation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from alignment_common import (BATCH_EVAL, BATCH_TRAIN, CALIBRATION_CACHE, CONFIRMATION,
                              EXP11_FROZEN_COMPONENTS, HUBER_BETA, LR, MAX_EPOCHS, OUT,
                              PREFIXES, TRAINABLE_PARAMETERS, WEIGHT_DECAY, alignment_offset,
                              apply_post_calibration, atomic_fractions, build_model, ec_mae,
                              fit_post_calibration, group_labels, load_element_reference,
                              load_graphs, load_meta, per_atom, route_prediction, run_dir,
                              trainable_state_hash)

ROUTE_OF = {'none': 'U', 'element_reference': 'A'}
HISTORY_COLUMNS = ['seed', 'route', 'epoch', 'optimizer_steps',
                   'train_native_mae', 'val_native_mae',
                   'train_calibrated_mae', 'val_calibrated_mae',
                   'val_ec_mae', 'val_native_ec_mae',
                   'training_time_seconds', 'evaluation_time_seconds',
                   'checkpoint_path']


def setup(args):
    route = ROUTE_OF[args.alignment]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta()
    fractions = atomic_fractions(meta)
    mu, bias = load_element_reference()
    offset = alignment_offset(fractions, mu, bias)

    model = build_model(device)
    init_hash = trainable_state_hash(model)

    graphs = load_graphs(('train', 'val', 'test'))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(graphs['train'], batch_size=BATCH_TRAIN, shuffle=True,
                              generator=generator)
    eval_loaders = {split: DataLoader(graphs[split], batch_size=BATCH_EVAL, shuffle=False)
                    for split in ('train', 'val')}
    split_idx = {split: np.flatnonzero(meta.split.to_numpy() == split)
                 for split in ('train', 'val')}
    if len(split_idx['train']) != 8192 or len(split_idx['val']) != 1024:
        raise ValueError('unexpected medium split sizes')
    return dict(route=route, device=device, meta=meta, fractions=fractions, mu=mu, bias=bias,
                offset=offset, model=model, init_hash=init_hash, graphs=graphs,
                train_loader=train_loader, eval_loaders=eval_loaders, split_idx=split_idx)


def forward_split(ctx, route, split, offset_t):
    """Model prediction for one split, in the global manifest_row indexing."""
    values = np.full(len(ctx['meta']), np.nan)
    with torch.no_grad():
        for batch in ctx['eval_loaders'][split]:
            batch = batch.to(ctx['device'])
            values[batch.manifest_row.cpu().numpy()] = \
                route_prediction(route, ctx['model'], batch, offset_t).cpu().numpy()
    return values


def initialization_checks(ctx, args):
    """Initialization invariants demanded by the protocol."""
    model, device = ctx['model'], ctx['device']
    offset_t = torch.tensor(ctx['offset'], dtype=torch.float32, device=device)
    meta, idx = ctx['meta'], ctx['split_idx']
    checks = {}
    route = ctx['route']

    raw = forward_split(ctx, 'U', 'train', offset_t)          # f_theta0 / N
    aligned = forward_split(ctx, 'A', 'train', offset_t)      # f_theta0 / N + d
    train_rows = ctx['split_idx']['train']

    # (1) the two routes differ by exactly d(z), and each routed call matches its formula
    checks['route_A_minus_route_U_equals_d_max_abs'] = float(
        np.abs((aligned - raw - ctx['offset'])[train_rows]).max())
    probe = next(iter(DataLoader(ctx['graphs']['train'][:64], batch_size=32))).to(device)
    probe_rows = probe.manifest_row.cpu().numpy()
    with torch.no_grad():
        formula_u = per_atom(model, probe).cpu().numpy()
        routed_u = route_prediction('U', model, probe, offset_t).cpu().numpy()
        routed_a = route_prediction('A', model, probe, offset_t).cpu().numpy()
    checks['route_U_call_equals_raw_max_abs'] = float(np.abs(routed_u - formula_u).max())
    checks['route_A_call_equals_raw_plus_d_max_abs'] = float(
        np.abs(routed_a - (formula_u + ctx['offset'][probe_rows])).max())

    # (2) cross-check against experiment11's archived base(X) on the medium split
    frozen = pd.read_csv(EXP11_FROZEN_COMPONENTS)
    if not np.array_equal(frozen.id.to_numpy(), meta.id.to_numpy()):
        raise ValueError('experiment11 frozen_components id order differs')
    checks['A_initial_vs_exp11_base_max_abs_ev_atom'] = float(
        np.abs((aligned - frozen.base.to_numpy())[train_rows]).max())

    # (3) cross-check on the frozen confirmation set using its archived graphs
    conf_graphs = torch.load(CONFIRMATION / 'sevennet_graphs.pt', map_location='cpu',
                             weights_only=False)
    conf_manifest = pd.read_csv(CONFIRMATION / 'manifest.csv', keep_default_na=False)
    conf_offset = alignment_offset(atomic_fractions(conf_manifest), ctx['mu'], ctx['bias'])
    conf_raw = np.full(len(conf_manifest), np.nan)
    with torch.no_grad():
        for batch in DataLoader(conf_graphs, batch_size=BATCH_EVAL):
            batch = batch.to(device)
            conf_raw[batch.manifest_row.cpu().numpy()] = per_atom(model, batch).cpu().numpy()
    archived = pd.read_csv(CONFIRMATION / 'sevennet_20260528.csv')
    checks['A_initial_vs_confirmation_base_max_abs_ev_atom'] = float(
        np.abs(conf_raw + conf_offset - archived.base.to_numpy()).max())
    checks['confirmation_d_within_group_max_spread_ev_atom'] = float(
        pd.DataFrame({'d': conf_offset, 'g': conf_manifest.exact_group})
        .groupby('g').d.agg(lambda s: s.max() - s.min()).max())

    # (4) total energy divided by natoms exactly once
    batch = next(iter(DataLoader(ctx['graphs']['train'][:4], batch_size=4))).to(device)
    import sevenn._keys as keys
    with torch.no_grad():
        out = model(batch.clone())
        atomic = per_atom(model, batch).cpu().numpy()
    explicit = (out[keys.PRED_TOTAL_ENERGY].reshape(-1)
                / out[keys.NUM_ATOMS].reshape(-1)).cpu().numpy()
    difference = float(np.abs(atomic - explicit).max())
    if difference > 1e-5:
        raise AssertionError(f'per-atom normalization mismatch: {difference}')
    checks['per_atom_single_division_max_abs_ev_atom'] = difference

    # (5) d(z) is a pure composition function on the medium split
    fractions = ctx['fractions']
    counts = np.rint(fractions * meta.natoms.to_numpy()[:, None]).astype(int)
    groups = np.array(['|'.join(map(str, row)) for row in counts])
    spread = pd.DataFrame({'d': ctx['offset'], 'g': groups}).groupby('g').d.agg(lambda s: s.max() - s.min())
    checks['medium_d_within_group_max_spread_ev_atom'] = float(spread.max())
    checks['d_mean_ev_atom'] = float(ctx['offset'].mean())
    checks['d_std_ev_atom'] = float(ctx['offset'].std())

    # (6) recorded batch order for the U/A comparison, without consuming the loader
    probe = DataLoader(ctx['graphs']['train'], batch_size=BATCH_TRAIN, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    checks['_first_batch_first_rows'] = [int(b.manifest_row[0]) for b in list(probe)[:5]]
    return checks


def post_calibrated_metrics(ctx, native_global):
    """Fit the unified train-only ridge, then score train and val."""
    meta, fractions, idx = ctx['meta'], ctx['fractions'], ctx['split_idx']
    truth = meta.y_true.to_numpy()
    groups = group_labels(meta)
    train_idx, val_idx = idx['train'], idx['val']
    fitted = fit_post_calibration(fractions[train_idx],
                                 truth[train_idx] - native_global[train_idx])
    out = {}
    for split, rows in (('train', train_idx), ('val', val_idx)):
        calibrated = native_global[rows] + apply_post_calibration(fitted, fractions[rows])
        out[f'{split}_native_mae'] = float(1000 * np.abs(native_global[rows] - truth[rows]).mean())
        out[f'{split}_calibrated_mae'] = float(1000 * np.abs(calibrated - truth[rows]).mean())
        if split == 'val':
            out['val_native_ec_mae'] = ec_mae(truth[rows], native_global[rows], groups[rows])[0]
            out['val_ec_mae'] = ec_mae(truth[rows], calibrated, groups[rows])[0]
    return out, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment', choices=sorted(ROUTE_OF), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--smoke', action='store_true', help='one epoch into smoke/')
    parser.add_argument('--verify-init-only', action='store_true')
    args = parser.parse_args()

    ctx = setup(args)
    route = ctx['route']
    checks = initialization_checks(ctx, args)
    first_rows = checks.pop('_first_batch_first_rows')

    if args.verify_init_only:
        print(json.dumps({'route': route, 'seed': args.seed, 'init_hash': ctx['init_hash'],
                          'checks': checks, 'first_batch_first_rows': first_rows}, indent=2))
        return

    run = (OUT / 'smoke' / f'{route}_{args.seed}') if args.smoke else run_dir(route, args.seed)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'status.json').exists() and not args.smoke:
        if json.loads((run / 'status.json').read_text()).get('status') == 'complete':
            raise FileExistsError(f'{run} already complete; refusing to overwrite')

    epochs = 1 if args.smoke else args.epochs
    device = ctx['device']
    offset_t = torch.tensor(ctx['offset'], dtype=torch.float32, device=device)

    (run / 'run_manifest.json').write_text(json.dumps({
        'route': route, 'alignment': args.alignment, 'seed': args.seed,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'prediction': 'f_theta(X)/N' if route == 'U' else 'f_theta(X)/N + d(z)',
        'd_definition': 'd(z) = -(x @ mu_Z + bias), frozen train-only element reference',
        'max_epochs': epochs, 'no_early_stopping': True,
        'selection': 'argmin val calibrated MAE, ties -> earlier epoch',
        'post_calibration': 'ridge on 118 atomic fractions + unpenalized intercept, alpha=1e-6, fit on train only, no backprop',
        'learning_rate': LR, 'weight_decay': WEIGHT_DECAY, 'huber_beta': HUBER_BETA,
        'batch_train': BATCH_TRAIN, 'batch_eval': BATCH_EVAL,
        'trainable_parameters': TRAINABLE_PARAMETERS, 'prefixes': list(PREFIXES),
        'calibration_coefficients': str(CALIBRATION_CACHE),
        'init_hash': ctx['init_hash'], 'init_checks': checks,
        'first_batch_first_rows': first_rows,
    }, indent=2))

    optimizer = torch.optim.AdamW([p for p in ctx['model'].parameters() if p.requires_grad],
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    loss_function = torch.nn.SmoothL1Loss(beta=HUBER_BETA)
    steps_per_epoch = len(ctx['train_loader'])

    history, calibration_rows = [], []
    best = dict(mae=float('inf'), epoch=0)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        ctx['model'].train()
        losses = []
        for batch in ctx['train_loader']:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(route_prediction(route, ctx['model'], batch, offset_t),
                                 batch.target_per_atom.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        ctx['model'].eval()
        native_global = np.full(len(ctx['meta']), np.nan)
        for split in ('train', 'val'):
            native_global[ctx['split_idx'][split]] = forward_split(ctx, route, split, offset_t)[
                ctx['split_idx'][split]]
        metrics, fitted = post_calibrated_metrics(ctx, native_global)
        evaluation_time = time.perf_counter() - start

        checkpoint = run / f'epoch{epoch:03d}.pt'
        state = {name: value.detach().cpu() for name, value in ctx['model'].state_dict().items()}
        torch.save({'model': state, 'route': route, 'seed': args.seed, 'epoch': epoch}, checkpoint)

        history.append({'seed': args.seed, 'route': route, 'epoch': epoch,
                        'optimizer_steps': epoch * steps_per_epoch,
                        **{key: round(value, 6) for key, value in metrics.items()},
                        'training_time_seconds': round(training_time, 3),
                        'evaluation_time_seconds': round(evaluation_time, 3),
                        'checkpoint_path': str(checkpoint)})
        pd.DataFrame(history)[HISTORY_COLUMNS].to_csv(run / 'history.csv', index=False)

        calibration_rows.append({'seed': args.seed, 'route': route, 'epoch': epoch,
                                 **{f'w_{i}': float(w) for i, w in enumerate(fitted.coef_, start=1)},
                                 'b': float(fitted.intercept_)})
        pd.DataFrame(calibration_rows).to_csv(run / 'calibration_history.csv', index=False)

        val_idx = ctx['split_idx']['val']
        np.savez_compressed(run / f'val_epoch{epoch:03d}.npz',
                            manifest_row=val_idx.astype(np.int32),
                            native=native_global[val_idx].astype(np.float32),
                            calibrated=(native_global[val_idx] + apply_post_calibration(
                                fitted, ctx['fractions'][val_idx])).astype(np.float32))

        if metrics['val_calibrated_mae'] < best['mae'] - 1e-12:
            best = dict(mae=metrics['val_calibrated_mae'], epoch=epoch)
            torch.save({'model': state, 'route': route, 'seed': args.seed, 'epoch': epoch},
                       run / 'best.pt')
        print(json.dumps({'route': route, 'seed': args.seed, 'epoch': epoch,
                          'train_loss': round(float(np.mean(losses)), 6),
                          'val_native_mae': round(metrics['val_native_mae'], 4),
                          'val_calibrated_mae': round(metrics['val_calibrated_mae'], 4),
                          'best_epoch': best['epoch']}), flush=True)

    if args.smoke:
        (run / 'status.json').write_text(json.dumps({'status': 'smoke_only', 'epochs': epochs,
                                                     'init_checks': checks}, indent=2))
        return

    (run / 'status.json').write_text(json.dumps({
        'status': 'complete', 'route': route, 'seed': args.seed, 'epochs': epochs,
        'no_early_stopping': True, 'best_epoch': best['epoch'],
        'best_val_calibrated_mae': best['mae'], 'init_hash': ctx['init_hash'],
        'init_checks': checks, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(json.dumps({'route': route, 'seed': args.seed, 'best_epoch': best['epoch'],
                      'best_val_calibrated_mae': best['mae']}))


if __name__ == '__main__':
    main()
