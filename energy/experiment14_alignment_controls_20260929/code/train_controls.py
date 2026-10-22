"""Matched controls for the U-vs-A comparison (guidance step 2).

Three routes, one script, everything else identical:

    U   y_hat = f_theta(X)/N                       (no reference handling)
    A   y_hat = f_theta(X)/N + d(z)                (frozen element reference)
    T   y_hat = f_theta(X)/N + (x @ w + b)         (element reference TRAINABLE,
                                                    initialised exactly at d(z))

T asks whether the network can complete the reference adjustment by itself when
it is given an explicit composition-linear, trainable term.  A and T start from
the same prediction; only whether d can move differs.

--lr overrides the shared learning rate, which is what the small learning-rate
selection budget needs.  Everything else (data, splits, batch order, loss,
optimiser, post-hoc calibration, checkpoint rule) is unchanged from experiment12.

Gradient and update magnitudes are logged every epoch so the "larger residual ->
larger gradient" argument can be checked directly instead of assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ENERGY = Path('/home/yxzhang/energy')
EXP12_CODE = ENERGY / 'experiment12_20260929_alignment/code'
OUT = ENERGY / 'experiment14_alignment_controls_20260929'
sys.path.insert(0, str(EXP12_CODE))

from alignment_common import (BATCH_EVAL, BATCH_TRAIN, HUBER_BETA, LR, MAX_EPOCHS,  # noqa: E402
                              PREFIXES, TRAINABLE_PARAMETERS, WEIGHT_DECAY, alignment_offset,
                              apply_post_calibration, atomic_fractions, build_model, ec_mae,
                              fit_post_calibration, group_labels, load_element_reference,
                              load_graphs, load_meta, per_atom, trainable_state_hash)

ROUTES = ('U', 'A', 'T')
REFERENCE_PARAMETERS = 119
HISTORY_COLUMNS = ['seed', 'route', 'lr', 'epoch', 'optimizer_steps',
                   'train_native_mae', 'val_native_mae',
                   'train_calibrated_mae', 'val_calibrated_mae',
                   'val_ec_mae', 'val_native_ec_mae',
                   'grad_norm_first', 'grad_norm_mean', 'param_delta_norm',
                   'reference_drift_norm',
                   'training_time_seconds', 'evaluation_time_seconds',
                   'checkpoint_path']


class ReferenceTerm(torch.nn.Module):
    """d(z) = x @ w + b, initialised at the frozen element reference."""

    def __init__(self, mu: np.ndarray, bias: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-mu, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(-float(bias), dtype=torch.float32))

    def forward(self, fractions):
        return fractions @ self.weight + self.bias

    @torch.no_grad()
    def drift_norm(self, mu: np.ndarray, bias: float) -> float:
        """||(w, b) - (w_init, b_init)||, i.e. how far the reference has moved."""
        target_w = torch.as_tensor(-mu, dtype=torch.float32, device=self.weight.device)
        target_b = torch.as_tensor(-float(bias), dtype=torch.float32, device=self.bias.device)
        return float(torch.sqrt(((self.weight - target_w) ** 2).sum()
                                + (self.bias - target_b) ** 2))


def route_prediction_ext(route, model, reference, batch, offset_tensor, fractions_tensor):
    energy = per_atom(model, batch)
    rows = batch.manifest_row.long()
    if route == 'U':
        return energy
    if route == 'A':
        return energy + offset_tensor[rows]
    if route == 'T':
        return energy + reference(fractions_tensor[rows])
    raise ValueError(f'unknown route {route!r}')


def setup(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta()
    fractions = atomic_fractions(meta)
    mu, bias = load_element_reference()
    offset = alignment_offset(fractions, mu, bias)

    model = build_model(device)
    reference = ReferenceTerm(mu, bias).to(device)
    init_hash = trainable_state_hash(model)

    graphs = load_graphs(('train', 'val', 'test'))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(graphs['train'], batch_size=BATCH_TRAIN, shuffle=True,
                              generator=generator)
    split_idx = {split: np.flatnonzero(meta.split.to_numpy() == split)
                 for split in ('train', 'val')}
    expect_train = int(getattr(args, 'expect_train', 8192))
    expect_val = int(getattr(args, 'expect_val', 1024))
    if len(split_idx['train']) != expect_train or len(split_idx['val']) != expect_val:
        raise ValueError(f'unexpected split sizes {len(split_idx["train"])}/'
                         f'{len(split_idx["val"])}, expected {expect_train}/{expect_val}')
    return dict(device=device, meta=meta, fractions=fractions, mu=mu, bias=bias, offset=offset,
                model=model, reference=reference, init_hash=init_hash, graphs=graphs,
                train_loader=train_loader, split_idx=split_idx)


def init_checks(ctx, args):
    device = ctx['device']
    fractions_t = torch.tensor(ctx['fractions'], dtype=torch.float32, device=device)
    offset_t = torch.tensor(ctx['offset'], dtype=torch.float32, device=device)
    checks = {}
    batch = next(iter(DataLoader(ctx['graphs']['train'][:16], batch_size=16))).to(device)
    with torch.no_grad():
        u = route_prediction_ext('U', ctx['model'], ctx['reference'], batch, offset_t, fractions_t)
        a = route_prediction_ext('A', ctx['model'], ctx['reference'], batch, offset_t, fractions_t)
        t = route_prediction_ext('T', ctx['model'], ctx['reference'], batch, offset_t, fractions_t)
    checks['A_minus_U_equals_d_max_abs'] = float((a - u - offset_t[batch.manifest_row.long()]).abs().max())
    checks['T_initial_equals_A_max_abs'] = float((t - a).abs().max())
    checks['trainable_parameters_main'] = int(sum(p.numel() for p in ctx['model'].parameters()
                                                  if p.requires_grad))
    checks['reference_parameters'] = int(sum(p.numel() for p in ctx['reference'].parameters()))
    if checks['trainable_parameters_main'] != TRAINABLE_PARAMETERS:
        raise AssertionError('trainable parameter count changed')
    if checks['T_initial_equals_A_max_abs'] > 1e-4:
        raise AssertionError('T does not start at the frozen reference')
    return checks


def forward_split(ctx, route, split):
    device = ctx['device']
    fractions_t = ctx['fractions_t']
    offset_t = ctx['offset_t']
    rows = ctx['split_idx'][split]
    out = np.full(len(ctx['meta']), np.nan)
    dataset = ctx['graphs'][split]
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_EVAL):
            batch = batch.to(device)
            prediction = route_prediction_ext(route, ctx['model'], ctx['reference'], batch,
                                              offset_t, fractions_t)
            out[batch.manifest_row.cpu().numpy()] = prediction.cpu().numpy()
    return out[rows]


def post_calibrated_metrics(ctx, native_global):
    import alignment_common as A
    meta, fractions, idx = ctx['meta'], ctx['fractions'], ctx['split_idx']
    truth = meta.y_true.to_numpy()
    groups = group_labels(meta)
    train_idx, val_idx = idx['train'], idx['val']
    fitted = A.fit_post_calibration(fractions[train_idx],
                                    truth[train_idx] - native_global[train_idx])
    out = {}
    for split, rows in (('train', train_idx), ('val', val_idx)):
        calibrated = native_global[rows] + A.apply_post_calibration(fitted, fractions[rows])
        out[f'{split}_native_mae'] = float(1000 * np.abs(native_global[rows] - truth[rows]).mean())
        out[f'{split}_calibrated_mae'] = float(1000 * np.abs(calibrated - truth[rows]).mean())
        if split == 'val':
            out['val_native_ec_mae'] = ec_mae(truth[rows], native_global[rows], groups[rows])[0]
            out['val_ec_mae'] = ec_mae(truth[rows], calibrated, groups[rows])[0]
    return out, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--tag', default='controls')
    parser.add_argument('--out-root', default=None,
                        help='override the run root (default experiment14/runs)')
    parser.add_argument('--expect-train', type=int, default=8192)
    parser.add_argument('--expect-val', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify-init-only', action='store_true')
    args = parser.parse_args()

    ctx = setup(args)
    route = args.route
    checks = init_checks(ctx, args)
    if args.verify_init_only:
        print(json.dumps({'route': route, 'seed': args.seed, 'checks': checks}, indent=2))
        return

    device = ctx['device']
    ctx['fractions_t'] = torch.tensor(ctx['fractions'], dtype=torch.float32, device=device)
    ctx['offset_t'] = torch.tensor(ctx['offset'], dtype=torch.float32, device=device)
    offset_t, fractions_t = ctx['offset_t'], ctx['fractions_t']

    out_root = Path(args.out_root) if args.out_root else OUT
    run = out_root / ('smoke' if args.smoke else 'runs') / f'{args.tag}_{route}_{args.seed}_lr{args.lr:g}'
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'status.json').exists() and not args.smoke:
        if json.loads((run / 'status.json').read_text()).get('status') == 'complete':
            raise FileExistsError(f'{run} already complete')

    epochs = 1 if args.smoke else args.epochs
    groups = [{'params': [p for p in ctx['model'].parameters() if p.requires_grad],
               'lr': args.lr, 'weight_decay': WEIGHT_DECAY}]
    if route == 'T':
        groups.append({'params': list(ctx['reference'].parameters()), 'lr': args.lr,
                       'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(groups)
    loss_function = torch.nn.SmoothL1Loss(beta=HUBER_BETA)
    steps_per_epoch = len(ctx['train_loader'])

    (run / 'run_manifest.json').write_text(json.dumps({
        'route': route, 'seed': args.seed, 'learning_rate': args.lr, 'tag': args.tag,
        'prediction': {'U': 'f_theta(X)/N', 'A': 'f_theta(X)/N + d(z) frozen',
                       'T': 'f_theta(X)/N + (x@w + b) trainable, init at d(z)'}[route],
        'reference_initialisation': 'w = -mu_Z, b = -bias (identical to the frozen d)',
        'reference_weight_decay': 0.0 if route == 'T' else None,
        'max_epochs': epochs, 'no_early_stopping': True,
        'selection': 'argmin val calibrated MAE, ties -> earlier epoch',
        'post_calibration': 'ridge on 118 atomic fractions + unpenalized intercept, alpha=1e-6, train only',
        'weight_decay': WEIGHT_DECAY, 'huber_beta': HUBER_BETA,
        'batch_train': BATCH_TRAIN, 'batch_eval': BATCH_EVAL,
        'trainable_parameters': TRAINABLE_PARAMETERS, 'prefixes': list(PREFIXES),
        'init_hash': ctx['init_hash'], 'init_checks': checks,
        'started_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2))

    history, calibration_rows = [], []
    best = dict(mae=float('inf'), epoch=0)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        ctx['model'].train()
        ctx['reference'].train()
        losses, grad_norms, grad_first = [], [], None
        theta_before = torch.cat([p.detach().reshape(-1) for p in ctx['model'].parameters()
                                  if p.requires_grad]).clone()
        for batch in ctx['train_loader']:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                route_prediction_ext(route, ctx['model'], ctx['reference'], batch, offset_t,
                                     fractions_t),
                batch.target_per_atom.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum()
                                             for p in ctx['model'].parameters()
                                             if p.requires_grad and p.grad is not None)))
            if grad_first is None:
                grad_first = grad_norm
            grad_norms.append(grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        training_time = time.perf_counter() - start
        theta_after = torch.cat([p.detach().reshape(-1) for p in ctx['model'].parameters()
                                 if p.requires_grad])
        param_delta = float((theta_after - theta_before).norm())

        start = time.perf_counter()
        ctx['model'].eval()
        native_global = np.full(len(ctx['meta']), np.nan)
        for split in ('train', 'val'):
            native_global[ctx['split_idx'][split]] = forward_split(ctx, route, split)
        metrics, fitted = post_calibrated_metrics(ctx, native_global)
        evaluation_time = time.perf_counter() - start

        checkpoint = run / f'epoch{epoch:03d}.pt'
        payload = {'model': {k: v.detach().cpu() for k, v in ctx['model'].state_dict().items()},
                   'route': route, 'seed': args.seed, 'epoch': epoch, 'lr': args.lr}
        if route == 'T':
            payload['reference'] = {k: v.detach().cpu() for k, v in ctx['reference'].state_dict().items()}
        torch.save(payload, checkpoint)

        history.append({'seed': args.seed, 'route': route, 'lr': args.lr, 'epoch': epoch,
                        'optimizer_steps': epoch * steps_per_epoch,
                        **{key: round(value, 6) for key, value in metrics.items()},
                        'grad_norm_first': round(grad_first, 8),
                        'grad_norm_mean': round(float(np.mean(grad_norms)), 8),
                        'param_delta_norm': round(param_delta, 8),
                        'reference_drift_norm': (round(ctx['reference'].drift_norm(ctx['mu'],
                                                                                  ctx['bias']), 8)
                                                 if route == 'T' else 0.0),
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
            torch.save(payload, run / 'best.pt')
        print(json.dumps({'route': route, 'seed': args.seed, 'lr': args.lr, 'epoch': epoch,
                          'train_loss': round(float(np.mean(losses)), 6),
                          'val_native_mae': round(metrics['val_native_mae'], 4),
                          'val_calibrated_mae': round(metrics['val_calibrated_mae'], 4),
                          'grad_first': round(grad_first, 4),
                          'param_delta': round(param_delta, 6),
                          'best_epoch': best['epoch']}), flush=True)

    (run / 'status.json').write_text(json.dumps({
        'status': 'smoke_only' if args.smoke else 'complete', 'route': route, 'seed': args.seed,
        'lr': args.lr, 'epochs': epochs, 'no_early_stopping': True, 'best_epoch': best['epoch'],
        'best_val_calibrated_mae': best['mae'], 'init_hash': ctx['init_hash'],
        'init_checks': checks, 'finished_at_utc': datetime.now(timezone.utc).isoformat()},
        indent=2))
    print(json.dumps({'route': route, 'seed': args.seed, 'lr': args.lr, 'best_epoch': best['epoch'],
                      'best_val_calibrated_mae': best['mae']}))


if __name__ == '__main__':
    main()
