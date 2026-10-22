"""Hard acceptance checks for the U-vs-A long-budget comparison."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from alignment_common import (CALIBRATION_ALPHA, MAX_EPOCHS, OUT, ROUTES, SEEDS,
                              atomic_fractions, load_element_reference, load_meta,
                              run_dir)

# A and U predictions are produced by two independent forward passes, so their
# difference carries float32 nondeterminism of order 1e-6 eV on ~5 eV outputs.
# Any real defect in d(z) (sign, double-add, wrong row indexing) would show up at
# the ~1 eV level, so 1e-4 eV separates noise from bugs with a wide margin.
IDENTITY_TOLERANCE_EV = 1e-4

HISTORY_COLUMNS = ['seed', 'route', 'epoch', 'optimizer_steps',
                   'train_native_mae', 'val_native_mae',
                   'train_calibrated_mae', 'val_calibrated_mae',
                   'val_ec_mae', 'val_native_ec_mae',
                   'training_time_seconds', 'evaluation_time_seconds',
                   'checkpoint_path']


def main():
    problems, notes = [], {}
    meta = load_meta()
    fractions = atomic_fractions(meta)
    mu, bias = load_element_reference()
    truth = meta.y_true.to_numpy()
    val_rows = np.flatnonzero(meta.split.to_numpy() == 'val')

    init_hashes, batch_orders, histories = {}, {}, {}
    for route in ROUTES:
        for seed in SEEDS:
            run = run_dir(route, seed)
            status = json.loads((run / 'status.json').read_text())
            if status.get('status') != 'complete':
                problems.append(f'{route}_{seed}: status {status.get("status")}')
            history = pd.read_csv(run / 'history.csv')
            histories[(route, seed)] = history
            if list(history.columns) != HISTORY_COLUMNS:
                problems.append(f'{route}_{seed}: history columns differ')
            if len(history) != MAX_EPOCHS:
                problems.append(f'{route}_{seed}: {len(history)} epochs, expected {MAX_EPOCHS}')
            if not np.isfinite(history.select_dtypes(include=[np.number])).all().all():
                problems.append(f'{route}_{seed}: non-finite history values')
            if len(list(run.glob('epoch*.pt'))) != MAX_EPOCHS:
                problems.append(f'{route}_{seed}: missing per-epoch checkpoints')
            if len(list(run.glob('val_epoch*.npz'))) != MAX_EPOCHS:
                problems.append(f'{route}_{seed}: missing per-epoch val predictions')
            # selection rule: best.pt must be the argmin of val_calibrated_mae, ties -> earlier
            expected = int(history.loc[history.val_calibrated_mae.idxmin(), 'epoch'])
            if status['best_epoch'] != expected:
                problems.append(f'{route}_{seed}: best_epoch {status["best_epoch"]} != argmin {expected}')
            # post-hoc calibration is linear in composition -> EC-MAE unchanged by construction
            shift = float((history.val_ec_mae - history.val_native_ec_mae).abs().max())
            notes.setdefault('ec_shift', {})[f'{route}_{seed}'] = shift
            if shift > 1e-6:
                problems.append(f'{route}_{seed}: EC-MAE changed by calibration ({shift})')
            # saved per-epoch calibration reproduces the saved calibrated val predictions
            calibration = pd.read_csv(run / 'calibration_history.csv')
            weights = calibration[[f'w_{i}' for i in range(1, 119)]].to_numpy(float)
            intercept = calibration['b'].to_numpy(float)
            check_epochs = (1, MAX_EPOCHS // 2, MAX_EPOCHS)
            for epoch in check_epochs:
                saved = np.load(run / f'val_epoch{epoch:03d}.npz')
                reconstructed = saved['native'] + fractions[val_rows] @ weights[epoch - 1] + intercept[epoch - 1]
                deviation = float(np.abs(reconstructed - saved['calibrated']).max())
                if deviation > 1e-4:
                    problems.append(f'{route}_{seed}: calibration reconstruction off at epoch {epoch} ({deviation})')
                notes.setdefault('calibration_reconstruction', {})[f'{route}_{seed}_e{epoch}'] = deviation
            init_hashes[(route, seed)] = status['init_hash']
            batch_orders[(route, seed)] = status['init_checks']

    # U and A must start from identical parameters and identical batch order
    for seed in SEEDS:
        if init_hashes[('U', seed)] != init_hashes[('A', seed)]:
            problems.append(f'seed {seed}: initialization hash differs between U and A')
        checks = json.loads((run_dir('A', seed) / 'status.json').read_text())['init_checks']
        if checks['route_A_minus_route_U_equals_d_max_abs'] > IDENTITY_TOLERANCE_EV:
            problems.append(f'seed {seed}: A - U != d(z) '
                            f'({checks["route_A_minus_route_U_equals_d_max_abs"]:.3e} eV)')
        if checks['A_initial_vs_exp11_base_max_abs_ev_atom'] > 1e-4:
            problems.append(f'seed {seed}: A initial prediction != archived base')
    for seed in SEEDS:
        rows_u = json.loads((run_dir('U', seed) / 'run_manifest.json').read_text())['first_batch_first_rows']
        rows_a = json.loads((run_dir('A', seed) / 'run_manifest.json').read_text())['first_batch_first_rows']
        if rows_u != rows_a:
            problems.append(f'seed {seed}: first-batch order differs between U and A')

    for name in ('PROTOCOL.md', 'FINAL_COMPARISON.md'):
        if not (OUT / name).exists():
            problems.append(f'missing deliverable {name}')
    for name in ('training_history.csv', 'selected_checkpoints.csv'):
        if not (OUT / 'results' / name).exists():
            problems.append(f'missing deliverable results/{name}')

    print(json.dumps({'hard_checks': 'passed' if not problems else 'failed',
                      'problems': problems}, indent=2))
    (OUT / 'FINAL_VERIFICATION.json').write_text(json.dumps(
        {'hard_checks': 'passed' if not problems else 'failed', 'problems': problems,
         'notes': notes}, indent=2))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
