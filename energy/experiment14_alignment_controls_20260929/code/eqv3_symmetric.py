"""Step 3 evaluation: symmetric EqV3 comparison on the frozen confirmation set.

Routes, matched inside EqV3 (same dataset, same LoRA configuration, same
trainable parameters, same 12-epoch budget, same checkpoint rule):

  residual (aligned)   the repo's existing runs
                       run/runs/rclfa_v3/eqv3_lora_fegx_multiseed_20260904/seed*/native/
  direct   (unaligned) trained in experiment14 with --target-mode direct

Both get the SAME unified post-hoc calibration: a ridge on 118 atomic fractions
plus an unpenalized intercept (alpha 1e-6) fitted on the medium TRAIN split for
that route and seed, then applied unchanged to the confirmation set.

Reported: absolute MAE, EC-MAE, pair MAE, argmin hit rate, selection cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ENERGY = Path('/home/yxzhang/energy')
EXP11 = ENERGY / 'experiment11_20260928'
EXP12_CODE = ENERGY / 'experiment12_20260929_alignment/code'
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
CONF = EXP11 / 'confirmation'
MEDIUM_META = (ENERGY / 'runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision'
               / 'fair_finetune_293/graph_cache/medium/graph_metadata.csv')
RESIDUAL_ROOT = ENERGY / 'runs/rclfa_v3/eqv3_lora_fegx_multiseed_20260904'
sys.path.insert(0, str(EXP11 / 'code'))
sys.path.insert(0, str(EXP12_CODE))

from confirmation_eqv3 import (composition_features, ensure_eqv3_imports,  # noqa: E402
                               reference_offset)
from alignment_common import atomic_fractions  # noqa: E402

SEEDS = (20260528, 20260529, 20260530)
TIERS = [('residual', 'aligned'), ('direct', 'unaligned')]
TIE_TAUS = (0.0, 1.0, 2.0, 5.0, 10.0)
MATERIAL_TAU = 10.0
DRAWS, BOOT_SEED = 5000, 20260928


def checkpoint_path(route, seed):
    if route == 'residual':
        return RESIDUAL_ROOT / f'seed{seed}' / 'native' / f'residual_lora_seed{seed}_best.pt'
    return EXP14 / 'eqv3_direct' / f'seed{seed}' / f'direct_lora_seed{seed}_best.pt'


def predictions_path(route, seed):
    if route == 'residual':
        return (RESIDUAL_ROOT / f'seed{seed}' / 'native'
                / f'residual_lora_seed{seed}_predictions.csv')
    return EXP14 / 'eqv3_direct' / f'seed{seed}' / f'direct_lora_seed{seed}_predictions.csv'


def build_model(device):
    ensure_eqv3_imports()
    from fairchem.core import OCPCalculator
    calculator = OCPCalculator(
        checkpoint_path=ENERGY / 'checkpoints/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt',
        cpu=False, seed=20260928, only_output=['energy'])
    calculator.trainer.scaler = None
    for normalizer in calculator.trainer.normalizers.values():
        normalizer.to(device)
    for references in calculator.trainer.elementrefs.values():
        references.to(device)
    model = getattr(calculator.trainer, '_unwrapped_model', None) or calculator.trainer.model
    model = model.to(device).eval()
    model.regress_forces = False
    model.regress_stress = False
    if hasattr(model, 'gradient_checkpointing_block_list'):
        model.gradient_checkpointing_block_list = [0] * len(model.blocks)
    return calculator, model


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
                            **{f'acceptable_{t:g}': int(cost <= t) for t in TIE_TAUS}))
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
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    device = args.device
    torch.cuda.set_device(device)

    manifest = pd.read_csv(CONF / 'manifest.csv', keep_default_na=False)
    base = pd.read_csv(CONF / 'eqv3_base.csv').set_index('id').loc[manifest.id].reset_index()
    truth = manifest.y_true.to_numpy(float)
    groups = manifest.exact_group.to_numpy()
    overall = manifest.overall_cohort.to_numpy(bool)
    relative = manifest.relative_cohort.to_numpy(bool)
    conf_fractions = atomic_fractions(manifest)
    reference = base.native_reference.to_numpy(float)
    natoms = manifest.natoms.to_numpy(float)

    # fractions on the medium train split, for the unified post-hoc ridge
    medium = pd.read_csv(MEDIUM_META)
    medium_fractions = atomic_fractions(medium)

    calculator, model = build_model(device)
    from fairchem.core.datasets import data_list_collater
    from ase.io import read

    graphs = []
    for row in manifest.itertuples(index=False):
        atoms = read(row.cif_path)
        if len(atoms) != row.natoms:
            raise ValueError(f'atom count mismatch for {row.id}')
        graphs.append(calculator.a2g.convert(atoms))
    print(f'EqV3 confirmation graphs: {len(graphs)}', flush=True)

    def batch_for(items):
        return data_list_collater([item.clone() for item in items], otf_graph=True).to(device)

    lora_state = {'configured': False}

    def run_checkpoint(route, seed):
        path = checkpoint_path(route, seed)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        config = payload['config']
        from rclfa_v3.eqv3_native import configure_eqv3_tuning_mode
        if not lora_state['configured']:
            # attach LoRA once; every route/seed here uses the same rank and alpha
            configure_eqv3_tuning_mode(model, tuning_type='lora',
                                       lora_rank=int(config['lora_rank']),
                                       lora_alpha=float(config['lora_alpha']))
            lora_state['configured'] = True
        model.load_state_dict(payload['model'], strict=True)
        scale = float(payload.get('scale_divisor', 1.0))
        model.avg_num_nodes = scale
        model.eval()
        model.regress_forces = False
        model.regress_stress = False
        model.gradient_checkpointing_block_list = [0] * len(model.blocks)
        out = []
        with torch.no_grad():
            for start in range(0, len(graphs), args.batch):
                batch = batch_for(graphs[start:start + args.batch])
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    energy = model(batch)['energy'].reshape(-1)
                out.extend((energy.float() * scale).cpu().tolist())
        model.gradient_checkpointing_block_list = [0] * len(model.blocks)
        return np.asarray(out, float)

    rows, frames, provenance, preds_store = [], {}, [], {}
    for route, label in TIERS:
        for seed in SEEDS:
            if not checkpoint_path(route, seed).exists():
                print(f'MISSING {route} seed {seed}', flush=True)
                continue
            delta_total = run_checkpoint(route, seed)
            if route == 'direct':
                native = delta_total / natoms
            else:
                native = reference + delta_total / natoms

            # unified post-hoc: ridge fitted on the medium train split of this run
            preds = pd.read_csv(predictions_path(route, seed))
            train_rows = preds[preds.split.eq('train')].set_index('id')
            fractions_by_id = pd.DataFrame(medium_fractions, index=medium.id.to_numpy())
            missing = train_rows.index.difference(fractions_by_id.index)
            if len(missing):
                raise KeyError(f'{len(missing)} train ids missing from the medium metadata')
            ridge_fractions = fractions_by_id.loc[train_rows.index].to_numpy(float)
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1e-6, fit_intercept=True)
            ridge.fit(ridge_fractions,
                      train_rows.y_true_per_atom_ev.to_numpy(float)
                      - train_rows.y_pred_per_atom_ev.to_numpy(float))
            calibrated = native + ridge.predict(conf_fractions)

            preds_store[(route, seed)] = calibrated
            frame = selection_metrics(truth, calibrated, groups, relative)
            frames[(route, seed)] = frame
            boot = bootstrap(frame)
            entry = dict(method=f'{route} ({label})', route=route, seed=seed,
                         overall_abs_mae=1000 * float(np.abs(calibrated[overall] - truth[overall]).mean()),
                         relative_abs_mae=1000 * float(np.abs(calibrated[relative] - truth[relative]).mean()),
                         ec_mae=float(frame.ec_abs_sum.sum() / frame.n.sum()),
                         pair_mae=float(frame.pair_err_sum.sum() / frame.pair_count.sum()),
                         exact_hit_rate=float(frame.exact_hit.mean()),
                         acceptable_1_rate=float(frame['acceptable_1'].mean()),
                         cost_mean=float(frame.cost_mev.mean()),
                         cost_p90=float(np.quantile(frame.cost_mev, .90)),
                         wrong_pick_rate_gt_10mev=float((frame.cost_mev > MATERIAL_TAU).mean()))
            for key in ('ec_mae', 'pair_mae', 'exact_hit', 'acceptable_1', 'cost_mean',
                        'cost_p90', 'wrong10'):
                entry[f'{key}_lo'], entry[f'{key}_hi'] = ci(boot[key])
            rows.append(entry)
            provenance.append(dict(route=route, seed=seed, checkpoint=str(checkpoint_path(route, seed)),
                                   ridge_intercept=float(ridge.intercept_)))
            print(f'{route} seed={seed}: overall={entry["overall_abs_mae"]:.2f} '
                  f'ec={entry["ec_mae"]:.3f} hit={entry["exact_hit_rate"]:.3f}', flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(EXP14 / 'results_eqv3_summary.csv', index=False)

    long_rows = []
    for (route, seed), frame in preds_store.items():
        long_rows.append(pd.DataFrame({'id': manifest.id, 'method': route, 'seed': seed,
                                       'prediction': frame}))
    (EXP14 / 'unified').mkdir(parents=True, exist_ok=True)
    pd.concat(long_rows, ignore_index=True).to_csv(
        EXP14 / 'unified' / 'predictions_eqv3.csv', index=False)
    pd.DataFrame(provenance).to_csv(EXP14 / 'results_eqv3_provenance.csv', index=False)
    print('\n=== EqV3 symmetric: frozen confirmation ===')
    print(summary[['method', 'seed', 'overall_abs_mae', 'relative_abs_mae', 'ec_mae',
                   'exact_hit_rate', 'acceptable_1_rate', 'cost_p90',
                   'wrong_pick_rate_gt_10mev']].round(3).to_string(index=False))

    paired = []
    for seed in SEEDS:
        if ('direct', seed) not in frames or ('residual', seed) not in frames:
            continue
        fd = frames[('direct', seed)].sort_values('group').reset_index(drop=True)
        fr = frames[('residual', seed)].sort_values('group').reset_index(drop=True)
        if not (fd.group.to_numpy() == fr.group.to_numpy()).all():
            raise ValueError('group sets differ')
        rng = np.random.default_rng(BOOT_SEED)
        pick = rng.integers(len(fd), size=(DRAWS, len(fd)))
        cu, cr = fd.cost_mev.to_numpy()[pick], fr.cost_mev.to_numpy()[pick]
        defs = {
            'ec_mae_mev': (float(fd.ec_abs_sum.sum() / fd.n.sum() - fr.ec_abs_sum.sum() / fr.n.sum()),
                           fd.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fd.n.to_numpy()[pick].sum(axis=1)
                           - fr.ec_abs_sum.to_numpy()[pick].sum(axis=1) / fr.n.to_numpy()[pick].sum(axis=1)),
            'exact_hit_rate': (float(fd.exact_hit.mean() - fr.exact_hit.mean()),
                               fd.exact_hit.to_numpy()[pick].mean(axis=1)
                               - fr.exact_hit.to_numpy()[pick].mean(axis=1)),
            'acceptable_1_rate': (float(fd['acceptable_1'].mean() - fr['acceptable_1'].mean()),
                                  fd['acceptable_1'].to_numpy()[pick].mean(axis=1)
                                  - fr['acceptable_1'].to_numpy()[pick].mean(axis=1)),
            'cost_mean_mev': (float(fd.cost_mev.mean() - fr.cost_mev.mean()),
                              cu.mean(axis=1) - cr.mean(axis=1)),
            'cost_p90_mev': (float(np.quantile(fd.cost_mev, .90) - np.quantile(fr.cost_mev, .90)),
                             np.quantile(cu, .90, axis=1) - np.quantile(cr, .90, axis=1)),
            'wrong_pick_rate_gt_10mev': (float((fd.cost_mev > MATERIAL_TAU).mean()
                                               - (fr.cost_mev > MATERIAL_TAU).mean()),
                                         (cu > MATERIAL_TAU).mean(axis=1)
                                         - (cr > MATERIAL_TAU).mean(axis=1)),
        }
        for metric, (point, dist) in defs.items():
            lo, hi = ci(dist)
            paired.append(dict(seed=seed, metric=metric, difference_unaligned_minus_aligned=point,
                               ci95_low=lo, ci95_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    paired = pd.DataFrame(paired)
    paired.to_csv(EXP14 / 'results_eqv3_paired.csv', index=False)
    print('\n=== direct (unaligned) - residual (aligned) ===')
    print(paired.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
