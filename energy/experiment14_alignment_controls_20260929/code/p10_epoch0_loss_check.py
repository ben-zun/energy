"""Step 1 -- EqV3 starting-point check.  NO TRAINING; forward passes only.

Q1  before any parameter update, what are the direct and clean errors on TRAIN and
    VAL, and what is the actual training LOSS?
    The loss comes from the trainer's own run_epoch called with optimizer=None --
    the identical code path used for validation -- so it is the real
    SmoothL1Loss(beta=huber_beta) on the real batches, not a re-derived formula.

Q2  was the reference fitted from the CURRENT initialisation network's own
    predictions, including output head, normalisation and units?
    (a) the offset was fitted as d(z) ~ y - e0 on the TRAIN split, with e0 the
        frozen per-atom output recorded in the calibration archive;
    (b) LoRA's B matrix is zero-initialised, so at init the live network IS the
        frozen network -- asserted numerically, not assumed;
    (c) a live forward pass is compared against the archived e0;
    (d) unit bookkeeping is asserted: reference_total = ref_per_atom * natoms once.

Any failure raises instead of printing a number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
SCRIPTS = ENERGY / 'rclfa_v3' / 'scripts'
UNIFIED = EXP14 / 'unified'
CKPT = ENERGY / 'checkpoints/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt'
DATASET = ENERGY / 'runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269'
CALIB = (ENERGY / 'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration'
         / 'calibrated_predictions.csv')
ROUTES = (('direct', 'direct', None, None),
          ('clean', 'residual', EXP14 / 'eqv3_fitted_reference.csv',
           'pred_element_offset_fitted'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--max-train', type=int, default=1024)
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS))
    import torch
    from rclfa_v3 import eqv2_native, eqv3_native
    import run_eqv3_native_upper_bound_smoke_20260725 as trainer

    device = args.device
    torch.cuda.set_device(device)
    report = {}

    # ---------- Q2(a): how the offset was fitted ----------------------------
    frame = pd.read_csv(CALIB)
    coeffs = np.load(EXP14 / 'eqv3_fitted_offset_coefficients.npz')
    w, b = coeffs['w'], float(coeffs['b'])
    from pymatgen.core import Composition, Element
    def frac(formula):
        c = Composition(str(formula))
        return np.array([c.get_atomic_fraction(Element.from_Z(z)) for z in range(1, 119)])
    delta = np.vstack([frac(f) for f in frame.formula]) @ w + b
    y = frame.y_true.to_numpy(float)
    e0 = frame.model_energy_per_atom.to_numpy(float)
    tr = frame.split.to_numpy() == 'train'
    report['reference_fit'] = dict(
        target='y - e0, with e0 = model_energy_per_atom (frozen network output)',
        split=f'train only ({int(tr.sum())} structures)',
        features='118 atomic fractions + intercept, Ridge(alpha=1e-6)',
        fitted_on='the same frozen checkpoint every route is initialised from',
        train_bias_after_offset_mev=1000 * float((y - e0 - delta)[tr].mean()),
        val_bias_after_offset_mev=1000 * float((y - e0 - delta)[~tr].mean()))

    # ---------- live model, attached exactly as the runs did ----------------
    calc = eqv2_native.OCPCalculator(checkpoint_path=CKPT, cpu=False, seed=20260528,
                                     only_output=['energy'])
    calc.trainer.scaler = None
    for normalizer in calc.trainer.normalizers.values():
        normalizer.to(device)
    for references in calc.trainer.elementrefs.values():
        references.to(device)
    model = getattr(calc.trainer, '_unwrapped_model', None) or calc.trainer.model
    model = model.to(device)
    model.regress_forces = False
    model.regress_stress = False
    if hasattr(model, 'gradient_checkpointing_block_list'):
        model.gradient_checkpointing_block_list = [0] * len(model.blocks)
    scale_divisor = eqv3_native.configure_eqv3_energy_scaling(model)
    eqv3_native.configure_eqv3_tuning_mode(model, tuning_type='lora', lora_rank=8,
                                           lora_alpha=16.0)
    stats = eqv3_native.summarize_parameter_stats(model)
    lora_b_max = max((float(p.detach().abs().max())
                      for n, p in model.named_parameters() if n.endswith('lora_B')),
                     default=float('nan'))
    report['initialisation'] = dict(checkpoint=str(CKPT),
                                    scale_divisor=float(scale_divisor),
                                    trainable_parameters=stats['trainable_parameters'],
                                    max_abs_lora_B=lora_b_max,
                                    lora_delta_is_exactly_zero=bool(lora_b_max == 0.0))
    if lora_b_max != 0.0:
        raise AssertionError('LoRA is not zero-initialised; the init is not the frozen model')

    amp_dtype, _ = trainer.resolve_amp_dtype(device, 'bfloat16')
    criterion = torch.nn.SmoothL1Loss(beta=0.05)

    # ---------- Q1 + Q2(b)(c)(d) -------------------------------------------
    rows, checks, unit_rows = [], [], []
    for route, target_mode, ref_csv, ref_col in ROUTES:
        records = eqv2_native.build_structure_records(
            dataset_root=DATASET, workspace_root=ENERGY, a2g=calc.a2g,
            reference_csv=ref_csv, reference_column=ref_col or 'pred_element_offset_fitted',
            max_train=args.max_train, max_val=None, max_test=0)
        residual_mode = target_mode == 'residual'

        for split in ('train', 'val'):
            metrics, _ = trainer.run_epoch(
                model, records[split], batch_size=args.batch,
                scale_divisor=scale_divisor, residual_mode=residual_mode,
                device=device, criterion=criterion, optimizer=None, scaler=None,
                clip_grad_norm=0.0, seed=0, amp_dtype=amp_dtype, grad_accum_steps=1,
                stop_requested=[False])
            rows.append(dict(route=route, split=split, structures=len(records[split]),
                             loss=float(metrics['loss']),
                             mae_mev_atom=float(metrics['mae_mev_atom'])))

        # Q2(d): the trainer multiplies the per-atom reference by natoms exactly once
        sample = records['val'][:16]
        bundle = eqv2_native.collate_graph_records(sample, scale_divisor=scale_divisor,
                                                   residual_mode=residual_mode)
        unit_rows.append(dict(
            route=route,
            reference_is_per_atom=bool(np.allclose(
                bundle.reference_total_ev,
                bundle.reference_per_atom_ev * bundle.natoms, atol=1e-4)),
            target_equals_label_minus_reference=bool(np.allclose(
                bundle.target_total_ev,
                bundle.y_true_total_ev - bundle.reference_total_ev
                if residual_mode else bundle.y_true_total_ev, atol=1e-4))))

        # Q2(b)(c): the live network must reproduce the archived e0
        with torch.no_grad():
            out = model(bundle.batch.to(device))['energy'].reshape(-1).float().cpu().numpy()
        live_per_atom = out * scale_divisor / bundle.natoms
        archived = pd.read_csv(CALIB).set_index('id').loc[bundle.ids].model_energy_per_atom
        checks.append(dict(route=route, sampled=len(bundle.ids),
                           max_abs_dev_vs_archived_e0_ev=float(
                               np.abs(live_per_atom - archived.to_numpy(float)).max())))

    epoch0, live, units = pd.DataFrame(rows), pd.DataFrame(checks), pd.DataFrame(unit_rows)
    worst = float(live.max_abs_dev_vs_archived_e0_ev.max())
    if worst > 1e-3:
        raise AssertionError(f'live init does not reproduce archived e0 (max dev {worst:.3e} '
                             f'eV/atom): the reference was NOT fitted on this initialisation')
    if not bool(units.reference_is_per_atom.all()):
        raise AssertionError('reference unit bookkeeping is wrong')
    report['epoch0'] = epoch0.to_dict(orient='records')
    report['live_vs_archived_e0'] = live.to_dict(orient='records')
    report['unit_checks'] = units.to_dict(orient='records')
    pivot = epoch0.pivot(index='split', columns='route', values='mae_mev_atom')
    report['verdict'] = dict(
        aligned_improves_train=bool(pivot.loc['train', 'clean'] < pivot.loc['train', 'direct']),
        aligned_improves_val=bool(pivot.loc['val', 'clean'] < pivot.loc['val', 'direct']),
        note='if the aligned route is better on BOTH train and val at step 0, then any '
             'later disadvantage is caused by what training does, not by a broken reference')

    epoch0.to_csv(UNIFIED / 'epoch0_loss.csv', index=False)
    (UNIFIED / 'epoch0_loss_check.json').write_text(json.dumps(report, indent=2,
                                                               ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nwrote', UNIFIED / 'epoch0_loss_check.json')


if __name__ == '__main__':
    main()
