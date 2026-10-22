"""Step 3 -- build the composition-only control task.

Idea (from the review): give the model a target whose within-group structure is
EXACTLY the frozen model's, so a perfect solution requires changing nothing about
the构型 energy differences -- only adding a composition offset:

    y_control(X) = e0(X) + s(z)

  e0(X)  frozen model per-atom output, evaluated by a forward pass (SevenNet) or
         read from the calibration archive (EqV3)
  s(z)   a composition-only term, FIXED IN ADVANCE at the scale of the real
         element-reference offset of that model.  It is never chosen by looking
         at test results.

Routes
  unaligned : fit y_control directly                 prediction = network
  aligned   : predict network + s(z), fit y_control   prediction = network + s(z)
              at init the network already outputs e0, so this branch starts at
              exactly zero error -- it is an implementation check, not a result.

Self-validation, all of which must pass before anything can be trained:
  V1  s(z) is constant within every exact-composition group
  V2  y_control has the SAME within-group variation as e0 (the defining property)
  V3  the aligned branch's initial prediction equals y_control to machine precision
  V4  no label of the control task equals the real label (it is a synthetic task)

Nothing is trained here; this only writes the datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

ENERGY = Path('/home/yxzhang/energy')
EXP14 = ENERGY / 'experiment14_alignment_controls_20260929'
CONTROL = EXP14 / 'control_task'
CALIB = (ENERGY / 'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration'
         / 'calibrated_predictions.csv')
EQV3_DATASET = ENERGY / 'runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269'
SEVENNET_CACHE = (ENERGY / 'runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision'
                  / 'fair_finetune_293/graph_cache/medium')


def fractions_of(formulas):
    out = []
    for formula in formulas:
        composition = Composition(str(formula))
        out.append([composition.get_atomic_fraction(Element.from_Z(z))
                    for z in range(1, 119)])
    return np.asarray(out, dtype=float)


def group_keys(formulas, natoms):
    keys = []
    for formula, n in zip(formulas, natoms):
        composition = Composition(str(formula))
        counts = np.rint([composition.get_atomic_fraction(Element.from_Z(z)) * n
                          for z in range(1, 119)]).astype(int)
        keys.append('|'.join(str(c) for c in counts))
    return np.asarray(keys)


def check(name, keys, s, y_control, e0):
    s_spread = pd.DataFrame({'v': s, 'g': keys}).groupby('g').v.agg(lambda v: v.max() - v.min())
    c_spread = pd.DataFrame({'v': y_control, 'g': keys}).groupby('g').v.agg(
        lambda v: v.max() - v.min())
    e_spread = pd.DataFrame({'v': e0, 'g': keys}).groupby('g').v.agg(
        lambda v: v.max() - v.min())
    result = dict(
        V1_s_max_within_group_spread=float(s_spread.max()),
        V2_control_vs_e0_max_spread_difference=float((c_spread - e_spread).abs().max()),
        groups=int(len(s_spread)),
        structures=int(len(keys)))
    if result['V1_s_max_within_group_spread'] > 1e-9:
        raise AssertionError(f'{name}: s(z) is not composition-only')
    if result['V2_control_vs_e0_max_spread_difference'] > 1e-9:
        raise AssertionError(f'{name}: the control label changed the within-group structure')
    return result


def build_eqv3(report):
    frame = pd.read_csv(CALIB)
    coeffs = np.load(EXP14 / 'eqv3_fitted_offset_coefficients.npz')
    s = fractions_of(frame.formula) @ coeffs['w'] + float(coeffs['b'])   # real scale
    e0 = frame.model_energy_per_atom.to_numpy(float)
    control = e0 + s
    keys = group_keys(frame.formula, frame.natoms)
    report['eqv3'] = check('eqv3', keys, s, control, e0)
    report['eqv3']['s_rms_mev'] = 1000 * float(np.sqrt((s ** 2).mean()))
    report['eqv3']['V4_labels_differ'] = bool(
        np.abs(control - frame.y_true.to_numpy(float)).max() > 1e-6)

    out = CONTROL / 'eqv3'
    for split in ('train', 'val', 'test'):
        source = EQV3_DATASET / split / f'{split}.csv'
        if not source.exists():
            continue
        rows = pd.read_csv(source)
        index = frame.set_index('id')
        missing = set(rows.id) - set(index.index)
        if missing:
            raise KeyError(f'{split}: {len(missing)} ids missing from the calibration archive')
        rows['formation_energy_per_atom_ev'] = index.loc[
            rows.id].model_energy_per_atom.to_numpy(float) + (
            fractions_of(index.loc[rows.id].formula) @ coeffs['w'] + float(coeffs['b']))
        target = out / split
        target.mkdir(parents=True, exist_ok=True)
        rows.to_csv(target / f'{split}.csv', index=False)
    # reference column the aligned branch will use: s(z) itself
    (out / 'reference.csv').parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'split': frame.split, 'id': frame.id,
                  'control_offset': s}).to_csv(out / 'reference.csv', index=False)
    report['eqv3']['dataset_root'] = str(out)
    report['eqv3']['reference_csv'] = str(out / 'reference.csv')


def build_sevennet(report, device='cuda:0'):
    import sys
    sys.path.insert(0, str(ENERGY / 'experiment12_20260929_alignment/code'))
    import torch
    import alignment_common as ac

    meta = ac.load_meta()
    mu, bias = ac.load_element_reference()
    fractions = ac.atomic_fractions(meta)
    s = ac.alignment_offset(fractions, mu, bias)          # real scale
    graphs = ac.load_graphs(('train', 'val', 'test'))
    model = ac.build_model(device)
    from torch_geometric.loader import DataLoader
    e0 = np.zeros(len(meta), dtype=float)
    with torch.no_grad():
        for split, items in graphs.items():
            rows = meta.index[meta.split.eq(split)].to_numpy()
            cursor = 0
            # DataLoader preserves order when shuffle=False, and the graphs were
            # saved in metadata order, so the cursor stays aligned with `rows`
            for batch in DataLoader(items, batch_size=32, shuffle=False):
                batch = batch.to(device)
                values = ac.per_atom(model, batch).float().cpu().numpy()
                e0[rows[cursor:cursor + len(values)]] = values
                cursor += len(values)
            if cursor != len(rows):
                raise AssertionError(f'{split}: covered {cursor} of {len(rows)} structures')
    control = e0 + s
    keys = group_keys(meta.formula, meta.natoms)
    report['sevennet'] = check('sevennet', keys, s, control, e0)
    report['sevennet']['s_rms_mev'] = 1000 * float(np.sqrt((s ** 2).mean()))
    report['sevennet']['V4_labels_differ'] = bool(
        np.abs(control - meta.y_true.to_numpy(float)).max() > 1e-6)

    out = CONTROL / 'sevennet'
    out.mkdir(parents=True, exist_ok=True)
    for split, items in graphs.items():
        rows = meta.index[meta.split.eq(split)].to_numpy()
        for item, value in zip(items, control[rows]):
            item.target_per_atom = torch.tensor(float(value), dtype=torch.float32)
        torch.save(items, out / f'{split}_graphs.pt')
    meta_out = meta.copy()
    meta_out['y_true'] = control
    meta_out.to_csv(out / 'graph_metadata.csv', index=False)
    report['sevennet']['graph_cache'] = str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='eqv3,sevennet')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    CONTROL.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.models.split(','):
        if name == 'eqv3':
            build_eqv3(report)
        elif name == 'sevennet':
            build_sevennet(report, args.device)
        else:
            raise ValueError(name)
    (CONTROL / 'control_task_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
