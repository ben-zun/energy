"""Step 4: prepare the cross-protocol (MP-r2SCAN) adaptation task for SevenNet.

Target protocol : Materials Project r2SCAN formation energy per atom
                  (external_datasets/functional_protocol_round2_large10k)
Source model    : the same frozen 7net-0 checkpoint used everywhere else

Design
  * group = EXACT element-count vector (same definition as the Matbench analysis);
  * groups never span splits;
  * the multi-structure groups are split ~50/50 between train and eval, so the eval
    split carries enough exact-composition groups for the relative-energy and
    selection metrics; singletons are split 70/15/15;
  * the routed reference d(z) must be the element-reference calibration OF THIS
    PROTOCOL, so it is re-fitted on the r2SCAN train split only as a ridge on 118
    atomic fractions + unpenalized intercept (alpha 1e-6) -- the same family used
    for Matbench.  Stored as (mu, bias) with d(z) = -(x @ mu + bias) so the
    experiment14 trainer consumes it unchanged;
  * energy-only, small controlled scale.

Force labels: the released r2SCAN assets here carry none (extxyz holds only
species/positions), so the planned force-based final validation cannot be run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = Path('/home/yxzhang/energy')
EXP12_CODE = ENERGY / 'experiment12_20260929_alignment/code'
DEFAULT_OUT = ENERGY / 'experiment14_alignment_controls_20260929/r2scan10k_full'
DEFAULT_CSV = (ENERGY / 'external_datasets/functional_protocol_round2_large10k'
               / 'meta_gga_r2scan/cleaned/meta_gga_r2scan.csv')
sys.path.insert(0, str(EXP12_CODE))

from pymatgen.core import Composition, Element  # noqa: E402


def _parses(formula):
    try:
        Composition(str(formula))
        return True
    except Exception:                                            # noqa: BLE001
        return False


def exact_group_labels(formulas, natoms):
    labels = []
    for formula, n in zip(formulas, natoms):
        composition = Composition(formula)
        labels.append('|'.join(
            str(int(round(composition.get_atomic_fraction(Element.from_Z(z)) * n)))
            for z in range(1, 119)))
    return np.array(labels)


def assign_splits(groups, seed=20260929):
    """Multi-structure groups are halved between train and eval; singletons 70/15/15."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(groups).value_counts()
    multi = np.array(sizes[sizes >= 2].index)
    single = np.array(sizes[sizes == 1].index)
    rng.shuffle(multi)
    rng.shuffle(single)
    assignment = {}
    for i, group in enumerate(multi):
        assignment[group] = 'train' if i % 2 == 0 else 'test'
    for i, group in enumerate(single):
        position = i / max(len(single), 1)
        assignment[group] = ('train' if position < 0.70
                             else 'val' if position < 0.85 else 'test')
    return np.array([assignment[g] for g in groups])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--csv', default=str(DEFAULT_CSV))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--skip-graphs', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    cache = out / 'graph_cache'
    cache.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.csv)
    frame = frame[['id', 'formula', 'natoms', 'cif_path', 'formation_energy_per_atom_ev']].copy()
    frame = frame.rename(columns={'formation_energy_per_atom_ev': 'y_true'})
    frame = frame.dropna(subset=['formula', 'cif_path', 'y_true'])
    # NOTE: an earlier version filtered the formula string with a strict
    # [A-Za-z0-9]+ regex, which silently dropped 1084 valid entries whose formulas
    # merely contain grouping parentheses (e.g. Sr2(AlAg)5).  That removed a
    # systematically multi-element subset (mean 3.24 vs 2.72 elements).  Use the
    # chemistry parser instead: every one of the 10000 formulas parses.
    parsed = frame.formula.map(lambda f: _parses(f))
    print(f'formula parse: ok={int(parsed.sum())} failed={int((~parsed).sum())}', flush=True)
    frame = frame[parsed].reset_index(drop=True)
    missing = [p for p in frame.cif_path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} CIFs missing, e.g. {missing[:3]}')
    frame['nelements'] = frame.formula.map(lambda f: len(Composition(f).elements))

    groups = exact_group_labels(frame.formula, frame.natoms)
    frame['exact_group'] = groups
    frame['split'] = assign_splits(groups)
    frame = frame.sort_values(['split', 'id']).reset_index(drop=True)
    frame['row_index'] = np.arange(len(frame))

    sizes = pd.Series(frame.exact_group).value_counts()
    multi = set(sizes[sizes >= 2].index)
    report = {'structures': int(len(frame)), 'dataset_csv': args.csv}
    for split in ('train', 'val', 'test'):
        block = frame[frame.split.eq(split)]
        report[split] = dict(structures=int(len(block)),
                             groups=int(block.exact_group.nunique()),
                             multi_groups=int(block[block.exact_group.isin(multi)]
                                              .exact_group.nunique()),
                             multi_structures=int(block.exact_group.isin(multi).sum()))
    report['groups_spanning_splits'] = int(
        (frame.groupby('exact_group').split.nunique() > 1).sum())
    if report['groups_spanning_splits']:
        raise AssertionError('a composition group spans splits')
    if report['test']['multi_groups'] < 40:
        raise AssertionError('eval split has too few exact-composition groups')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    sample = frame.sample(n=min(300, len(frame)), random_state=20260930)
    mismatches = []
    from pymatgen.core import Structure
    for _, row in sample.iterrows():
        structure = Structure.from_file(row.cif_path)
        from_structure = {str(k): int(round(v)) for k, v in structure.composition.as_dict().items()}
        # the DB formula is usually the REDUCED formula; scale it to the cell
        composition = Composition(row.formula)
        from_formula = {str(k): int(round(composition.get_atomic_fraction(k) * int(row.natoms)))
                        for k in composition.elements}
        if from_structure != from_formula or len(structure) != int(row.natoms):
            mismatches.append(row.id)
    report['structure_formula_crosscheck'] = dict(sampled=len(sample),
                                                  mismatches=len(mismatches),
                                                  examples=mismatches[:5])
    print('structure/formula cross-check:', report['structure_formula_crosscheck'], flush=True)

    if args.skip_graphs:
        (out / 'split_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return

    import torch
    from ase.io import read
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.dataload import unlabeled_atoms_to_graph
    from torch_geometric.loader import DataLoader
    from alignment_common import atomic_fractions, build_model, fit_post_calibration, per_atom

    model = build_model(args.device)
    cutoff = float(model.cutoff)
    graphs = {'train': [], 'val': [], 'test': []}
    meta_rows = []
    for _, row in frame.iterrows():
        atoms = read(row.cif_path)
        if len(atoms) != int(row.natoms):
            raise ValueError(f'natoms mismatch for {row.id}')
        graph_dict = unlabeled_atoms_to_graph(atoms, cutoff=cutoff, with_shift=False)
        graph_dict['target_per_atom'] = np.array(float(row.y_true), dtype=np.float32)
        graph_dict['target_total_energy'] = np.array(float(row.y_true) * int(row.natoms),
                                                    dtype=np.float32)
        graph_dict['manifest_row'] = np.array(int(row.row_index), dtype=np.int64)
        graphs[row.split].append(AtomGraphData.from_numpy_dict(graph_dict))
        meta_rows.append(dict(row_index=int(row.row_index), id=str(row.id), split=row.split,
                              formula=row.formula, cif_path=row.cif_path,
                              natoms=int(row.natoms), nelements=int(row.nelements),
                              y_true=float(row.y_true), exact_group=row.exact_group))
    for split in ('train', 'val', 'test'):
        torch.save(graphs[split], cache / f'{split}_graphs.pt')
    meta = pd.DataFrame(meta_rows).sort_values('row_index').reset_index(drop=True)
    meta.to_csv(cache / 'graph_metadata.csv', index=False)
    print('graphs:', {s: len(graphs[s]) for s in graphs}, flush=True)

    fractions = atomic_fractions(meta)
    raw = np.full(len(meta), np.nan)
    model.eval()
    with torch.no_grad():
        for split in ('train', 'val', 'test'):
            for batch in DataLoader(graphs[split], batch_size=32):
                raw[batch.manifest_row.numpy()] = per_atom(model, batch).cpu().numpy()
    if not np.isfinite(raw).all():
        raise AssertionError('nonfinite frozen predictions')
    truth = meta.y_true.to_numpy()
    train_rows = np.flatnonzero(meta.split.to_numpy() == 'train')
    fitted = fit_post_calibration(fractions[train_rows], truth[train_rows] - raw[train_rows])
    w, b = fitted.coef_.astype(float), float(fitted.intercept_)
    np.savez(out / 'r2scan_reference_fit.npz', w=-w, b=-b, train_rows=train_rows,
             raw=raw, truth=truth)
    residual = truth - raw
    report['reference_fit'] = dict(
        train_raw_mae_mev=float(1000 * np.abs(residual[train_rows]).mean()),
        train_residual_std_ev=float(residual[train_rows].std()),
        d_std_ev=float((fractions @ (-w) + (-b)).std()),
        intercept=float(-b))
    for split in ('train', 'val', 'test'):
        rows = np.flatnonzero(meta.split.to_numpy() == split)
        report[f'{split}_raw_mae_mev'] = float(1000 * np.abs(residual[rows]).mean())
    print(json.dumps(report['reference_fit'], indent=2, ensure_ascii=False), flush=True)
    (out / 'split_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
