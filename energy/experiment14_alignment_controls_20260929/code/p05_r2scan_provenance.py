"""Audit item: verify label-structure correspondence for the r2SCAN cohort.

Three things are checked, all from the actual files rather than from names:
  1. provenance   : which dataset / record id each label came from, and whether the
                    label file and the CIF set refer to the same materials;
  2. counts       : the database formula is usually the REDUCED formula, so it must
                    be scaled by natoms before comparison with the cell contents;
  3. label/structure pairing: the CIF read from disk must have exactly the
                    composition and atom count that the label row claims.

The earlier 172/300 'mismatch' was a bug in this check (unscaled formula), not a
data defect; this version scales by natoms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

ENERGY = Path('/home/yxzhang/energy')
CSV = (ENERGY / 'external_datasets/functional_protocol_round2_large10k'
       / 'meta_gga_r2scan/cleaned/meta_gga_r2scan.csv')
MANIFEST = (ENERGY / 'external_datasets/functional_protocol_round2_large10k'
            / 'meta_gga_r2scan/raw/download_manifest.json')
SPLIT_REPORT = ENERGY / 'experiment14_alignment_controls_20260929/r2scan10k_full/split_report.json'
OUT = ENERGY / 'experiment14_alignment_controls_20260929/unified/r2scan_provenance.json'


def main():
    frame = pd.read_csv(CSV)
    report = {}
    report['label_source'] = dict(
        dataset_csv=str(CSV),
        source_dataset=frame.source_dataset.unique().tolist(),
        source_thermo_type=frame.source_thermo_type.unique().tolist(),
        id_prefix_examples=frame.id.head(3).tolist(),
        id_equals_material_id=bool((frame.id.astype(str)
                                    == frame.material_id.astype(str)).all()),
        source_record_index_equals_id=bool((frame.source_record_index.astype(str)
                                            == frame.id.astype(str)).all()),
        n_rows=int(len(frame)),
        unique_ids=int(frame.id.nunique()),
        unique_material_ids=int(frame.material_id.nunique()))
    if MANIFEST.exists():
        report['label_source']['download_manifest'] = json.loads(MANIFEST.read_text())

    # ---- counts from the structure, scaled formula comparison ----------------
    sample = frame.sample(n=min(400, len(frame)), random_state=20260930)
    bad_counts, bad_atoms, missing = [], [], []
    for _, row in sample.iterrows():
        if not Path(row.cif_path).exists():
            missing.append(row.id)
            continue
        structure = Structure.from_file(row.cif_path)
        cell = {str(k): int(round(v)) for k, v in structure.composition.as_dict().items()}
        composition = Composition(str(row.formula))
        n = int(row.natoms)
        scaled = {str(k): int(round(composition.get_atomic_fraction(k) * n))
                  for k in composition.elements}
        if cell != scaled:
            bad_counts.append(dict(id=row.id, formula=row.formula, natoms=n,
                                   from_structure=cell, scaled_formula=scaled))
        if len(structure) != n:
            bad_atoms.append(dict(id=row.id, cif_atoms=len(structure), claimed=n))
    report['pairing_check'] = dict(
        sampled=int(len(sample)),
        missing_cif=int(len(missing)),
        composition_mismatch=int(len(bad_counts)),
        atomcount_mismatch=int(len(bad_atoms)),
        examples_composition=bad_counts[:5],
        examples_atomcount=bad_atoms[:5])

    # ---- duplicate structures across the cohort -----------------------------
    hashes = {}
    dup = 0
    for _, row in frame.sample(n=min(1200, len(frame)), random_state=1).iterrows():
        structure = Structure.from_file(row.cif_path)
        key = structure.composition.reduced_formula + '|' + str(len(structure))
        hashes[key] = hashes.get(key, 0) + 1
    report['structure_identity'] = dict(sampled=1200,
                                       same_reduced_formula_and_size=sum(
                                           v - 1 for v in hashes.values() if v > 1))

    # ---- split provenance ---------------------------------------------------
    if SPLIT_REPORT.exists():
        split = json.loads(SPLIT_REPORT.read_text())
        report['split'] = {k: split[k] for k in ('structures', 'train', 'val', 'test',
                                                 'groups_spanning_splits')}
        report['split']['crosscheck_note'] = (
            'the crosscheck recorded in split_report.json was produced by the earlier '
            'unscaled comparison; this file supersedes it')

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
