"""Audit item: does the r2SCAN label belong to the structure we actually use?

The download helper took the ENERGY from mpr.materials.thermo (thermo_type r2SCAN)
but the STRUCTURE from mpr.materials.summary.  MP's r2SCAN workflow relaxes the
structure at the r2SCAN level, so those can be different geometries -- in which case
the label is not the energy of the input structure.

This script settles it from the API: for each sampled material it compares the CIF we
train on against
    (a) the r2SCAN ComputedStructureEntry structure  (the geometry the label belongs to)
    (b) the summary structure                        (what the helper downloaded)
using pymatgen's StructureMatcher, and reports the volume and site-count agreement.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

warnings.filterwarnings('ignore')

ENERGY = Path('/home/yxzhang/energy')
CSV = (ENERGY / 'external_datasets/functional_protocol_round2_large10k'
       / 'meta_gga_r2scan/cleaned/meta_gga_r2scan.csv')
OUT = ENERGY / 'experiment14_alignment_controls_20260929/unified/r2scan_label_structure.json'
N = 12


def main():
    from mp_api.client import MPRester

    frame = pd.read_csv(CSV).sample(n=N, random_state=20260930)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    rows = []
    with MPRester() as mpr:
        for _, row in frame.iterrows():
            mid = row.material_id
            cif = Structure.from_file(row.cif_path)
            thermo = mpr.materials.thermo.search(
                material_ids=[mid],
                fields=['material_id', 'thermo_type', 'formation_energy_per_atom',
                        'entries'])
            r2 = None
            for doc in thermo:
                if str(doc.thermo_type).casefold() == 'r2scan':
                    entry = doc.entries.get('r2SCAN') if isinstance(doc.entries, dict) else None
                    if entry is not None:
                        ident = getattr(entry, 'entry_id', None)
                    if isinstance(ident, dict):
                        ident = ident.get('identifier', str(ident))
                    if ident is None:
                        ident = getattr(entry, 'identifier', None) or getattr(
                            entry, 'calc_id', None) or 'unknown'
                    r2 = dict(identifier=str(ident),
                                  structure=entry.structure,
                                  formation=float(doc.formation_energy_per_atom))
            summary = mpr.materials.summary.search(
                material_ids=[mid], fields=['material_id', 'structure'])
            sm = summary[0].structure if summary else None

            def cmp(target):
                if target is None or len(target) != len(cif):
                    return None
                try:
                    return bool(matcher.fit(cif, target))
                except Exception:                                  # noqa: BLE001
                    return None

            rows.append(dict(
                material_id=mid, natoms=int(row.natoms),
                cif_formula=cif.composition.reduced_formula,
                r2scan_identifier=r2['identifier'] if r2 else None,
                r2scan_natoms=len(r2['structure']) if r2 else None,
                summary_natoms=len(sm) if sm is not None else None,
                cif_matches_r2scan_structure=cmp(r2['structure'] if r2 else None),
                cif_matches_summary_structure=cmp(sm),
                r2scan_volume=round(float(r2['structure'].volume), 4) if r2 else None,
                summary_volume=round(float(sm.volume), 4) if sm is not None else None,
                cif_volume=round(float(cif.volume), 4)))
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    flag_r2 = table.cif_matches_r2scan_structure.fillna(False).astype(bool)
    flag_sm = table.cif_matches_summary_structure.fillna(False).astype(bool)
    match_r2 = int(flag_r2.sum())
    match_sm = int(flag_sm.sum())
    comparable = int(table.cif_matches_r2scan_structure.notna().sum())
    # volume agreement between the two candidate structures
    vol = (table.r2scan_volume - table.cif_volume).abs() / table.cif_volume
    report = dict(
        sampled=int(len(table)),
        comparable=int(comparable),
        cif_matches_r2scan_structure=match_r2,
        cif_matches_summary_structure=match_sm,
        same_structure_both=int((flag_r2 & flag_sm).sum()),
        identical_volumes=int((table.r2scan_volume == table.cif_volume).sum()),
        mean_abs_relative_volume_difference_vs_r2scan=float(vol.mean()),
        max_abs_relative_volume_difference_vs_r2scan=float(vol.max()),
        verdict=('the CIF matches the r2SCAN entry structure (for these materials the '
                 'summary structure IS the r2SCAN structure, so the download helper was '
                 'correct)' if match_r2 == len(table) and match_sm == len(table) else
                 'the CIF matches the r2SCAN entry structure'
                 if match_r2 > match_sm else
                 'the CIF matches the summary (GGA) structure, NOT the r2SCAN one'
                 if match_sm > match_r2 else 'inconclusive on this sample'),
        detail=table.to_dict(orient='records'))
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('\n=== verdict ===')
    print(json.dumps({k: v for k, v in report.items() if k != 'detail'},
                     indent=2, ensure_ascii=False))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
