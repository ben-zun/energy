"""P0-1  Unified structure cleaning and evaluation manifest.

One manifest, one cleaning rule, used by every method.  The rule is decided ONLY
from structure identity and calculation metadata -- never from any model's error.

Rule, applied within each exact-composition group (members sorted by id):
  R1  duplicate records        : identical canonical_hash -> keep the first by id
  R2  strict geometric identity: StructureMatcher(ltol=0.05, stol=0.1, angle_tol=1,
                                 primitive_cell=True, scale=True) -> keep the earliest
                                 id, drop the later member(s)
  R3  label-conflict flag      : a collapsed strict pair with |Delta y| > 2 meV/atom
                                 marks the GROUP as label_conflict.  Reported and used
                                 as a sensitivity scope; not removed by default, since a
                                 group with n>2 may still contain usable pairs.
  M1  mixed calculation source : group whose members disagree on `source`
  M2  mixed medium split       : recorded for provenance, not an error
  M3  magnetic state           : NOT in the frozen manifest -> recorded as unavailable

Scopes
  raw                every group with >= 2 members
  dedup              after R1+R2; groups left with < 2 survivors drop out
  dedup_no_conflict  dedup minus groups flagged by R3

The same rule is applied to the r2SCAN eval split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

ENERGY = Path('/home/yxzhang/energy')
CONF = ENERGY / 'experiment11_20260928/confirmation'
R2 = ENERGY / 'experiment14_alignment_controls_20260929/r2scan10k_full/graph_cache'
OUT = ENERGY / 'experiment14_alignment_controls_20260929/unified'
LABEL_CONFLICT_MEV = 2.0
STRICT = dict(ltol=0.05, stol=0.1, angle_tol=1, primitive_cell=True, scale=True,
              attempt_supercell=False)


def clean(frame, group_col, y_col, cif_col, id_col, source_col=None):
    matcher = StructureMatcher(**STRICT)
    keep = pd.Series(True, index=frame.index)
    reasons = pd.Series('', index=frame.index)
    cache, audit_rows = {}, []

    for group, block in frame.groupby(group_col, sort=True):
        block = block.sort_values(id_col)
        n_input = len(block)

        # --- R1 duplicate records -------------------------------------------
        if 'canonical_hash' in block.columns:
            dup = block.duplicated('canonical_hash', keep='first')
            for idx in block.index[dup]:
                keep[idx] = False
                reasons[idx] = 'R1_duplicate_record'
            block = block[~dup]

        # --- R2 strict geometric identity -----------------------------------
        survivors = []                      # list of (index_label, id, structure)
        for idx, row in block.iterrows():
            sid = str(row[id_col])
            if sid not in cache:
                cache[sid] = Structure.from_file(row[cif_col])
            structure = cache[sid]
            partner = next((entry for entry in survivors
                            if matcher.fit(structure, entry[2])), None)
            if partner is None:
                survivors.append((idx, sid, structure))
            else:
                keep[idx] = False
                reasons[idx] = f'R2_strict_equivalent_to_{partner[1]}'

        survivor_ids = [entry[1] for entry in survivors]
        collapsed = [(idx, str(row[id_col])) for idx, row in block.iterrows()
                     if str(row[id_col]) not in survivor_ids]

        # --- R3 label conflict among collapsed strict pairs ------------------
        conflicts = []
        for idx, sid in collapsed:
            partner = reasons[idx].replace('R2_strict_equivalent_to_', '')
            row = frame.loc[idx]
            match = block[block[id_col].astype(str).eq(partner)]
            if len(match):
                dy = 1000 * abs(float(row[y_col]) - float(match.iloc[0][y_col]))
                conflicts.append(dict(a=sid, b=partner, abs_dy_mev=round(dy, 4)))
        max_dy = max([c['abs_dy_mev'] for c in conflicts], default=0.0)

        audit_rows.append(dict(
            group=group, n_input=n_input, n_surviving=len(survivors),
            dropped_ids=json.dumps([entry[1] for entry in collapsed]),
            strict_pairs=len(conflicts),
            max_label_diff_mev=max_dy,
            label_conflict=bool(max_dy > LABEL_CONFLICT_MEV),
            label_conflict_detail=json.dumps(conflicts),
            sources=json.dumps(sorted(set(block[source_col].astype(str)))
                               if source_col else []),
            mixed_source=bool(source_col and block[source_col].nunique() > 1),
            splits=json.dumps(sorted(set(block['split'].astype(str))))
            if 'split' in block.columns else '[]'))

    manifest = frame.copy()
    manifest['keep'] = keep.to_numpy()
    manifest['exclusion_reason'] = reasons.to_numpy()
    return manifest, pd.DataFrame(audit_rows)


def scope_table(audit):
    out = {}
    for name, frame, structures in (
            ('raw', audit[audit.n_input >= 2], 'n_input'),
            ('dedup', audit[audit.n_surviving >= 2], 'n_surviving'),
            ('dedup_no_conflict',
             audit[(audit.n_surviving >= 2) & (~audit.label_conflict)], 'n_surviving')):
        sizes = frame[structures]
        out[name] = dict(groups=int(len(frame)), structures=int(sizes.sum()),
                         pairs=int((sizes * (sizes - 1) // 2).sum()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------------- Matbench frozen confirmation ----------------
    manifest = pd.read_csv(CONF / 'manifest.csv', keep_default_na=False)
    rel = manifest[manifest.relative_cohort.astype(bool)].copy()
    cleaned, audit = clean(rel, 'exact_group', 'y_true', 'cif_path', 'id', 'source')
    cleaned = cleaned.merge(audit[['group', 'label_conflict', 'max_label_diff_mev']],
                            left_on='exact_group', right_on='group', how='left')
    cleaned['cohort'] = 'matbench_confirmation'
    cleaned['magnetic_state'] = 'UNAVAILABLE'
    cleaned.to_csv(out / 'evaluation_manifest.csv', index=False)
    audit.to_csv(out / 'group_cleaning_audit.csv', index=False)
    report['matbench'] = dict(
        input_structures=int(len(rel)),
        R1_duplicate_records=int(cleaned.exclusion_reason.eq('R1_duplicate_record').sum()),
        R2_strict_equivalence=int(cleaned.exclusion_reason.str.startswith('R2_').sum()),
        groups_with_label_conflict=int(audit.label_conflict.sum()),
        groups_with_mixed_source=int(audit.mixed_source.sum()),
        magnetic_state_metadata='unavailable in the frozen manifest',
        scopes=scope_table(audit))
    print('=== Matbench frozen confirmation ===')
    print(json.dumps(report['matbench'], indent=2, ensure_ascii=False))

    # ---------------- r2SCAN eval split ----------------
    r2meta = pd.read_csv(R2 / 'graph_metadata.csv').sort_values('row_index').reset_index(drop=True)
    r2eval = r2meta[r2meta.split.eq('test')].copy()
    # no canonical_hash in this cohort: drop the column so rule R1 is skipped,
    # instead of filling it with a constant (which would collapse every group)
    r2eval = r2eval.drop(columns=[c for c in ('canonical_hash',) if c in r2eval.columns])
    r2eval['source'] = 'mp_r2scan'
    r2clean, r2audit = clean(r2eval, 'exact_group', 'y_true', 'cif_path', 'id', 'source')
    r2clean = r2clean.merge(r2audit[['group', 'label_conflict', 'max_label_diff_mev']],
                            left_on='exact_group', right_on='group', how='left')
    r2clean['cohort'] = 'r2scan_eval'
    r2clean['magnetic_state'] = 'UNAVAILABLE'
    r2clean.to_csv(out / 'evaluation_manifest_r2scan.csv', index=False)
    r2audit.to_csv(out / 'group_cleaning_audit_r2scan.csv', index=False)
    report['r2scan'] = dict(
        input_structures=int(len(r2eval)),
        R1_duplicate_records=int(r2clean.exclusion_reason.eq('R1_duplicate_record').sum()),
        R2_strict_equivalence=int(r2clean.exclusion_reason.str.startswith('R2_').sum()),
        groups_with_label_conflict=int(r2audit.label_conflict.sum()),
        magnetic_state_metadata='unavailable',
        scopes=scope_table(r2audit))
    print('\n=== r2SCAN eval split ===')
    print(json.dumps(report['r2scan'], indent=2, ensure_ascii=False))

    (out / 'cleaning_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nwrote', out)


if __name__ == '__main__':
    main()
