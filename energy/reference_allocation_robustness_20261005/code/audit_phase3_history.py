"""ID-only history audit. No prediction values are read and no models are run.

Run locally for tests first. The export mode reads the original server gzip and
only ID columns in historical CSVs, writing to a new audit directory.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn '
            'Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag '
            'Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm '
            'Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa '
            'U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn '
            'Nh Fl Mc Lv Ts Og').split()
ID_PATTERN = re.compile(r'(?:mb-mp-e-form-|matbench-mp-e-form-)(\d+)\Z')

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()

def canonical_id(value):
    """Current official fold uses one-based mb IDs; external raw uses zero-based.

    This mapping must pass full metadata label/count and sampled geometry checks
    in export(); string normalization alone is never treated as version proof.
    """
    match = ID_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f'Unrecognized Matbench ID: {value!r}')
    number=int(match.group(1))
    if str(value).strip().startswith('matbench-mp-e-form-'):
        number += 1
    elif number == 0:
        raise ValueError('Current official-fold mb namespace is one-based')
    return f'mb-mp-e-form-{number:06d}'

def counts_from_structure(structure):
    counts = [0]*118
    for site in structure['sites']:
        species = site['species']
        if len(species) != 1 or abs(float(species[0].get('occu', 1))-1) > 1e-12:
            raise ValueError('Disordered/partially occupied site; requires separate policy')
        counts[ELEMENTS.index(species[0]['element'])] += 1
    if not sum(counts):
        raise ValueError('Empty structure')
    return counts

def read_id_columns(path, columns):
    rows = []
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if any(col not in reader.fieldnames for col in columns):
            raise ValueError(f'Missing ID column in {path}')
        for row in reader:
            values = [canonical_id(row[col]) for col in columns if row[col]]
            if not values or len(set(values)) != 1:
                raise ValueError(f'Absent/inconsistent ID aliases at record {len(rows)}')
            rows.append(values[0])
    return rows

def coverage(source_ids, history_ids):
    source, history = set(source_ids), set(history_ids)
    if len(source) != len(source_ids):
        raise ValueError('Duplicate original dataset IDs')
    return dict(source_n=len(source), history_unique_n=len(history),
                intersection_n=len(source & history),
                remaining_n=len(source-history),
                unknown_history_n=len(history-source))

def export(args):
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a phase3 audit')
    paths = [args.source, args.history, args.registry, *args.metadata]
    hashes = {str(p):sha(p) for p in paths}
    print('Loading original Matbench JSON; no model predictions are inspected', flush=True)
    with gzip.open(args.source, 'rt') as stream:
        raw = json.load(stream)
    columns = raw['columns']
    if columns != ['structure', 'e_form']:
        raise ValueError(f'Unexpected original source columns: {columns}')
    indices, data = raw['index'], raw['data']
    if len(indices) != len(data) or len(set(indices)) != len(indices):
        raise ValueError('Ambiguous source indices')
    source = {}
    sample_ids=set(sorted((canonical_id(f'matbench-mp-e-form-{idx}') for idx in indices),
                         key=lambda sid:(hashlib.sha256(sid.encode()).hexdigest(),sid))[:32])
    sample_structures={}
    for idx, row in zip(indices, data):
        if not isinstance(idx, int) or idx < 0:
            raise ValueError('Source index is not a nonnegative integer')
        counts = counts_from_structure(row[0]); truth = float(row[1])
        if not math.isfinite(truth):
            raise ValueError('Nonfinite source truth')
        sid = canonical_id(f'matbench-mp-e-form-{idx}')
        source[sid] = dict(id=sid, source_index=idx, natoms=sum(counts),
                           composition_key='|'.join(map(str, counts)),
                           truth_ev_atom=truth)
        if sid in sample_ids:sample_structures[sid]=row[0]
    del raw, data
    # Explicitly only consume ID strings from the archived prediction table.
    historical = read_id_columns(args.history, ['id_label','safe_id','id_pred'])
    covered = coverage(list(source), historical)
    registry_ids = []
    registry_max_error = 0.0
    with args.registry.open() as stream:
        for line in stream:
            row = json.loads(line); sid = canonical_id(row['id'])
            registry_ids.append(sid)
            if sid not in source or int(row['natoms']) != source[sid]['natoms']:
                raise ValueError('Historical registry/source atom count mismatch')
            registry_max_error = max(registry_max_error,
                abs(float(row['formation_energy_per_atom_ev'])-source[sid]['truth_ev_atom']))
    if registry_max_error > 1e-10:
        raise ValueError('Historical registry/source target version mismatch')
    metadata_checks = []
    sample_paths={}
    for path in args.metadata:
        count = 0; maximum = 0.0
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                sid = canonical_id(row['id'])
                target_column='y_true' if 'y_true' in row else 'formation_energy_per_atom_ev'
                if sid not in source or int(row['natoms']) != source[sid]['natoms']:
                    matches=[v['id'] for v in source.values()
                             if int(row['natoms']) == v['natoms'] and
                             abs(float(row[target_column])-v['truth_ev_atom']) < 1e-12]
                    raise ValueError(f'Current development/source atom mismatch: {sid}; '
                                     f'current={row["natoms"]},{row[target_column]}; '
                                     f'source={source.get(sid)}; '
                                     f'count-and-target candidate aliases={matches}')
                maximum = max(maximum, abs(float(row[target_column])-source[sid]['truth_ev_atom']))
                if sid in sample_ids:sample_paths[sid]=Path(row['cif_path'])
                count += 1
        if maximum > 1e-10:
            raise ValueError('Current development/source target version mismatch')
        metadata_checks.append(dict(path=str(path), records=count,
                                   max_abs_target_difference_ev_atom=maximum,
                                   atom_counts_match=True))
    if len(set(registry_ids)) != len(registry_ids):
        raise ValueError('Registry duplicate IDs')
    registry_coverage = coverage(list(source), registry_ids)
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
    matcher=StructureMatcher(ltol=0.2,stol=0.3,angle_tol=5,
        primitive_cell=False,scale=False,attempt_supercell=False,comparator=ElementComparator())
    geometry_checks=[]
    if set(sample_paths) != sample_ids:
        raise ValueError('Need full official-fold metadata to verify 32 geometry samples')
    for sid in sorted(sample_ids):
        path=sample_paths[sid]; hashes[str(path)]=sha(path)
        original=Structure.from_dict(sample_structures[sid])
        current=Structure.from_file(path)
        ok=bool(matcher.fit(original,current))
        if not ok:raise ValueError(f'Source-row geometry mapping mismatch: {sid}')
        geometry_checks.append(dict(id=sid,source_index=source[sid]['source_index'],
                                   cif_path=str(path),structure_match=True))
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Read-only input changed during audit')
    args.output.mkdir(parents=True)
    with (args.output/'source_metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(next(iter(source.values()))))
        writer.writeheader();writer.writerows(source.values())
    with (args.output/'historical_prediction_ids.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['id'])
        writer.writerows([sid] for sid in sorted(set(historical)))
    audit=dict(status='PASS', original_source_columns=columns,
               prediction_history=str(args.history), prediction_rows=len(historical),
               coverage=covered, registry_coverage=registry_coverage,
               registry_max_target_difference_ev_atom=registry_max_error,
               current_development_source_checks=metadata_checks,
               source_geometry_mapping_checks=geometry_checks,
               id_normalization='External zero-based row i maps to official-fold mb ID i+1; verified against metadata',
               input_hashes=hashes, script_sha256=sha(Path(__file__)),
               prediction_values_inspected=False, model_forwards=0,
               geometry_duplicate_audit='not_run_if_no_remaining_IDs')
    (args.output/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)

def main():
    parser=argparse.ArgumentParser()
    for key in ('source','history','registry','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--metadata',type=Path,action='append',default=[])
    export(parser.parse_args())

if __name__=='__main__':main()
