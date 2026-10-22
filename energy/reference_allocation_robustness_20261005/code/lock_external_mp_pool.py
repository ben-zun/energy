"""Audit downloaded pool and lock groups without running any model predictions."""
import argparse
import csv
import gzip
import hashlib
import json
from math import gcd
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from download_external_mp import structure_counts,after_cutoff,canonical_id

MATCHER={'ltol':0.2,'stol':0.3,'angle_tol':5,'primitive_cell':False,
         'scale':False,'attempt_supercell':False}

def key(counts):return '|'.join(map(str,counts))
def reduced(counts):
    divisor=0
    for n in counts:divisor=gcd(divisor,n)
    if not divisor:raise ValueError('Empty composition')
    return tuple(n//divisor for n in counts)

def choose_groups(groups):
    eligible=[k for k,ids in groups.items() if len(ids)>=2]
    return sorted(eligible,key=lambda k:(hashlib.sha256(k.encode()).hexdigest(),k))[:100]

def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--old-matbench',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--additional-history',type=Path)
    parser.add_argument('--train-metadata',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('New lock directory required')
    protocol=json.loads((args.download/'EXTERNAL_MP_PROTOCOL_v1.json').read_text())
    acquisition=json.loads((args.download/'DOWNLOAD_AUDIT.json').read_text())
    if acquisition['history_scan_unreadable_n']:
        raise ValueError('Unresolved unreadable history; cannot certify cohort')
    if acquisition['database_version']!=protocol['mp_database_version']:
        raise ValueError('Unexpected MP database version')
    hashes={str(p):sha(p) for p in [args.old_matbench,args.download/'candidate_records.jsonl.gz',
        args.download/'historical_mp_ids.csv',args.download/'EXTERNAL_MP_PROTOCOL_v1.json']}
    train_ids=set();train_keys=set()
    if args.train_metadata:
        hashes[str(args.train_metadata)]=sha(args.train_metadata)
        with args.train_metadata.open() as stream:
            train_ids={row['id'] for row in csv.DictReader(stream) if row['split']=='train'}
        if len(train_ids)!=8192:raise ValueError('Unexpected original train ID count')
    with (args.download/'historical_mp_ids.csv').open() as stream:
        historical={row['canonical_material_id'] for row in csv.DictReader(stream)}
    local_history=set()
    if args.additional_history:
        hashes[str(args.additional_history)]=sha(args.additional_history)
        with args.additional_history.open() as stream:
            local_history={row['canonical_material_id'] for row in csv.DictReader(stream)}
    historical |= local_history
    records={};groups={}
    local_excluded=[]
    with gzip.open(args.download/'candidate_records.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);sid=canonical_id(row['material_id'])
            if sid in local_history:
                local_excluded.append(sid);continue
            if sid in records or sid in historical:raise ValueError('Duplicate/historical candidate ID')
            if row['thermo_type']!='GGA_GGA+U' or not after_cutoff(row['created_at'],protocol['cutoff_utc']):
                raise ValueError('Candidate violates functional/time rule')
            counts=structure_counts(row['structure'])
            if key(counts)!=row['composition_key'] or sum(counts)!=row['nsites']:
                raise ValueError('Candidate composition mismatch')
            records[sid]=row;groups.setdefault(key(counts),[]).append(sid)
    pairable_keys={k for k,g in groups.items() if len(g)>=2}
    pairable_ids={sid for k in pairable_keys for sid in groups[k]}
    reduced_keys={reduced([int(n) for n in records[sid]['composition_key'].split('|')]) for sid in pairable_ids}
    print(f'Pairable candidate groups {len(pairable_keys)}; structures {len(pairable_ids)}',flush=True)
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher,ElementComparator
    matcher=StructureMatcher(**MATCHER,comparator=ElementComparator())
    candidate_structures={sid:Structure.from_dict(records[sid]['structure']) for sid in pairable_ids}
    old={};old_n=0
    print('Loading full old Matbench; keep only relevant reduced-composition buckets',flush=True)
    with gzip.open(args.old_matbench,'rt') as stream:raw=json.load(stream)
    for index,(structure,label) in zip(raw['index'],raw['data']):
        counts=structure_counts(structure);bucket=reduced(counts);old_n+=1
        if f'mb-mp-e-form-{index+1:06d}' in train_ids:train_keys.add(key(counts))
        if bucket in reduced_keys:
            old.setdefault(bucket,[]).append((int(index),Structure.from_dict(structure)))
    del raw
    excluded=[];survivors={};comparisons=0
    for number,sid in enumerate(sorted(pairable_ids)):
        row=records[sid];counts=[int(n) for n in row['composition_key'].split('|')]
        bucket=reduced(counts);current=candidate_structures[sid];duplicate=None
        for index,previous in old.get(bucket,[]):
            comparisons+=1
            if matcher.fit(current,previous):duplicate=f'mb-mp-e-form-{index+1:06d}';break
        if duplicate:
            excluded.append({'material_id':sid,'reason':'old_matbench_geometry','matched_id':duplicate});continue
        # Same reduced composition may have multiple full-count buckets. The
        # fixed matcher itself decides whether their current cells are equal.
        duplicate=None
        for other in survivors.get(bucket,[]):
            comparisons+=1
            if matcher.fit(current,candidate_structures[other]):duplicate=other;break
        if duplicate:
            excluded.append({'material_id':sid,'reason':'candidate_geometry_duplicate','matched_id':duplicate})
        else:survivors.setdefault(bucket,[]).append(sid)
        if (number+1)%100==0:print(f'Geometry checked {number+1}/{len(pairable_ids)}',flush=True)
    remaining={};kept={sid for ids in survivors.values() for sid in ids}
    for sid in sorted(kept):remaining.setdefault(records[sid]['composition_key'],[]).append(sid)
    selected_keys=choose_groups(remaining)
    selected_ids=sorted(sid for k in selected_keys for sid in remaining[k])
    if set(selected_ids)&historical:raise ValueError('Historical overlap after locking')
    for p,digest in hashes.items():
        if sha(Path(p))!=digest:raise ValueError('Source changed during audit')
    args.output.mkdir(parents=True)
    with gzip.open(args.output/'locked_structures.jsonl.gz','wt') as stream:
        for sid in selected_ids:stream.write(json.dumps(records[sid],separators=(',',':'))+'\n')
    columns=['material_id','api_material_id','created_at','last_updated','thermo_type','formation_energy_per_atom',
             'nsites','composition_key','entry_id','entry_run_type','label_geometry_energy_difference_ev_atom']
    with (args.output/'cohort_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:records[sid].get(k) for k in columns} for sid in selected_ids)
    with (args.output/'geometry_exclusions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['material_id','reason','matched_id']);writer.writeheader();writer.writerows(excluded)
    audit={'status':'LOCKED_NO_MODEL_FORWARD' if len(selected_keys)>=20 else 'FEWER_THAN20_GROUPS_NO_MODEL_FORWARD',
        'old_matbench_structures':old_n,'old_relevant_structures':sum(len(g) for g in old.values()),
        'candidate_pairable_groups_before_geometry':len(pairable_keys),'candidate_pairable_structures_before_geometry':len(pairable_ids),
        'geometry_comparisons':comparisons,'geometry_old_duplicates':sum(x['reason']=='old_matbench_geometry' for x in excluded),
        'geometry_candidate_duplicates':sum(x['reason']=='candidate_geometry_duplicate' for x in excluded),
        'eligible_groups_after_geometry':sum(len(g)>=2 for g in remaining.values()),'locked_groups':len(selected_keys),
        'locked_structures':len(selected_ids),'locked_pairs':sum(len(remaining[k])*(len(remaining[k])-1)//2 for k in selected_keys),
        'historical_id_overlap':0,'matcher':MATCHER,'comparator':'ElementComparator','matbench_geometry_matches_in_locked':0,
        'local_history_ids':len(local_history),'local_additional_exclusions':len(local_excluded),
        'train_composition_overlap_locked_groups':len(set(selected_keys)&train_keys),
        'train_composition_overlap_allowed':True,
        'source_hashes_unchanged':True,'input_hashes':hashes,'manifest_sha256':sha(args.output/'cohort_manifest.csv'),
        'structure_file_sha256':sha(args.output/'locked_structures.jsonl.gz'),'script_sha256':sha(Path(__file__)),
        'new_model_forwards':0,'new_training':False,'new_cohort_label_fit':False,
        'scope':'Against whole Matbench geometry and ID inventory of supported historical project files; no pretraining-independence guarantee; fixed matcher cell settings apply'}
    (args.output/'COHORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='input_hashes'}),flush=True)

if __name__=='__main__':main()
