"""Check explicit historical task fields; does not analyze predictions.

Bare generic task IDs are possible MP hits, conservatively excluded. Archives,
XYZ and oversized JSON are disclosed, so this alone is not a history certificate.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from inventory_mp_trajectory_metadata import task_id

TASK_FIELDS={'task_id','task_ids','mp_task_id','vasp_task_id'}


def values(value):
    if isinstance(value,list):
        for item in value:yield from values(item)
    elif isinstance(value,dict):
        if 'identifier' in value:yield from values(value['identifier'])
    elif isinstance(value,(str,int)) and not isinstance(value,bool):
        try:yield task_id(value)
        except ValueError:pass


def task_values(data):
    if isinstance(data,dict):
        for k,v in data.items():
            if k.lower() in TASK_FIELDS:yield from values(v)
            else:yield from task_values(v)
    elif isinstance(data,list):
        for v in data:yield from task_values(v)


def inspect(path):
    opener=gzip.open if path.suffix=='.gz' else open
    plain=path.name.removesuffix('.gz').lower()
    with opener(path,'rt',encoding='utf-8-sig') as stream:
        if plain.endswith(('.csv','.tsv')):
            reader=csv.DictReader(stream,delimiter='\t' if plain.endswith('.tsv') else ',')
            columns=[n for n in (reader.fieldnames or []) if n.lower() in TASK_FIELDS]
            if not columns:return set(),False
            found=set()
            for row in reader:
                for n in columns:
                    value=row.get(n)
                    if value and value.startswith(('[','{')):
                        try:value=json.loads(value)
                        except json.JSONDecodeError:continue
                    found.update(values(value))
            return found,True
        if plain.endswith('.jsonl'):
            found=set()
            for line in stream:
                if len(line)>64*1024*1024:raise ValueError('Oversized JSONL record')
                found.update(task_values(json.loads(line)))
            return found,True
        raw=stream.read(64*1024*1024+1)
        if len(raw)>64*1024*1024:raise ValueError('Oversized JSON document')
        return set(task_values(json.loads(raw))),True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--exclude',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output path required')
    csv.field_size_limit(64*1024*1024)
    candidates={r['canonical_task_id'] for r in csv.DictReader(a.manifest.open())}
    excluded=[p.resolve() for p in a.exclude]+[a.output.resolve()]
    examined=0;task_field_files=0;hits={};failures=[];archives=0
    for folder,dirs,files in os.walk(a.root):
        dirs[:]=[d for d in dirs if d not in {'.git','node_modules','__pycache__','external_repos','external_packages'}
                 and not any((Path(folder)/d).resolve().is_relative_to(e) for e in excluded)]
        for name in files:
            f=Path(folder)/name
            if any(f.resolve().is_relative_to(e) for e in excluded):continue
            plain=name.removesuffix('.gz').lower()
            if name.endswith(('.zip','.tar','.tar.gz','.tgz')):archives+=1
            tabular=plain.endswith(('.csv','.tsv'))
            js=plain.endswith(('.json','.jsonl')) and any(w in str(f).lower() for w in
                ['manifest','registry','prediction','diagnostic','split','results','reports','audit'])
            if not (tabular or js):continue
            examined+=1
            try:
                ids,had=inspect(f);task_field_files+=had
                overlap=ids & candidates
                if overlap:hits[str(f)]=sorted(overlap)
            except Exception as e:failures.append({'path':str(f),'error_type':type(e).__name__,'message':str(e)[:200]})
    union=sorted({v for vv in hits.values() for v in vv})
    r={'status':'explicit_task_field_audit_only_not_geometry_certificate',
       'candidate_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
       'candidates':len(candidates),'examined_files':examined,'task_field_files':task_field_files,
       'possible_historical_task_hits':len(union),'hit_task_ids':union,'hit_files':hits,
       'failures':failures,'archive_files_not_scanned':archives,
       'unresolved_scope':['archive contents','XYZ/extXYZ bare task fields','historical geometry'],
       'excluded_roots':[str(p) for p in excluded],'model_forwards':0}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ['examined_files','task_field_files','possible_historical_task_hits','archive_files_not_scanned']}),flush=True)


if __name__=='__main__':main()
