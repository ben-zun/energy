"""Close CIF identifier and Parquet reference coverage without redoing geometry scans."""
import argparse,gzip,hashlib,json,re
from pathlib import Path
from collections import Counter
from build_paper_history_registry import TASK_FIELDS,MATERIAL_FIELDS,id_values,parse_stream,ARCHIVES
from prepare_referenced_history_inputs_v2 import normalize

def cif_identifiers(raw):
    for line in raw.splitlines():
        match=re.match(r'^\s*_(\w+)\s+(.+?)\s*$',line)
        if not match:continue
        field,value=match.groups();field=field.lower()
        if field not in TASK_FIELDS|MATERIAL_FIELDS:continue
        value=value.strip().strip("'\"")
        for identifier in id_values(value):yield ('task' if field in TASK_FIELDS else 'material',identifier)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--referenced-scope',type=Path,required=True);p.add_argument('--reference-resolution',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);scope=json.loads(a.scope.read_text());refs=json.loads(a.referenced_scope.read_text());known={str(Path(r['path']).resolve()) for r in scope['files']+refs['files']};new={};tasks={};materials={};counts=Counter();unresolved=[];evidence=[]
    def ids(kind,value,source):
        (tasks if kind=='task' else materials).setdefault(value,set()).add(source)
    def reference(value,origin):
        if not value or str(value).lower() in {'nan','none'}:return
        raw=Path(str(value));choices=[normalize(raw),Path('/home/yxzhang/energy')/raw] if not raw.is_absolute() else [normalize(raw)]
        # Published project-relative JARVIS references retain their directory suffix.
        if 'jarvis_forme_official/' in str(raw):choices.append(Path('/home/yxzhang/energy/jarvis_forme_official')/str(raw).rsplit('jarvis_forme_official/',1)[1])
        q=next((q.resolve() for q in choices if q.is_file()),None)
        if q is None:unresolved.append({'path':str(raw),'origin':origin});return
        counts['resolved_references']+=1
        if str(q) in known:counts['already_audited_geometry']+=1;return
        if str(q) not in new:
            st=q.stat();new[str(q)]={'path':str(q),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'example_reference_origin':origin}
    for r in scope['files']:
        path=Path(r['path']);ext=path.suffix.lower()
        if ext=='.cif':
            try:
                with path.open() as f:
                    for line in f:
                        for kind,value in cif_identifiers(line):ids(kind,value,str(path))
                counts['cif_identifier_files']+=1
            except Exception as e:unresolved.append({'path':str(path),'error':str(e)})
        elif ext=='.parquet':
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(path);columns=[c for c in pf.schema.names if c.lower() in {'cif_path','structure_path','r2scan_cif_path'}]
            for batch in pf.iter_batches(columns=columns,batch_size=4096):
                for row in batch.to_pylist():
                    for field,value in row.items():reference(value,str(path)+':'+field)
            counts['parquet_files']+=1
        elif path.name.lower().endswith(ARCHIVES):
            if path.stat().st_size==0:
                evidence.append({'path':str(path),'bytes':0,'sha256':hashlib.sha256(b'').hexdigest(),'resolution':'empty container: contains no entries or geometry'});continue
            try:
                with path.open('rb') as f:
                    for kind,value in parse_stream(f,path.name):
                        if kind in {'task','material'}:ids(kind,value,str(path))
                counts['archive_identifier_rescans']+=1
            except Exception as e:unresolved.append({'path':str(path),'error':str(e)})
        if counts['cif_identifier_files'] and counts['cif_identifier_files']%100000==0:print(json.dumps(dict(counts)),flush=True)
    resolution=json.loads(a.reference_resolution.read_text())
    for row in resolution['unresolved']:reference(row['path'],row['origin'])
    for name,data in [('TASK_REGISTRY_A.json',tasks),('MATERIAL_PROVENANCE_REGISTRY.json',materials)]:
        (a.output/name).write_text(json.dumps({k:sorted(v) for k,v in data.items()},indent=2)+'\n')
    (a.output/'ADDITIONAL_GEOMETRY_INVENTORY.json').write_text(json.dumps({'roots':[],'files':list(new.values()),'role':'actual scoped Parquet/JARVIS historical inputs'},indent=2)+'\n')
    result={'counts':dict(counts),'additional_inputs':len(new),'unresolved':unresolved,'empty_container_resolutions':evidence,'task_ids':len(tasks),'material_ids':len(materials),'input_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.scope,a.referenced_scope,a.reference_resolution]}}
    (a.output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)
if __name__=='__main__':main()
