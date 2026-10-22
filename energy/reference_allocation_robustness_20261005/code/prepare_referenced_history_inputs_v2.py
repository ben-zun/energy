"""Resolve historical path aliases and underlying manifest-referenced inputs.

Scope derives only from frozen paper manifests, never from model outcomes.
Published one-based Matbench IDs are covered by the audited full raw source.
"""
import argparse,csv,gzip,json,re
from pathlib import Path
from collections import Counter
BASE=Path('/home/yxzhang/energy')
def normalize(path):
 text=str(path)
 anchors=['/runs/rclfa_v3/','/external_data/','/external_datasets/','/external_benchmarks/']
 hits=[(text.rfind(anchor),anchor) for anchor in anchors if anchor in text]
 if hits:
  _,anchor=max(hits);return BASE/(anchor.strip('/')+'/'+text.rsplit(anchor,1)[1])
 return Path(text)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--server-refs',type=Path,required=True);p.add_argument('--local-refs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=Path,required=True);a=p.parse_args()
 if a.output.exists() or a.resolution.exists():raise FileExistsError('New referenced-input audit paths required')
 scope=json.loads(a.scope.read_text());known={str(Path(r['path']).resolve()) for r in scope['files']};server=json.loads(a.server_refs.read_text());additional={};counts=Counter();unresolved=[];mb_ids=set()
 def add(q,origin):
  resolved=q.resolve();key=str(resolved)
  if key in known:counts['already_frozen_scope']+=1;return
  if key in additional:counts['duplicate_underlying_source']+=1;return
  st=resolved.stat();additional[key]={'path':key,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'example_reference_origin':origin};counts['additional_unique_inputs']+=1
 for row in server['files']:add(Path(row['path']),row['reference_origins'][0])
 def resolve(path,origin):
  q=normalize(path)
  if q.exists():add(q,origin);counts['resolved_existing_geometry']+=1;return
  mb=re.fullmatch(r'mb-mp-e-form-(\d+)\.cif',q.name)
  if mb and 1<=int(mb[1])<=132752:
   mb_ids.add(int(mb[1]));counts['one_based_matbench_source_aliases']+=1;return
  unresolved.append({'path':str(q),'original_reference':path,'origin':origin})
 for row in server['missing_references']:resolve(row['referenced_path'],row['origins'][0])
 with gzip.open(a.local_refs,'rt') as f:
  for row in csv.DictReader(f):resolve(row['path'],row['example_origin'])
 a.output.write_text(json.dumps({'roots':[],'files':list(additional.values()),'role':'underlying historical inputs referenced exclusively by frozen current-paper manifests'},indent=2)+'\n')
 a.resolution.write_text(json.dumps({'counts':dict(counts),'matbench_one_based_ids':len(mb_ids),'matbench_raw_source':'/home/yxzhang/energy/external_benchmarks/matbench_mp_e_form/matbench_mp_e_form.json.gz','mapping_evidence':'phase3_independent_v1/phase3_history_audit_v1/AUDIT.json verifies source version, full132752 labels/counts and sampled geometries; raw full geometry is parsed in scoped registry','unresolved':unresolved,'history_scope_complete':False},indent=2)+'\n');print(json.dumps({'counts':counts,'unresolved':len(unresolved)}),flush=True)
if __name__=='__main__':main()
