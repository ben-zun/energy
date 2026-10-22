"""Require every scoped history gate and reconcile documented invalid fixtures."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root
 if a.output.exists():raise FileExistsError(a.output)
 immutable_binary_files=0
 for row in json.loads((root/'SERVER_SCOPE_INVENTORY.json').read_text())['files']:
  q=Path(row['path'])
  if q.suffix.lower() in {'.pt','.npz','.h5','.hdf5','.parquet','.npy'} and 'bytes' in row:
   st=q.stat()
   if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise ValueError('Frozen binary history input changed: '+str(q))
   immutable_binary_files+=1
 def read(name):return json.loads((root/name).read_text())
 main=read('server_history_registry_v1/AUDIT.json');repairs=read('server_history_repairs_v2/AUDIT.json');referenced=read('referenced_history_registry_v2/AUDIT.json');local=read('local_history_registry_v2/AUDIT.json');supplement=read('history_supplement_v2/AUDIT.json');graphs=read('history_graph_audit_v2/AUDIT.json');graph_repairs=read('history_graph_repairs_v2/AUDIT.json');graph_intermediate=read('history_graph_repairs_v8/AUDIT.json');graph_final=read('history_graph_repairs_v9/AUDIT.json');lid=read('local_identifiers_supplement_v2/AUDIT.json');fixtures=read('INVALID_TEST_FIXTURE_RESOLUTION.json')
 if not referenced['history_scope_complete'] or not local['history_scope_complete'] or graph_final['failures'] or supplement['unresolved'] or lid['unresolved_binary_inputs']:raise ValueError('Unresolved history coverage')
 for name in ['LOCAL_ARRAY_SCHEMA_AUDIT.json','SERVER_ARRAY_SCHEMA_AUDIT.json']:
  if read(name)['unresolved_array_inputs']:raise ValueError('Unknown array inputs: '+name)
 resolved_graph_paths={r['path'] for r in graph_repairs['cache_checks']+graph_intermediate['cache_checks']+graph_final['cache_checks']}
 if any(r['path'] not in resolved_graph_paths for r in graph_repairs['failures']+graph_intermediate['failures']):raise ValueError('Unresolved graph repair')
 if any(r['path'] not in resolved_graph_paths for r in graphs['failures']):raise ValueError('Unresolved graph input')
 repaired={r['path']:r for r in map(json.loads,(root/'server_history_repairs_v2/FILE_AUDIT.jsonl').open()) if r['status']=='parsed'}
 fixture_paths={r['cif_path'] for r in fixtures['fixtures']};empty={r['path']:r for r in supplement['empty_container_resolutions']};resolutions=[]
 for failure in main['failures']:
  path=failure['path']
  if path in repaired:
   if failure['sha256']!=repaired[path]['sha256']:raise ValueError('Repaired history input changed')
   resolution='same source reparsed successfully'
  elif path in fixture_paths:resolution='documented intentionally invalid test input'
  elif path in empty and failure['sha256']==empty[path]['sha256']:resolution='empty archive: no entries or geometry'
  else:raise ValueError('Unresolved main input: '+path)
  resolutions.append({'path':path,'sha256':failure['sha256'],'resolution':resolution})
 for failure in repairs['failures']:
  if failure['path'] not in fixture_paths|set(empty):raise ValueError('Unresolved parser repair')
 coverage={}
 for row in map(json.loads,(root/'server_history_registry_v1/FILE_AUDIT.jsonl').open()):
  if row['path'].endswith('matbench_mp_e_form.json.gz'):coverage['Matbench']=row
  if row['path'].endswith('2023-02-07-mp-computed-structure-entries.json.gz'):coverage['MP2022']=row
 if set(coverage)!={'Matbench','MP2022'} or coverage['Matbench'].get('events',{}).get('structure')!=132752 or coverage['MP2022'].get('events',{}).get('structure')!=154718:raise ValueError('Full raw historical source coverage missing')
 names=['server_history_registry_v1/AUDIT.json','server_history_repairs_v2/AUDIT.json','referenced_history_registry_v2/AUDIT.json','local_history_registry_v2/AUDIT.json','history_supplement_v2/AUDIT.json','history_graph_audit_v2/AUDIT.json','history_graph_repairs_v2/AUDIT.json','history_graph_repairs_v8/AUDIT.json','history_graph_repairs_v9/AUDIT.json','GRAPH_REPAIR4_SCOPE_v2.json','GRAPH_REPAIR6_SCOPE_v2.json','GRAPH_REPAIR_SCOPE_v2.json','local_identifiers_supplement_v2/AUDIT.json','LOCAL_ARRAY_SCHEMA_AUDIT.json','SERVER_ARRAY_SCHEMA_AUDIT.json','INVALID_TEST_FIXTURE_RESOLUTION.json','SERVER_SCOPE_INVENTORY.json','REFERENCED_INPUTS_SCOPE_v2.json']
 for registry in ['server_history_registry_v1','server_history_repairs_v2','referenced_history_registry_v2','local_history_registry_v2']:
  names.extend(registry+'/'+f for f in ['STRUCTURE_BUCKETS.jsonl.gz','TASK_REGISTRY_A.json','MATERIAL_PROVENANCE_REGISTRY.json','FILE_AUDIT.jsonl'])
 names.extend(registry+'/STRUCTURE_BUCKETS.jsonl.gz' for registry in ['history_graph_audit_v2','history_graph_repairs_v2','history_graph_repairs_v8','history_graph_repairs_v9'])
 for registry in ['history_supplement_v2','local_identifiers_supplement_v2']:names.extend(registry+'/'+f for f in ['TASK_REGISTRY_A.json','MATERIAL_PROVENANCE_REGISTRY.json'])
 result={'history_scope_complete':True,'scope':'frozen current-paper development scope plus its actual referenced structure inputs','task_exclusion':'explicit task namespace only','geometry_exclusion':'snapshot-level full-count bucket and preregistered matcher','material_exclusion':False,'full_raw_source_coverage':coverage,'failure_resolutions':resolutions,'local_counts':local['counts'],'server_counts':main['counts'],'referenced_counts':referenced['counts'],'repaired_counts':repairs['counts'],'graph_counts':graphs['counts'],'graph_repair_counts':graph_repairs['counts'],'graph_intermediate_counts':graph_intermediate['counts'],'graph_final_counts':graph_final['counts'],'supplement_counts':supplement['counts'],'archives':'all frozen scoped archives read; empty container separately resolved; no unrelated archive scope claim','input_sha256':{name:sha(root/name) for name in names},'not_claimed':'absence from original model pretraining is not established','immutable_binary_files_checked':immutable_binary_files,'model_forwards':0}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','failure_resolutions','full_raw_source_coverage']}),flush=True)
if __name__=='__main__':main()
