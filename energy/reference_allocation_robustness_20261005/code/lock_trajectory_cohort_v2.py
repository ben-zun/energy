"""Lock the certified cohort with original models/reference and complete hashes."""
import argparse,csv,gzip,hashlib,json,math
from download_external_mp import structure_counts
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--certification',type=Path,required=True);p.add_argument('--history-certificate',type=Path,required=True);p.add_argument('--material-provenance',type=Path,required=True);p.add_argument('--material-registry',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 protocol=json.loads(a.protocol.read_text());ca=json.loads((a.certification/'CERTIFICATION_AUDIT.json').read_text());history=json.loads(a.history_certificate.read_text())
 if history.get('history_scope_complete') is not True or ca.get('history_scope_complete') is not True:raise ValueError('Incomplete history')
 n=ca['certified_task_groups']
 if n<20:raise ValueError('Original minimum20 not met')
 if n<50 and ca['fixed_candidate_prefix']!=10000:raise ValueError('Must continue fixed prefix10000 before locking fewer than50')
 links=json.loads(a.material_provenance.read_text())['task_material_links'];seen=set()
 for file in a.material_registry:seen.update(json.loads(file.read_text()))
 source=a.certification/'CERTIFIED_SNAPSHOTS_PRELOCK.jsonl.gz'
 if sha(source)!=ca['certified_snapshots_sha256']:raise ValueError('Certified snapshots changed')
 for q in [a.protocol,a.history_certificate]:
  if ca['input_sha256'].get(str(q))!=sha(q):raise ValueError('Certification input changed: '+str(q))
 rows=list(map(json.loads,gzip.open(source,'rt')));task_ids={r['canonical_task_id'] for r in rows}
 if len(task_ids)!=n or set(links)!=task_ids:raise ValueError('Task provenance coverage mismatch')
 if any(sum(1 for r in rows if r['canonical_task_id']==t)<2 for t in task_ids):raise ValueError('Task lost pairability')
 a.output.mkdir(parents=True);cohort=[]
 with gzip.open(a.output/'FINAL_TRAJECTORY_STRUCTURES.jsonl.gz','wt') as f:
  for r in rows:
   counts=list(map(int,r['composition_key'].split('|')))
   if len(counts)!=118 or counts!=structure_counts(r['structure']) or sum(counts)!=r['nsites'] or not 0<r['num_electronic_steps']<r['NELM']:raise ValueError('Snapshot counts/SCF contract changed')
   if not all(math.isfinite(r[k]) for k in ['dft_total_energy_ev','dft_energy_ev_atom']):raise ValueError('Nonfinite DFT energy')
   if abs(r['dft_total_energy_ev']/r['nsites']-r['dft_energy_ev_atom'])>1e-10:raise ValueError('Energy unit mismatch')
   geometry_hash=hashlib.sha256(json.dumps(r['structure'],sort_keys=True,separators=(',',':')).encode()).hexdigest();mids=links[r['canonical_task_id']]
   cohort.append({'task_id':r['canonical_task_id'],'material_id':'|'.join(mids),'material_link_status':'available' if mids else 'unavailable_in_current_core','material_seen_before':any(m in seen for m in mids),'run_type':r['run_type'],'composition_key':r['composition_key'],'snapshot_id':r['snapshot_id'],'ionic_step':r['ionic_step_index'],'n_atoms':r['nsites'],'DFT_energy_eV':r['dft_total_energy_ev'],'DFT_energy_per_atom':r['dft_energy_ev_atom'],'structure_hash':geometry_hash,'history_task_hit':False,'history_geometry_hit':False,'source_match':True,'SCF_pass':True,'num_electronic_steps':r['num_electronic_steps'],'NELM':r['NELM']})
   f.write(json.dumps({'snapshot_id':r['snapshot_id'],'structure':r['structure'],'structure_hash':geometry_hash},allow_nan=False)+'\n')
 with (a.output/'FINAL_TRAJECTORY_COHORT.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cohort[0]));w.writeheader();w.writerows(cohort)
 protocol.update({'initial_protocol_sha256':sha(a.protocol),'cohort_locked_before_models':True,'certified_task_groups':n,'actual_snapshots':len(cohort),'certification_sha256':sha(a.certification/'CERTIFICATION_AUDIT.json'),'material_provenance_sha256':sha(a.material_provenance),'no_add_or_remove_groups_after_lock':True})
 (a.output/'FINAL_TRAJECTORY_PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
 history.update({'certified_task_groups':n,'snapshots':len(cohort),'pairs':ca['pairs'],'stopping_rank':ca['stopping_rank'],'candidate_prefix':ca['fixed_candidate_prefix'],'certification':ca,'material_provenance_sha256':sha(a.material_provenance),'material_history_is_not_an_exclusion':True})
 (a.output/'FINAL_HISTORY_AUDIT.json').write_text(json.dumps(history,indent=2)+'\n')
 files=['FINAL_TRAJECTORY_COHORT.csv','FINAL_TRAJECTORY_STRUCTURES.jsonl.gz','FINAL_TRAJECTORY_PROTOCOL.json','FINAL_HISTORY_AUDIT.json']
 (a.output/'FINAL_ARTIFACT_SHA256.json').write_text(json.dumps({name:sha(a.output/name) for name in files},indent=2)+'\n');print(json.dumps({'locked_tasks':n,'snapshots':len(cohort),'model_forwards':0}))
if __name__=='__main__':main()
