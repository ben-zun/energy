"""Eight frozen-checkpoint outputs, strictly after certified cohort lock."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def h_offset(coefficients,counts):
 reference=[r for r in coefficients if r['calibration']=='element_reference']
 mu={int(r['coefficient'][4:]):float(r['value']) for r in reference if r['coefficient'].startswith('mu_Z')}
 biases=[float(r['value']) for r in reference if r['coefficient']=='bias']
 if len(biases)!=1:raise ValueError('Original reference bias missing')
 if len(mu)!=118:raise ValueError('Original reference incomplete')
 return -(sum(mu[z]*counts[z-1] for z in range(1,119))/sum(counts)+biases[0])

def validate_lock(root):
 lock=json.loads((root/'FINAL_ARTIFACT_SHA256.json').read_text())
 required={'FINAL_TRAJECTORY_COHORT.csv','FINAL_TRAJECTORY_STRUCTURES.jsonl.gz','FINAL_TRAJECTORY_PROTOCOL.json','FINAL_HISTORY_AUDIT.json'}
 if set(lock)!=required:raise ValueError('Incomplete locked artifact set')
 for name,digest in lock.items():
  if sha(root/name)!=digest:raise ValueError('Locked artifact changed: '+name)
 audit=json.loads((root/'FINAL_HISTORY_AUDIT.json').read_text())
 if audit.get('history_scope_complete') is not True or audit.get('certified_task_groups',0)<20:raise ValueError('Cohort not certified/minimum unmet')
 protocol=json.loads((root/'FINAL_TRAJECTORY_PROTOCOL.json').read_text())
 if protocol.get('cohort_locked_before_models') is not True or protocol.get('certified_task_groups')!=audit['certified_task_groups']:raise ValueError('Locked protocol/audit mismatch')
 return protocol

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 protocol=validate_lock(a.cohort_root)
 if a.output.exists():raise FileExistsError(a.output)
 ref=Path(protocol['reference_file']['path'])
 if sha(ref)!=protocol['reference_file']['sha256']:raise ValueError('Frozen h hash changed')
 coeff=list(csv.DictReader(ref.open()))
 rows=list(csv.DictReader((a.cohort_root/'FINAL_TRAJECTORY_COHORT.csv').open()))
 geom={r['snapshot_id']:r['structure'] for r in map(json.loads,gzip.open(a.cohort_root/'FINAL_TRAJECTORY_STRUCTURES.jsonl.gz','rt'))}
 if len(geom)!=len(rows) or set(geom)!={r['snapshot_id'] for r in rows}:raise ValueError('Locked geometry incomplete')
 # Heavy model dependencies are only loaded after lock validation.
 import torch
 from sevenn.calculator import SevenNetCalculator
 from pymatgen.core import Structure
 from pymatgen.io.ase import AseAtomsAdaptor
 torch.set_num_threads(4)
 predictions=[];model_audit=[];states={}
 # Validate all six original checkpoint wrappers before any predictions.
 template=SevenNetCalculator('7net-0',device=a.device);expected_state=template.model.state_dict()
 for selection in protocol['selected_checkpoints']:
  checkpoint=Path(selection['checkpoint'])
  if sha(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
  for field,value in [('route',selection['route']),('seed',selection['seed']),('epoch',selection['selected_epoch'])]:
   if payload.get(field)!=value:raise ValueError('Original checkpoint metadata mismatch: '+field)
  state=payload['model']
  if set(state)!=set(expected_state) or any(state[k].shape!=expected_state[k].shape for k in state):raise ValueError('Checkpoint model parameter contract changed')
  states[str(checkpoint)]=state
 del template
 expected_initial={x['initial_hash'] for x in protocol['selected_checkpoints']}
 if len(expected_initial)!=1:raise ValueError('Inconsistent original initial model hashes')
 routes=[('Frozen',None)]+[(('Direct_' if x['route']=='U' else 'Aligned_')+str(x['seed']),x) for x in protocol['selected_checkpoints']]
 a.output.mkdir(parents=True)
 for name,selection in routes:
  calculator=SevenNetCalculator('7net-0',device=a.device)
  if name=='Frozen':
   fingerprint=hashlib.sha256();count=0
   for parameter_name,parameter in sorted(calculator.model.named_parameters()):
    if parameter_name.startswith(('3_','4_','reduce_input_to_hidden','reduce_hidden_to_energy','rescale_atomic_energy')):
     count+=parameter.numel();fingerprint.update(parameter_name.encode());fingerprint.update(parameter.detach().cpu().numpy().astype(np.float64).tobytes())
   if count!=301300 or fingerprint.hexdigest() not in expected_initial:raise ValueError('Original pretrained model identity changed')
  if selection:
   checkpoint=Path(selection['checkpoint'])
   if sha(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
   calculator.model.load_state_dict(states[str(checkpoint)],strict=True);calculator.model.to(a.device).eval()
  for row in rows:
   sid=row['snapshot_id'];s=Structure.from_dict(geom[sid]);atoms=AseAtomsAdaptor.get_atoms(s);atoms.calc=calculator
   energy=float(atoms.get_potential_energy())/len(atoms)
   if not np.isfinite(energy):raise ValueError('Nonfinite model energy')
   offset=h_offset(coeff,list(map(int,row['composition_key'].split('|'))))
   predictions.append({'snapshot_id':sid,'task_id':row['task_id'],'route':name,'raw_network_energy_per_atom':energy,'native_energy_per_atom':energy+offset if name.startswith('Aligned_') else energy})
   if name=='Frozen':
    offset=h_offset(coeff,list(map(int,row['composition_key'].split('|'))))
    predictions.append({'snapshot_id':sid,'task_id':row['task_id'],'route':'Frozen+h','raw_network_energy_per_atom':energy,'native_energy_per_atom':energy+offset})
  model_audit.append({'route':name,'selection':selection,'snapshots':len(rows),**({'initial_trainable_parameters':count,'initial_trainable_hash':fingerprint.hexdigest()} if name=='Frozen' else {'checkpoint_sha256':sha(selection['checkpoint'])})});print(f'{name}: {len(rows)} frozen cohort outputs complete',flush=True)
  del calculator
  if torch.cuda.is_available():torch.cuda.empty_cache()
 with (a.output/'PREDICTIONS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
 (a.output/'MODEL_AUDIT.json').write_text(json.dumps({'routes':model_audit,'output_routes':8,'frozen_h_derived_from_identical_frozen_forward':True,'new_training':False,'refit':False,'locked_inputs':json.loads((a.cohort_root/'FINAL_ARTIFACT_SHA256.json').read_text())},indent=2)+'\n')
if __name__=='__main__':main()
