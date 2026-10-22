"""Compare calculator outputs with original batched energy-only graph inference."""
import argparse,csv,gzip,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort-root',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import torch
 import sevenn.util as util
 from sevenn.atom_graph_data import AtomGraphData
 from sevenn.train.dataload import unlabeled_atoms_to_graph
 from torch_geometric.loader import DataLoader
 from pymatgen.core import Structure
 from pymatgen.io.ase import AseAtomsAdaptor
 import sevenn._keys as KEY
 torch.set_num_threads(4)
 protocol=json.loads((a.cohort_root/'FINAL_TRAJECTORY_PROTOCOL.json').read_text());geometries={r['snapshot_id']:r['structure'] for r in map(json.loads,gzip.open(a.cohort_root/'FINAL_TRAJECTORY_STRUCTURES.jsonl.gz','rt'))};pred={(r['route'],r['snapshot_id']):float(r['raw_network_energy_per_atom']) for r in csv.DictReader(a.predictions.open())};snapshot_ids=['mp-3530684:ionic-step-2','mp-3530684:ionic-step-95'];records=[]
 routes=[('Frozen',None)]+[(('Direct_' if r['route']=='U' else 'Aligned_')+str(r['seed']),r) for r in protocol['selected_checkpoints'] if r['seed']==20260528]
 for route,selected in routes:
  model,_=util.model_from_checkpoint('7net-0');model.delete_module_by_key('force_output');model.key_grad=None;model.set_is_batch_data(True)
  if selected:model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True)
  model.to('cuda:0').eval()
  for sid in snapshot_ids:
   atoms=AseAtomsAdaptor.get_atoms(Structure.from_dict(geometries[sid]));graph=AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms,cutoff=float(model.cutoff),with_shift=False));batch=next(iter(DataLoader([graph],batch_size=1))).to('cuda:0')
   with torch.no_grad():output=model(batch.to_dict())
   value=float(output[KEY.PRED_TOTAL_ENERGY].reshape(-1)[0])/len(atoms);reference=pred[route,sid];difference=abs(value-reference)
   if not math.isclose(value,reference,rel_tol=2e-5,abs_tol=2e-5):raise ValueError('Original graph/calculator inconsistency: '+str((route,sid,value,reference)))
   records.append({'route':route,'snapshot_id':sid,'original_batched_graph_raw_eV_atom':value,'calculator_raw_eV_atom':reference,'absolute_difference':difference});print(route,sid,'PASS',difference,flush=True)
 a.output.write_text(json.dumps({'passed':True,'checks':len(records),'records':records,'purpose':'check high-error snapshot and ordinary late snapshot using original batched energy-only path; no model/data selection changes'},indent=2)+'\n')
if __name__=='__main__':main()
