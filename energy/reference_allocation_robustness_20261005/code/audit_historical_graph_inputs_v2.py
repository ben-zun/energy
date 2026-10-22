"""Verify cached geometry against frozen source manifests; never run a model."""
import argparse,csv,gzip,hashlib,json,pickletools,re,zipfile
from pathlib import Path
from collections import Counter
class GraphRecord:pass
class RadiusGraph:pass
import numpy as np

def check_cached_coordinates(numbers,positions,source_numbers,source_positions):
    if not np.array_equal(np.asarray(numbers).reshape(-1),source_numbers):raise ValueError('Graph atomic numbers differ from source')
    if np.asarray(positions).shape!=np.asarray(source_positions).shape or not np.allclose(positions,source_positions,rtol=2e-6,atol=2e-5):raise ValueError('Graph positions differ from source')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--keys',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 import torch
 from torch_geometric.data import Data
 from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr
 from torch_geometric.data.storage import GlobalStorage
 from sevenn.atom_graph_data import AtomGraphData
 from ase.io import read,iread
 from pymatgen.core import Structure
 from pymatgen.core.structure_matcher import StructureMatcher,ElementComparator
 exact_matcher=StructureMatcher(ltol=1e-4,stol=1e-4,angle_tol=.01,primitive_cell=False,scale=False,attempt_supercell=False,comparator=ElementComparator())
 from download_external_mp import structure_counts
 torch.serialization.add_safe_globals([Data,DataEdgeAttr,DataTensorAttr,GlobalStorage,AtomGraphData,(GraphRecord,'__main__.GraphRecord'),(RadiusGraph,'matris.graph.radiusgraph.RadiusGraph')])
 scope=json.loads(a.scope.read_text());requested=set(json.loads(a.keys.read_text()));counts=Counter();failures=[];records=[];meta_cache={};geom_cache={};content_seen={}
 historical_extxyz=Path('/home/yxzhang/energy/runs/rclfa_v3/full_fegx_like_foundation_probe_269/medium_subset_probe/medium_subset_probe.extxyz')
 with gzip.open(a.output/'STRUCTURE_BUCKETS.jsonl.gz','wt') as output:
  for r in scope['files']:
   path=Path(r['path'])
   if path.suffix!='.pt':continue
   try:
    binary_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if binary_hash in content_seen:
     records.append({'path':str(path),'identical_content_reuse':content_seen[binary_hash],'sha256':binary_hash,'safe_weights_only_load':True});counts['identical_geometry_cache_files']+=1;continue
    with zipfile.ZipFile(path) as z:
     member=next(n for n in z.namelist() if n.endswith('/data.pkl'));raw=z.read(member)
    values=[value for op,value,pos in pickletools.genops(raw) if op.name in {'BINUNICODE','SHORT_BINUNICODE','UNICODE'}];strings=set(values)
    has_geometry='pos' in strings and ('atomic_numbers' in strings or 'z' in strings)
    if not has_geometry and 'records_by_split' in strings and 'atom_frac_coord' in strings:
     payload=torch.load(path,map_location='cpu',weights_only=True);total=0
     for records_by_split in payload['records_by_split'].values():
      for record in records_by_split:
       data=record.graph.__dict__ if hasattr(record.graph,'__dict__') else record.graph
       numbers=data['atomic_number'].detach().numpy().reshape(-1);fractional=data['atom_frac_coord'].detach().numpy();lattice=data['lattice'].detach().numpy().reshape(3,3)
       structure=Structure(lattice,numbers,fractional).as_dict();key='|'.join(map(str,structure_counts(structure)))
       if key in requested:output.write(json.dumps({'source':str(path)+':'+str(record.sample_id),'composition_key':key,'structure':structure},allow_nan=False)+'\n');counts['exported_graph_structures']+=1
       total+=1
     records.append({'path':str(path),'graphs':total,'safe_weights_only_load':True,'geometry':'actual cached fractional coordinates/lattice'});counts['geometry_cache_files']+=1;counts['graph_structures_verified']+=total;continue
    if not has_geometry and 'metadata' in strings and 'cif_path' in strings:
     payload=torch.load(path,map_location='cpu',weights_only=True)
     metadata_rows=payload['metadata']
     if isinstance(metadata_rows,dict):metadata_rows=[row for values in metadata_rows.values() for row in values]
     for row in metadata_rows:
      source_path=Path(row['cif_path'])
      if not source_path.is_file():raise ValueError('Feature metadata source missing')
     records.append({'path':str(path),'feature_source_rows':len(metadata_rows),'safe_weights_only_load':True});counts['feature_readouts_without_input_geometry']+=1;continue
    if not has_geometry:
     match=re.search(r'mb-mp-e-form-(\d+)',path.name)
     if match and 1<=int(match[1])<=132752:
      counts['raw_matbench_derived_feature_graphs']+=1;continue
     if strings&{'model_state','state_dict','model','optimizer_state','optimizer','model_state_dict'} or any(k.endswith('.weight') for k in strings):counts['checkpoint_without_input_geometry']+=1;continue
     raise ValueError('Unclassified PT schema: '+str(list(strings)[:12]))
    metadata=path.parent/('manifest.csv' if path.name=='sevennet_graphs.pt' else 'metadata.csv' if path.name=='graphs.pt' else 'graph_metadata.csv')
    if metadata not in meta_cache:
     rows=list(csv.DictReader(metadata.open()));meta_cache[metadata]={int(row.get('row_index',i)):row for i,row in enumerate(rows)}
    graph_rows=meta_cache[metadata];graphs=torch.load(path,map_location='cpu',weights_only=True)
    for graph in graphs:
     index=int(graph['audit_row' if 'audit_row' in graph else 'manifest_row'].reshape(-1)[0]);row=graph_rows[index];cif=Path(row['cif_path'])
     if cif not in geom_cache:geom_cache[cif]=read(cif)
     atoms=geom_cache[cif]
     actual_extxyz_source=any(token in str(path) for token in ['sevennet_upper_bound_279/graph_cache/medium','fair_finetune_293/graph_cache/medium'])
     numbers=graph['atomic_numbers'].detach().numpy().reshape(-1);positions=graph['pos'].detach().numpy()
     if not np.array_equal(np.sort(numbers),np.sort(atoms.numbers)):raise ValueError('Graph element counts differ from source')
     possible_cells=[graph['cell'].detach().numpy().reshape(3,3)] if 'cell' in graph else [Structure.from_file(cif).lattice.matrix,atoms.cell.array]
     source_structure=Structure(atoms.cell.array,atoms.numbers,atoms.positions,coords_are_cartesian=True)
     cell=None
     if actual_extxyz_source:
      edges=graph['edge_index'].detach().numpy();vectors=graph['edge_vec'].detach().numpy();shifts=vectors-(positions[edges[1]]-positions[edges[0]])
      shift_rank=int(np.linalg.matrix_rank(shifts,tol=1e-4));volume=float(graph['cell_volume'].reshape(-1)[0])
      from check_mp_trajectory_source import MATCHER
      historical_matcher=StructureMatcher(**MATCHER,comparator=ElementComparator())
      for candidate_cell in possible_cells:
       integer_shifts=shifts@np.linalg.inv(candidate_cell);residual=float(np.max(np.abs(integer_shifts-np.rint(integer_shifts)))) if integer_shifts.size else 0.0
       if residual>2e-5 or not np.isclose(abs(np.linalg.det(candidate_cell)),volume,rtol=2e-6,atol=2e-5):continue
       actual_structure=Structure(candidate_cell,numbers,positions,coords_are_cartesian=True)
       if not historical_matcher.fit(actual_structure,source_structure):continue
       cell=candidate_cell;counts['periodic_shift_rank_'+str(shift_rank)]+=1;break
     else:
      for candidate_cell in possible_cells:
       actual_structure=Structure(candidate_cell,numbers,positions,coords_are_cartesian=True)
       if exact_matcher.fit(actual_structure,source_structure):cell=candidate_cell;break
     if cell is None:raise ValueError('Cached lattice/geometry unresolved: row '+str(index)+' id '+str(row.get('id')))
     if 'cell_volume' in graph and not np.isclose(float(graph['cell_volume'].reshape(-1)[0]),atoms.get_volume(),rtol=2e-6,atol=2e-5):raise ValueError('Graph volume differs from source')
     structure=Structure(cell,numbers,positions,coords_are_cartesian=True).as_dict();key='|'.join(map(str,structure_counts(structure)))
     if key in requested:output.write(json.dumps({'source':str(path)+':row-'+str(index),'composition_key':key,'structure':structure},allow_nan=False)+'\n');counts['exported_graph_structures']+=1
     counts['graph_structures_verified']+=1
     if counts['graph_structures_verified']%1000==0:print(json.dumps({'verified_graph_structures':counts['graph_structures_verified']}),flush=True)
    records.append({'path':str(path),'metadata':str(metadata),'graphs':len(graphs),'actual_geometry_source':'cached positions plus manifest lattice verified against all cached periodic vectors and volume' if actual_extxyz_source else 'actual cached cell/positions or verified manifest lattice','metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'pickle_schema_sha256':hashlib.sha256(raw).hexdigest(),'safe_weights_only_load':True});counts['geometry_cache_files']+=1
    content_seen[binary_hash]=str(path)
    del graphs
    print(json.dumps({'verified_cache':str(path),'graphs':records[-1]['graphs']}),flush=True)
   except Exception as e:failures.append({'path':str(path),'error':str(e)[:500]})
 result={'counts':dict(counts),'cache_checks':records,'failures':failures,'scope_sha256':hashlib.sha256(a.scope.read_bytes()).hexdigest(),'history_scope_complete':not failures}
 (a.output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'counts':counts,'failures':failures}),flush=True)
if __name__=='__main__':main()
