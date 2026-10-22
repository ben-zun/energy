"""Check source trajectory against its own official task, not model predictions."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.core.structure_matcher import StructureMatcher, ElementComparator
from inventory_mp_trajectory_metadata import task_id

MATCHER = {'ltol':.2,'stol':.3,'angle_tol':5,'primitive_cell':False,
           'scale':False,'attempt_supercell':False}


def check_source(trajectory, task):
    nsteps = trajectory['num_ionic_steps']
    identifier = task_id(trajectory['identifier'])
    if identifier != task_id(task['task_id']):
        raise ValueError('Task identity mismatch')
    if trajectory['run_type'] != task['run_type']:
        raise ValueError('Run type mismatch')
    elements = trajectory['elements']
    coords, cells = trajectory['cart_coords'], trajectory['lattice']
    if nsteps < 2 or len(coords) != nsteps or len(cells or []) not in (1,nsteps):
        raise ValueError('Incomplete per-frame geometry')
    if len(elements) != task['nsites'] or any(not 1 <= z <= 118 for z in elements):
        raise ValueError('Element/count mismatch')
    for c in coords:
        a = np.asarray(c,dtype=float)
        if a.shape != (len(elements),3) or not np.isfinite(a).all():
            raise ValueError('Invalid Cartesian coordinates')
    for cell in cells:
        a = np.asarray(cell,dtype=float)
        if a.shape != (3,3) or not np.isfinite(a).all() or abs(np.linalg.det(a)) <= 1e-12:
            raise ValueError('Invalid cell')
    for k in ['energy','e_wo_entrp','e_fr_energy']:
        data = trajectory.get(k)
        if data is None or len(data) != nsteps or not all(math.isfinite(v) for v in data):
            raise ValueError('Invalid/missing '+k)
    final = Structure(cells[-1],[Element.from_Z(z) for z in elements],coords[-1],
                      coords_are_cartesian=True)
    reference = Structure.from_dict(task['output']['structure'])
    matcher = StructureMatcher(**MATCHER,comparator=ElementComparator())
    geometry_match = matcher.fit(final,reference)
    target = float(task['output']['energy'])
    errors = {k:abs(float(trajectory[k][-1])-target)/len(elements)
              for k in ['energy','e_wo_entrp','e_fr_energy']}
    return {'canonical_task_id':identifier,'run_type':trajectory['run_type'],
            'steps':nsteps,'atoms':len(elements),'per_frame_geometry_shape_valid':True,
            'final_cell_max_abs_difference_angstrom':float(np.max(np.abs(final.lattice.matrix-reference.lattice.matrix))),
            'final_geometry_matches_task':bool(geometry_match),
            'energy_field_final_error_ev_atom':errors['energy'],
            'e_wo_entrp_final_error_ev_atom':errors['e_wo_entrp'],
            'e_fr_energy_final_error_ev_atom':errors['e_fr_energy'],
            'energy_field_matches_task':errors['energy']<=1e-6,
            'e_wo_entrp_matches_task':errors['e_wo_entrp']<=1e-6,
            'primary_energy_and_geometry_gate_pass':geometry_match and errors['energy']<=1e-6,
            'alternative_energy_and_geometry_gate_pass':geometry_match and errors['e_wo_entrp']<=1e-6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories',type=Path,required=True)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New output directory required')
    with gzip.open(args.tasks,'rt') as stream:
        docs = [json.loads(line) for line in stream]
    tasks = {task_id(d['task_id']):d for d in docs}
    if len(tasks) != len(docs):
        raise ValueError('Duplicate task docs')
    results=[]
    rejected=[]
    source_ids=set()
    snapshots=0
    with gzip.open(args.trajectories,'rt') as stream:
        for line in stream:
            trajectory = json.loads(line)
            identifier=task_id(trajectory['identifier'])
            if identifier in source_ids:
                raise ValueError('Duplicate trajectory identity')
            source_ids.add(identifier)
            snapshots+=trajectory['num_ionic_steps']
            if identifier not in tasks:
                raise ValueError('Missing corresponding task document')
            try:
                results.append(check_source(trajectory,tasks[identifier]))
            except (ValueError,TypeError,KeyError) as error:
                rejected.append({'canonical_task_id':identifier,'reason':str(error)})
    args.output.mkdir(parents=True)
    with (args.output/'PER_TASK_CHECKS.csv').open('w') as stream:
        if results:
            writer=csv.DictWriter(stream,fieldnames=list(results[0]))
            writer.writeheader();writer.writerows(results)
    (args.output/'REJECTED_INPUTS.json').write_text(json.dumps(rejected,indent=2)+'\n')
    result={'trajectories_checked':len(source_ids),
            'snapshots_checked':snapshots,
            'input_schema_valid_trajectories':len(results),
            'invalid_input_trajectories':len(rejected),
            'geometry_matches_task':sum(r['final_geometry_matches_task'] for r in results),
            'energy_matches_task':sum(r['energy_field_matches_task'] for r in results),
            'e_wo_entrp_matches_task':sum(r['e_wo_entrp_matches_task'] for r in results),
            'primary_energy_and_geometry_gate_pass':sum(r['primary_energy_and_geometry_gate_pass'] for r in results),
            'alternative_energy_and_geometry_gate_pass':sum(r['alternative_energy_and_geometry_gate_pass'] for r in results),
            'matcher':MATCHER,'model_forwards':0,
            'independence_audit':'not completed; source consistency gate must pass first'}
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
