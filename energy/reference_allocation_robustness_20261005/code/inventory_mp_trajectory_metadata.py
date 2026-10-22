"""Inventory official public MP Delta tables without structures/model outputs.

This is a feasibility upper bound, not a completed task/geometry history audit.
"""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from download_external_mp import canonical_id

OPTIONS = {'AWS_SKIP_SIGNATURE': 'true', 'AWS_REGION': 'us-east-1',
           'timeout': '120s', 'max_retries': '3'}


def task_id(value):
    text = str(value).strip()
    return canonical_id(text if text.startswith('mp-') else 'mp-'+text)


def main():
    from deltalake import DeltaTable
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path, action='append', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new metadata-output directory')
    args.output.mkdir(parents=True)
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    historical = set()
    hashes = {}
    for path in args.history:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            for row in csv.DictReader(stream):
                historical.add(canonical_id(row['canonical_material_id']))
    versions = {'tasks': 12, 'trajectories': 7}
    source_info = {}
    cutoff = datetime(2019,4,2,23,59,59,999999)
    tasks = DeltaTable('s3://materialsproject-parsed/core/tasks/',
                       version=versions['tasks'], storage_options=OPTIONS)
    task_data = tasks.to_pyarrow_dataset().to_table(
        columns=['task_id','run_type','task_type','completed_at','nsites','batch_id'],
        filter=ds.field('run_type').isin(['GGA','GGA+U']) &
               (ds.field('completed_at') > cutoff) &
               (ds.field('task_type') == 'Structure Optimization'))
    pq.write_table(task_data, args.output/'temporal_relaxation_task_metadata.parquet')
    print('Temporal GGA/GGA+U relaxation task records', len(task_data), flush=True)
    mapped = {}
    for row in task_data.to_pylist():
        identifier = task_id(row['task_id'])
        if identifier in mapped:
            raise ValueError('Duplicate canonical task ID in source task table')
        mapped[identifier] = row
    traj = DeltaTable('s3://materialsproject-parsed/core/trajectories/',
                      version=versions['trajectories'], storage_options=OPTIONS)
    trajectory_data = traj.to_pyarrow_dataset().to_table(
        columns=['identifier','run_type','task_type','num_ionic_steps','has_full_output'],
        filter=ds.field('run_type').isin(['GGA','GGA+U']) &
               (ds.field('task_type') == 'Structure Optimization') &
               (ds.field('num_ionic_steps') >= 2))
    pq.write_table(trajectory_data, args.output/'relaxation_trajectory_metadata.parquet')
    joined, unmatched, disagreements, seen = [], 0, [], set()
    for row in trajectory_data.to_pylist():
        identifier = task_id(row['identifier'])
        key = (identifier,row['run_type'],row['task_type'])
        if key in seen:
            raise ValueError('Duplicate canonical trajectory key')
        seen.add(key)
        if identifier not in mapped:
            unmatched += 1
            continue
        task = mapped[identifier]
        if task['run_type'] != row['run_type'] or task['task_type'] != row['task_type']:
            disagreements.append(identifier)
            continue
        if identifier in historical:
            continue
        joined.append(dict(task,canonical_task_id=identifier,
                           trajectory_identifier=row['identifier'],
                           num_ionic_steps=row['num_ionic_steps'],
                           has_full_output=row['has_full_output']))
    columns = ['canonical_task_id','task_id','trajectory_identifier','run_type',
               'task_type','completed_at','nsites','batch_id','num_ionic_steps','has_full_output']
    with (args.output/'candidate_task_upper_bound.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(joined)
    for name, table in [('tasks',tasks),('trajectories',traj)]:
        source_info[name] = {'uri':table.table_uri,'delta_version':table.version(),
                             'file_count':len(table.file_uris())}
    result = {'status':'metadata_upper_bound_only_not_independence_certification',
              'source_tables':source_info,'cutoff_utc':cutoff.isoformat()+'Z',
              'temporal_gga_relaxation_tasks':len(mapped),
              'gga_relaxation_trajectories_with_at_least_2_steps':len(trajectory_data),
              'trajectory_rows_without_temporal_task_match':unmatched,
              'task_trajectory_run_type_disagreements':disagreements,
              'historical_mp_identifier_union':len(historical),
              'remaining_task_upper_bound':len(joined),
              'remaining_raw_snapshot_upper_bound':sum(r['num_ionic_steps'] for r in joined),
              'history_input_sha256':hashes,
              'unresolved_gates':['explicit historical task fields and ID namespaces',
                                  'historical geometry/snapshot audit',
                                  'per-step geometry/energy/input checks',
                                  'material/task provenance mapping'],
              'model_forwards':0,'new_training':False}
    (args.output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
