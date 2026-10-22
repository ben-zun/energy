"""Fetch corresponding task docs; reads private API key in place, never exports it."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import requests
from download_external_mp import canonical_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-tasks',type=int,default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New output path required')
    rows = list(csv.DictReader(args.manifest.open()))
    if len(rows) != args.expected_tasks or len({r['canonical_task_id'] for r in rows}) != args.expected_tasks:
        raise ValueError('Unique frozen task manifest required')
    session = requests.Session()
    session.headers['X-API-KEY'] = (Path.home()/'.mp_api_key').read_text().strip()
    docs = []
    for start in range(0,len(rows),25):
        ids = [r['canonical_task_id'] for r in rows[start:start+25]]
        response = session.get('https://api.materialsproject.org/materials/tasks/',
            params={'task_ids':','.join(ids),'_limit':100,
                    '_fields':'task_id,run_type,task_type,completed_at,last_updated,nsites,input,output'},
            timeout=120)
        response.raise_for_status()
        batch = response.json()['data']
        found = {canonical_id('mp-'+d['task_id']) if not d['task_id'].startswith('mp-')
                 else canonical_id(d['task_id']) for d in batch}
        if found != set(ids) or len(batch) != len(ids):
            raise ValueError('Task document identities incomplete or duplicate')
        docs.extend(batch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'wt') as stream:
        for d in docs:
            stream.write(json.dumps(d,allow_nan=False)+'\n')
    print(f'{len(rows)} corresponding official task documents saved',flush=True)


if __name__ == '__main__':
    main()
