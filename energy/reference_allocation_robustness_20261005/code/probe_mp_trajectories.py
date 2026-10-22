"""Read-only official API capability probe. Authentication never leaves server."""
import argparse
import gzip
import json
from pathlib import Path
import requests


def request(session, suffix, params):
    response = session.get('https://api.materialsproject.org/'+suffix,
                           params=params, timeout=120)
    # Do not serialize request headers/authentication.
    return {'http_status': response.status_code,
            'body': response.json() if 'json' in response.headers.get('Content-Type','')
            else {'non_json_response': True}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new probe path')
    session = requests.Session()
    session.headers['X-API-KEY'] = (Path.home()/'.mp_api_key').read_text().strip()
    with gzip.open(args.candidates,'rt') as stream:
        ids = [json.loads(line)['api_material_id'] for line in stream][:3]
    result = {'probe_material_ids': ids, 'queries': {}}
    result['queries']['core_task_links'] = request(session, 'materials/core/', {
        'material_ids': ','.join(ids), '_fields': 'material_id,task_ids,origins', '_limit': 10})
    result['queries']['tasks_sample'] = request(session, 'materials/tasks/', {
        '_fields': 'task_id,run_type,task_type,completed_at,last_updated,nsites', '_limit': 3})
    links = result['queries']['core_task_links']['body'].get('data',[])
    tasks = []
    for doc in links:
        tasks.extend(doc.get('task_ids',[]))
    result['linked_task_ids'] = tasks
    if tasks:
        result['queries']['linked_tasks'] = request(session, 'materials/tasks/', {
            'task_ids': ','.join(map(str,tasks)),
            '_fields': 'task_id,run_type,task_type,completed_at,last_updated,nsites,input,output',
            '_limit': 100})
        result['queries']['trajectory_sample'] = request(session, 'materials/tasks/trajectory/', {
            'task_ids': str(tasks[0]), '_limit': 1})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'queries': {k: {'http_status':v['http_status'],
           'data_records':len(v['body'].get('data',[])),
           'body_keys':list(v['body'])} for k,v in result['queries'].items()},
           'linked_tasks_n':len(tasks)}))


if __name__ == '__main__':
    main()
