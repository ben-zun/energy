"""Serial official API pages with per-page uniqueness and completeness gates."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from download_external_mp import canonical_id

FIELDS='material_id,created_at,last_updated,composition,nsites,deprecated'

def validate_page(body,seen,total=None):
    expected=int(body['meta']['total_doc'])
    if total is not None and expected!=total:raise ValueError('API total changed while paging')
    records=body['data']
    ids=[canonical_id(r['material_id']) for r in records]
    if len(set(ids))!=len(ids) or set(ids)&seen:raise ValueError('Duplicate materials in API pages')
    if any(r.get('deprecated') is not False for r in records):raise ValueError('Deprecated/missing status')
    return expected,records,set(ids)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--database',required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing overwrite')
    import requests
    from mp_api.client import MPRester
    key=(Path.home()/'.mp_api_key').read_text().strip()
    with MPRester(key,mute_progress_bars=True) as mpr:
        if mpr.get_database_version()!=args.database:raise ValueError('Version changed before download')
    session=requests.Session();session.headers['X-API-KEY']=key
    seen=set();total=None;offset=0;pages=[]
    with gzip.open(args.output,'wt') as stream:
        while total is None or offset<total:
            response=session.get('https://api.materialsproject.org/materials/core/',
                params={'deprecated':False,'_fields':FIELDS,'_limit':1000,'_skip':offset},timeout=60)
            response.raise_for_status();body=response.json()
            total,records,ids=validate_page(body,seen,total)
            if not records:raise ValueError('Empty page before completeness')
            seen|=ids
            pages.append({'offset':offset,'rows':len(records),'total_doc':total,
                          'response_sha256':hashlib.sha256(response.content).hexdigest(),
                          'api_version':body['meta'].get('api_version')})
            for row in records:stream.write(json.dumps(row,separators=(',',':'))+'\n')
            offset+=len(records)
            if offset%10000==0 or offset==total:print(f'Serial core {offset}/{total}, unique {len(seen)}',flush=True)
    if len(seen)!=total or offset!=total:raise ValueError('API download incomplete')
    with MPRester(key,mute_progress_bars=True) as mpr:
        if mpr.get_database_version()!=args.database:raise ValueError('Version changed during download')
    audit={'status':'PASS','database_version':args.database,'expected_documents':total,
           'unique_canonical_material_ids':len(seen),'pages':pages,'paging':'serial raw official /materials/core/ _skip/_limit',
           'file_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'model_forwards':0}
    args.output.with_suffix('.AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='pages'}),flush=True)

if __name__=='__main__':main()
