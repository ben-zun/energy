"""Independent official core-metadata download; no models, labels or selection."""
import argparse
import gzip
import json
from pathlib import Path
from download_external_mp import to_plain

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--database',required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing overwrite')
    from mp_api.client import MPRester
    with MPRester((Path.home()/'.mp_api_key').read_text().strip(),use_document_model=False,
                  mute_progress_bars=True) as mpr:
        if mpr.get_database_version()!=args.database:raise ValueError('Version changed')
        print('Fetching official core metadata',flush=True)
        records=mpr.materials._search(deprecated=False,all_fields=False,chunk_size=1000,_sort_fields='material_id',
            fields=['material_id','created_at','last_updated','composition','nsites','deprecated'])
        ids=[str(row['material_id']) for row in records]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate IDs after stable sort; require serial raw API paging')
        with gzip.open(args.output,'wt') as stream:
            for row in records:stream.write(json.dumps(to_plain(row),separators=(',',':'))+'\n')
        if mpr.get_database_version()!=args.database:raise ValueError('Version changed during request')
        print(json.dumps({'core_records':len(records),'database':args.database,'model_forwards':0}),flush=True)

if __name__=='__main__':main()
