"""Map official MP tasks to core material provenance; no exclusion by material."""
import argparse,csv,gzip,json
from pathlib import Path
import requests
from inventory_mp_trajectory_metadata import task_id
from download_external_mp import canonical_id

def links(docs,requested):
 out={i:[] for i in requested}
 for doc in docs:
  material=canonical_id(doc['material_id'])
  for i in doc.get('task_ids') or []:
   identifier=task_id(i)
   if identifier in out and material not in out[identifier]:out[identifier].append(material)
 return {k:sorted(v) for k,v in out.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deprecated',choices=['true','false'],default='false');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 rows=list(csv.DictReader(a.manifest.open()));ids=[r['canonical_task_id'] for r in rows];assert len(ids)==len(set(ids))
 native=[r['task_id'] for r in rows]
 if any(task_id(raw)!=canonical for raw,canonical in zip(native,ids)):raise ValueError('Native/canonical task ID mismatch')
 session=requests.Session();session.headers['X-API-KEY']=(Path.home()/'.mp_api_key').read_text().strip();docs=[]
 for start in range(0,len(ids),25):
  batch=native[start:start+25];batch_ids=set(ids[start:start+25]);skip=0
  while True:
   r=session.get('https://api.materialsproject.org/materials/core/',params={'task_ids':','.join(batch),'_fields':'material_id,task_ids,deprecated','_limit':1000,'_skip':skip,'deprecated':a.deprecated},timeout=120);r.raise_for_status();data=r.json();got=data['data']
   for doc in got:
    if not batch_ids&{task_id(i) for i in doc.get('task_ids',[])}:raise ValueError('Core response lacks requested task: query semantics invalid')
   docs.extend(got);skip+=len(got)
   total=data['meta']['total_doc']
   if skip>=total:break
   if not got:raise ValueError('Incomplete core pagination')
 a.output.write_text(json.dumps({'task_material_links':links(docs,ids),'docs':docs,'deprecated_query':a.deprecated,'query_native_task_ids':native,'rule':'provenance only; no material automatic exclusion'},indent=2)+'\n');print(f'{len(ids)} task provenance queries saved',flush=True)
if __name__=='__main__':main()
