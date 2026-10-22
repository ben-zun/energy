"""Official API pagination probe, no credentials in output."""
import json
from pathlib import Path

def main():
    import requests
    session=requests.Session();session.headers['X-API-KEY']=(Path.home()/'.mp_api_key').read_text().strip()
    for extra in [{'_skip':0},{'_skip':3},{'_skip':0,'_sort_fields':'material_id'},
                  {'_skip':0,'sort_fields':'material_id'}]:
        params={'deprecated':False,'_limit':3,'_fields':'material_id,created_at,last_updated,composition,nsites,deprecated',**extra}
        response=session.get('https://api.materialsproject.org/materials/core/',params=params,timeout=60)
        body=response.json()
        print(json.dumps({'query':extra,'http_status':response.status_code,'meta':body.get('meta'),
                          'data':body.get('data'),'error':body.get('detail')}),flush=True)

if __name__=='__main__':main()
