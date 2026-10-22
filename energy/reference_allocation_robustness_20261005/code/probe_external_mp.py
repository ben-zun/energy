"""Probe installed official MP client/schema. Never display authentication values."""
import inspect
import json
from pathlib import Path
from importlib.metadata import version

def main():
    from mp_api.client import MPRester
    key_path=Path.home()/'.mp_api_key'
    if not key_path.exists():raise RuntimeError('MP key file unavailable')
    with MPRester(key_path.read_text().strip(),mute_progress_bars=True) as mpr:
        print(json.dumps({'database_version':mpr.get_database_version(),
                          'mp_api_version':version('mp-api'),
                          'core_search_signature':str(inspect.signature(mpr.materials.search)),
                          'thermo_search_signature':str(inspect.signature(mpr.materials.thermo.search)),
                          'core_fields':mpr.materials.available_fields,
                          'thermo_fields':mpr.materials.thermo.available_fields}))
        print('THERMO_SEARCH_SOURCE')
        print(inspect.getsource(mpr.materials.thermo.search))
        core=mpr.materials.search(material_ids=['mp-149'],fields=['material_id','created_at','last_updated','deprecated'],all_fields=False)
        print('CORE_PROBE',json.dumps([d.model_dump(mode='json') for d in core]))

if __name__=='__main__':main()
