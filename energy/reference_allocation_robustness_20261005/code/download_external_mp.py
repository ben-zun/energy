"""Download a temporal GGA/GGA+U candidate pool; no models or final cohort lock.

Official MP API only. All policy constants are frozen in a supplied JSON file.
Structures come from the ThermoDoc entry matching its corrected energy, never
from a summary document. Missing/ambiguous provenance is rejected.
"""
import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from functools import lru_cache

ELEMENTS=('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
 'Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te '
 'I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt '
 'Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf '
 'Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
PATTERN=re.compile(rb'mp-(?:[0-9]+|[a-z]+)(?![A-Za-z0-9])')
LEFT_BOUNDARY=set(b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-')

def to_plain(value):
    """Current MP client may return decoded Monty entries despite raw-doc mode."""
    if isinstance(value,datetime):return value.isoformat()
    if isinstance(value,dict):return {str(k):to_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [to_plain(v) for v in value]
    if hasattr(value,'as_dict'):return to_plain(value.as_dict())
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    raise TypeError(f'Unexpected API value type: {type(value).__name__}')

def extracted_ids(chunk,before=None):
    # Keep a literal-prefix regex so large numeric XYZ/features files scan
    # efficiently. Check the left boundary separately to avoid mb-mp aliases.
    return {canonical_id(m.group().decode()) for m in PATTERN.finditer(chunk)
            if (before is None or m.start()<before) and
               (m.start()==0 or chunk[m.start()-1] not in LEFT_BOUNDARY)}

@lru_cache(maxsize=500000)
def canonical_id(value):
    token=str(value).removeprefix('mp-')
    if not str(value).startswith('mp-') or not re.fullmatch(r'\d+|[a-z]+',token):
        raise ValueError('Invalid MP identifier')
    number=int(token) if token.isdigit() else 0
    if not token.isdigit():
        for char in token:number=26*number+ord(char)-ord('a')
    return f'mp-{number}'

def after_cutoff(value,cutoff):
    date=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
    threshold=datetime.fromisoformat(cutoff.replace('Z','+00:00'))
    return date > threshold

def structure_counts(structure):
    counts=[0]*118
    for site in structure['sites']:
        species=site['species']
        if len(species)!=1 or abs(float(species[0].get('occu',1))-1)>1e-12:
            raise ValueError('Disordered/partial occupancy')
        counts[ELEMENTS.index(species[0]['element'])]+=1
    if sum(counts)==0:raise ValueError('Empty structure')
    return counts

def select_entry(doc,tolerance=1e-8):
    if doc.get('thermo_type')!='GGA_GGA+U':raise ValueError('Mixed or wrong thermo type')
    if not math.isfinite(float(doc['formation_energy_per_atom'])):
        raise ValueError('Nonfinite formation energy')
    target=float(doc['energy_per_atom']);selected=[]
    for name,entry in doc['entries'].items():
        run_type=entry.get('parameters',{}).get('run_type',name)
        if run_type not in ('GGA','GGA+U','GGA_U'):continue
        structure=entry.get('structure')
        if not structure:continue
        counts=structure_counts(structure)
        corrected=(float(entry['energy'])+float(entry.get('correction',0)))/sum(counts)
        if abs(corrected-target)<=tolerance:selected.append((name,entry,counts,corrected))
    if len(selected)!=1:raise ValueError(f'Expected one label-matched entry, got {len(selected)}')
    return selected[0]

def scan_history(root,output,required=None):
    skip={'.git','node_modules','__pycache__','external_repos','external_packages'}
    used=set();rows=[];unreadable=[];scanned=0;total_bytes=0
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in skip and not (Path(folder)/d).resolve().is_relative_to(output.resolve())
                 and not (Path(folder).resolve()==output.parent.resolve() and d.startswith('external_mp_download_'))]
        for name in files:
            path=Path(folder)/name
            if path.resolve().is_relative_to(output.resolve()):continue
            # Every historical CSV/TSV/XYZ, and JSON manifests/registries/results.
            plain=name.removesuffix('.gz').lower()
            tabular=plain.endswith(('.csv','.tsv','.xyz','.extxyz','.jsonl'))
            historical_json=plain.endswith('.json') and any(w in str(path).lower() for w in
                ('manifest','registry','prediction','diagnostic','split','results','reports','audit'))
            if not (tabular or historical_json):continue
            digest=hashlib.sha256();ids=set();size=0;tail=b''
            try:
                opener=gzip.open if name.endswith('.gz') else open
                with opener(path,'rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(block);size+=len(block)
                        chunk=tail+block
                        if b'mp-' in chunk:
                            ids.update(extracted_ids(chunk,len(chunk)-256))
                        tail=chunk[-256:]
                    ids.update(extracted_ids(tail))
            except (OSError,EOFError,ValueError) as exc:
                unreadable.append({'path':str(path),'error_type':type(exc).__name__});continue
            used.update(ids)
            scanned+=1;total_bytes+=size
            if scanned % 250 == 0:
                print(f'History scanned {scanned} files / {total_bytes/1e9:.2f} GB / {len(used)} IDs',flush=True)
            if ids:rows.append(dict(path=str(path),unique_mp_ids=len(ids),uncompressed_bytes=size,
                                    sha256_uncompressed=digest.hexdigest()))
            if required and required <= used:
                print('Every temporal core ID is already covered by historical records; negative coverage proof complete',flush=True)
                return used,rows,unreadable
    return used,rows,unreadable

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--history-root',type=Path,default=Path('/home/yxzhang/energy'))
    parser.add_argument('--history-cache',type=Path)
    parser.add_argument('--core-cache',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('New download directory required')
    protocol=json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True)
    (args.output/'EXTERNAL_MP_PROTOCOL_v1.json').write_bytes(args.protocol.read_bytes())
    print('Scanning existing historical IDs without prediction values',flush=True)
    required=None
    if args.core_cache:
        with gzip.open(args.core_cache,'rt') as stream:
            required={canonical_id(d['material_id']) for line in stream
                      for d in [json.loads(line)] if d.get('created_at') and
                      after_cutoff(d['created_at'],protocol['cutoff_utc'])}
    if args.history_cache:
        cached=json.loads((args.history_cache/'HISTORY_SCAN.json').read_text())
        with (args.history_cache/'historical_mp_ids.csv').open() as stream:
            used={row['canonical_material_id'] for row in csv.DictReader(stream)}
        if len(used)!=cached['unique_mp_ids']:raise ValueError('History cache count mismatch')
        if (args.history_cache/'EXTERNAL_MP_PROTOCOL_v1.json').read_bytes()!=args.protocol.read_bytes():
            raise ValueError('History cache policy mismatch')
        files=cached['files_with_mp_ids'];unreadable=cached['unreadable']
    else:
        used,files,unreadable=scan_history(args.history_root,args.output,required=required)
    with (args.output/'historical_mp_ids.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['canonical_material_id']);w.writerows([x] for x in sorted(used))
    (args.output/'HISTORY_SCAN.json').write_text(json.dumps({'files_with_mp_ids':files,
        'unique_mp_ids':len(used),'unreadable':unreadable,'excluded_directories':['.git','node_modules','__pycache__','external_repos','external_packages','current acquisition runs external_mp_download_*'],
        'scope':'All CSV/TSV/XYZ/EXTXYZ/JSONL; JSON in manifests/registries/predictions/diagnostics/splits/results/reports/audits; gzip supported'},indent=2)+'\n')
    print(f'History inventory: {len(used)} IDs; {len(unreadable)} unreadable files',flush=True)
    from mp_api.client import MPRester
    from emmet.core.mpid import AlphaID
    from importlib.metadata import version
    key_path=Path.home()/'.mp_api_key'
    if not key_path.exists():raise RuntimeError('MP key file unavailable')
    with MPRester(key_path.read_text().strip(),use_document_model=False,
                  mute_progress_bars=True) as mpr:
        database=mpr.get_database_version()
        if database!=protocol['mp_database_version']:raise ValueError('Database version changed before download')
        print(f'Downloading core time metadata, database {database}',flush=True)
        if args.core_cache:
            with gzip.open(args.core_cache,'rt') as stream:core=[json.loads(line) for line in stream]
        else:
            core=[to_plain(d) for d in mpr.materials.search(deprecated=False,all_fields=False,chunk_size=1000,
                fields=['material_id','created_at','last_updated','composition','nsites','deprecated'])]
        core_ids=[canonical_id(d['material_id']) for d in core]
        if len(set(core_ids))!=len(core_ids):
            raise ValueError('Duplicate core IDs: use verified serial API snapshot')
        with gzip.open(args.output/'core_metadata.jsonl.gz','wt') as stream:
            for doc in core:stream.write(json.dumps(doc,separators=(',',':'))+'\n')
        eligible=[];rejected_time=0;missing_time=0;history_overlap=0
        for doc in core:
            sid=canonical_id(doc['material_id'])
            if int(AlphaID(doc['material_id']))!=int(sid[3:]):raise ValueError('ID normalization mismatch')
            doc['canonical_material_id']=sid
            if not doc.get('created_at'):missing_time+=1;continue
            if not after_cutoff(doc['created_at'],protocol['cutoff_utc']):rejected_time+=1;continue
            if sid in used:history_overlap+=1;continue
            eligible.append(doc)
        eligible.sort(key=lambda d:d['canonical_material_id'])
        by_id={d['canonical_material_id']:d for d in eligible}
        print(f'Core {len(core)}; temporal + history-excluded {len(eligible)}',flush=True)
        rejected=[];rows=[]
        with (gzip.open(args.output/'candidate_records.jsonl.gz','wt') as accepted,
              gzip.open(args.output/'raw_gga_thermo_docs.jsonl.gz','wt') as rawout):
            for start in range(0,len(eligible),200):
                ids=[d['material_id'] for d in eligible[start:start+200]]
                docs=mpr.materials.thermo.search(material_ids=ids,thermo_types=['GGA_GGA+U'],
                    all_fields=False,fields=['material_id','thermo_type','thermo_id','formation_energy_per_atom',
                    'energy_per_atom','uncorrected_energy_per_atom','energy_type','entries','last_updated','deprecated'],
                    chunk_size=200)
                for doc in docs:
                    doc=to_plain(doc)
                    rawout.write(json.dumps(doc,separators=(',',':'))+'\n')
                    sid=canonical_id(doc['material_id'])
                    if sid not in by_id:raise ValueError('Thermo API returned unrequested ID')
                    try:
                        if doc.get('deprecated'):raise ValueError('Deprecated thermo entry')
                        name,entry,counts,energy=select_entry(doc)
                        entry_mid=entry.get('data',{}).get('material_id')
                        if entry_mid and canonical_id(entry_mid)!=sid:raise ValueError('Entry material identity mismatch')
                        record={'material_id':sid,'api_material_id':doc['material_id'],
                            'created_at':by_id[sid]['created_at'],'last_updated':by_id[sid]['last_updated'],
                            'thermo_last_updated':doc.get('last_updated'),'thermo_type':doc['thermo_type'],
                            'formation_energy_per_atom':float(doc['formation_energy_per_atom']),
                            'structure':entry['structure'],'entry_id':entry.get('entry_id'),
                            'entry_run_type':name,'entry_parameters':entry.get('parameters'),
                            'entry_data':entry.get('data'),'energy_adjustments':entry.get('energy_adjustments'),
                            'entry_correction_ev':entry.get('correction',0),'nsites':sum(counts),
                            'composition_key':'|'.join(map(str,counts)),
                            'label_geometry_energy_difference_ev_atom':abs(energy-float(doc['energy_per_atom']))}
                        accepted.write(json.dumps(record,separators=(',',':'))+'\n')
                        rows.append({k:v for k,v in record.items() if k not in ('structure','entry_parameters','entry_data','energy_adjustments')})
                    except (ValueError,KeyError,TypeError) as exc:
                        rejected.append({'material_id':sid,'reason':str(exc)})
                accepted.flush();rawout.flush()
                print(f'Thermo queried {min(start+200,len(eligible))}/{len(eligible)}; accepted {len(rows)}',flush=True)
        if mpr.get_database_version()!=database:raise ValueError('Database changed during download')
    with (args.output/'candidate_manifest.csv').open('w',newline='') as stream:
        columns=list(rows[0]) if rows else ['material_id','api_material_id','created_at','last_updated',
            'thermo_type','formation_energy_per_atom','nsites','composition_key','entry_id','entry_run_type',
            'label_geometry_energy_difference_ev_atom']
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(rows)
    groups={}
    for row in rows:groups[row['composition_key']]=groups.get(row['composition_key'],0)+1
    audit={'status':'DOWNLOADED_CANDIDATE_POOL_NOT_COHORT_LOCKED','database_version':database,
        'mp_api_version':version('mp-api'),'download_completed_utc':datetime.now(timezone.utc).isoformat(),
        'core_records':len(core),'missing_created_at':missing_time,'excluded_created_at':rejected_time,
        'temporal_history_overlap':history_overlap,'requested_temporal_unseen_ids':len(eligible),
        'accepted_gga_ggau_records':len(rows),'pairable_full_count_groups_before_geometry_audit':sum(n>=2 for n in groups.values()),
        'rejected_thermo_records':rejected,'history_scan_unreadable_n':len(unreadable),
        'geometry_duplicate_audit':'pending; pool is not certified eligible','model_forwards':0,
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'DOWNLOAD_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='rejected_thermo_records'}),flush=True)

if __name__=='__main__':main()
