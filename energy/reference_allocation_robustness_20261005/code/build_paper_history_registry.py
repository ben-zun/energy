"""Read frozen scoped historical inputs, including archive members and extxyz.

Never execute or extract archive members. File failures prevent certification.
Only composition buckets requested by a frozen metadata candidate set are
exported; IDs are audited without any model errors or prediction selection.
"""
import argparse,ast,csv,gzip,hashlib,io,json,re,tarfile,zipfile
from pathlib import Path
from collections import Counter
from inventory_mp_trajectory_metadata import task_id
from download_external_mp import canonical_id,structure_counts
TASK_FIELDS={'task_id','task_ids','mp_task_id','vasp_task_id'}
MATERIAL_FIELDS={'material_id','material_ids','mp_id','mp_material_id'}
ARCHIVES=('.zip','.tar','.tar.gz','.tgz')
DATA=('.csv','.tsv','.json','.jsonl','.xyz','.extxyz','.cif','.vasp','.poscar')

def id_values(v):
 if hasattr(v,'tolist'):v=v.tolist()
 if isinstance(v,list):
  for x in v:yield from id_values(x)
 elif isinstance(v,dict):
  if 'identifier' in v:yield from id_values(v['identifier'])
  else:
   for x in v.values():yield from id_values(x)
 elif isinstance(v,(str,int)) and not isinstance(v,bool):
  try:yield task_id(v)
  except ValueError:pass

def walk_json(v):
 if isinstance(v,dict):
  if 'lattice' in v and 'sites' in v:yield ('structure',v);return
  for k,x in v.items():
   if k.lower() in TASK_FIELDS:
    for i in id_values(x):yield ('task',i)
   elif k.lower() in MATERIAL_FIELDS:
    for i in id_values(x):yield ('material',i)
   yield from walk_json(x)
 elif isinstance(v,list):
  for x in v:yield from walk_json(x)

def supported_name(name):
 plain=name.lower().removesuffix('.gz')
 return name.lower().endswith(ARCHIVES) or plain.endswith(DATA) or Path(name).name.upper() in {'POSCAR','CONTCAR'}

def parse_stream(stream,name):
 plain=name.lower()
 if plain.endswith('.gz') and not plain.endswith('.tar.gz'):
  with gzip.GzipFile(fileobj=stream) as s:yield from parse_stream(s,name[:-3])
  return
 if plain.endswith(ARCHIVES):
  if plain.endswith('.zip'):
   with zipfile.ZipFile(stream) as archive:
    for member in archive.infolist():
     if member.is_dir():continue
     if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink requires resolution: '+member.filename)
     yield ('member',member.filename)
     if not supported_name(member.filename):yield ('non_data',member.filename);continue
     with archive.open(member) as s:yield from parse_stream(io.BytesIO(s.read()),member.filename)
  else:
   with tarfile.open(fileobj=stream,mode='r:*') as archive:
    for member in archive:
     if member.isdir():continue
     if member.issym() or member.islnk():
      yield ('archive_link',{'name':member.name,'target':member.linkname});continue
     if not member.isfile():continue
     yield ('member',member.name)
     if not supported_name(member.name):yield ('non_data',member.name);continue
     with archive.extractfile(member) as s:yield from parse_stream(io.BytesIO(s.read()),member.name)
  return
 if not plain.endswith(DATA) and Path(name).name.upper() not in {'POSCAR','CONTCAR'}:
  yield ('non_data',name);return
 text=io.TextIOWrapper(stream,encoding='utf-8-sig')
 if plain.endswith(('.csv','.tsv')):
  for row in csv.DictReader(text,delimiter='\t' if plain.endswith('.tsv') else ','):
   for k,v in row.items():
    if not k or not v:continue
    if k.lower() in TASK_FIELDS:
     try:val=json.loads(v)
     except (ValueError,TypeError):val=v
     for i in id_values(val):yield ('task',i)
    elif k.lower() in MATERIAL_FIELDS:
     for i in id_values(v):yield ('material',i)
    elif v.startswith(('{','[')) and any(w in k.lower() for w in ('structure','atoms','entry')):
     try:decoded=json.loads(v)
     except json.JSONDecodeError:decoded=ast.literal_eval(v)
     yield from walk_json(decoded)
  return
 if plain.endswith('.jsonl'):
  for line in text:
   if line.strip():yield from walk_json(json.loads(line))
  return
 if plain.endswith('.json'):
  yield from walk_json(json.load(text));return
 from pymatgen.core import Structure
 if plain.endswith(('.xyz','.extxyz')):
  from ase.io import iread
  from pymatgen.io.ase import AseAtomsAdaptor
  for atoms in iread(text,format='extxyz',index=':'):
   yield from walk_json(atoms.info)
   if abs(__import__('numpy').linalg.det(atoms.cell.array))<1e-12:
    raise ValueError('Historical XYZ has no valid periodic cell; geometry unresolved')
   yield ('structure',AseAtomsAdaptor.get_structure(atoms).as_dict())
  return
 raw=text.read()
 if plain.endswith('.cif'):
  import gemmi
  block=gemmi.cif.read_string(raw).sole_block()
  for field in TASK_FIELDS:
   value=block.find_value('_'+field)
   if value:
    for identifier in id_values(value.strip("'\"")):yield ('task',identifier)
  for field in MATERIAL_FIELDS:
   value=block.find_value('_'+field)
   if value:
    for identifier in id_values(value.strip("'\"")):yield ('material',identifier)
  number=block.find_value('_symmetry_Int_Tables_number') or block.find_value('_space_group_IT_number')
  small=gemmi.make_small_structure_from_block(block)
  if number=='1' and all(abs(site.occ-1)<1e-12 for site in small.sites):
   # P1 atomic positions are the full cell; all other symmetry uses pymatgen.
   mat=small.cell.orth.mat.tolist();matrix=[list(row) for row in zip(*mat)]
   structure={'lattice':{'matrix':matrix},'sites':[{'species':[{'element':site.element.name,'occu':site.occ}],'abc':[site.fract.x,site.fract.y,site.fract.z],'properties':{}} for site in small.sites]}
   yield ('structure',structure);return
 fmt='cif' if plain.endswith('.cif') else 'poscar'
 yield ('structure',Structure.from_str(raw,fmt=fmt).as_dict())

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--keys',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 opener=gzip.open if a.inventory.suffix=='.gz' else open
 with opener(a.inventory,'rt') as f:inventory=json.load(f)
 requested=set(json.loads(a.keys.read_text()));counts=Counter();tasks={};materials={};failures=[];seen={};content_seen=set();links=[]
 csv.field_size_limit(256*1024*1024)
 with gzip.open(a.output/'STRUCTURE_BUCKETS.jsonl.gz','wt') as gs, (a.output/'FILE_AUDIT.jsonl').open('w') as log:
  for f in inventory['files']:
   path=Path(f['path']);plain=path.name.removesuffix('.gz').lower()
   if not (path.name.lower().endswith(ARCHIVES) or plain.endswith(DATA) or path.name.upper() in {'POSCAR','CONTCAR'}):
    counts['non_data_files']+=1;continue
   result={'path':str(path)}
   try:
    actual=path.resolve(strict=True)
    if path.is_symlink():links.append({'path':str(path),'resolved':str(actual)})
    st=actual.stat()
    if not path.is_symlink() and (st.st_size!=f['bytes'] or st.st_mtime_ns!=f['mtime_ns']):raise ValueError('Frozen history file changed')
    if str(actual) in seen:
     result.update(status='resolved_same_source',sha256=seen[str(actual)]);log.write(json.dumps(result)+'\n');continue
    digest=hashlib.sha256()
    with actual.open('rb') as s:
     for b in iter(lambda:s.read(4*1024*1024),b''):digest.update(b)
    result['sha256']=digest.hexdigest();seen[str(actual)]=result['sha256']; events=Counter()
    if (actual.name.removesuffix('.gz').split('.')[-1],result['sha256']) in content_seen:
     result['status']='identical_content_already_parsed';log.write(json.dumps(result)+'\n');counts['identical_content_files']+=1;continue
    with actual.open('rb') as s:
     for kind,value in parse_stream(s,actual.name):
      events[kind]+=1
      if kind=='task':tasks.setdefault(value,set()).add(str(path))
      elif kind=='material':materials.setdefault(value,set()).add(str(path))
      elif kind=='archive_link':raise ValueError('Unresolved archive link '+json.dumps(value))
      elif kind=='structure':
       key='|'.join(map(str,structure_counts(value)))
       if key in requested:
        geometry={'lattice':{'matrix':[[float(x) for x in row] for row in value['lattice']['matrix']]},'sites':[{'species':[{'element':sp['element'],'occu':float(sp.get('occu',1))} for sp in site['species']],'abc':[float(x) for x in site['abc']],'properties':{}} for site in value['sites']]}
        gs.write(json.dumps({'source':str(path),'composition_key':key,'structure':geometry},allow_nan=False)+'\n');counts['exported_structures']+=1
    result.update(status='parsed',events=dict(events));counts.update(events);counts['parsed_files']+=1;content_seen.add((actual.name.removesuffix('.gz').split('.')[-1],result['sha256']))
   except Exception as e:
    result.update(status='failed',error_type=type(e).__name__,message=str(e)[:300]);failures.append(result)
   log.write(json.dumps(result)+'\n');log.flush()
   if counts['parsed_files'] and counts['parsed_files']%100==0:print(json.dumps({'parsed_files':counts['parsed_files'],'failures':len(failures)}),flush=True)
 for filename,data in [('TASK_REGISTRY_A.json',tasks),('MATERIAL_PROVENANCE_REGISTRY.json',materials)]:
  (a.output/filename).write_text(json.dumps({k:sorted(v) for k,v in data.items()},indent=2)+'\n')
 audit={'status':'history registry parsed' if not failures else 'unresolved history inputs','history_scope_complete':not failures,'counts':dict(counts),'failures':failures,'resolved_symlinks':links,'task_ids':len(tasks),'material_ids':len(materials),'inventory_sha256':hashlib.sha256(a.inventory.read_bytes()).hexdigest(),'keys_sha256':hashlib.sha256(a.keys.read_bytes()).hexdigest()}
 (a.output/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in {'failures','resolved_symlinks'}}),flush=True)
if __name__=='__main__':main()
