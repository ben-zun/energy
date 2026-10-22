from pathlib import Path
import ast,csv,gzip,hashlib,io,json,os,re,shutil,subprocess,sys,time
ROOT=Path('/home/yxzhang/energy')
OUT=Path('/home/yxzhang/releases/energy-reference-alignment-20261016')
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'energy').mkdir();(OUT/'provenance').mkdir();(OUT/'tools').mkdir();(OUT/'environment').mkdir();(OUT/'manuscript').mkdir()
MAIN='experiment_reference_allocation_final_20261004'
ROB='reference_allocation_robustness_20261005'
E12='experiment12_20260929_alignment';E14='experiment14_alignment_controls_20260929';E17='experiment17_eqv3_physical_matched_20261001'
selected={};missing=[];exclusions=[]
def add(p,reason):
 p=Path(p)
 if not p.is_absolute():p=ROOT/p
 if not p.is_file():missing.append({'path':str(p),'reason':reason});return
 selected[str(p)]=reason

def tree(rel,reason,weights=False,skip=()):
 base=ROOT/rel
 if not base.is_dir():missing.append({'path':str(base),'reason':reason});return
 for d,dirs,files in os.walk(base):
  dirs[:]=[x for x in dirs if x not in ('__pycache__','.git','.cache','node_modules') and x not in skip]
  for f in files:
   p=Path(d)/f
   if p.suffix in ('.pyc','.tmp','.partial') or f.startswith('.env'):continue
   if not weights and p.suffix in ('.pt','.pth','.ckpt'):continue
   add(p,reason)

tree(MAIN,'final publication experiment',skip=('smoke','local_validation','upstream_snapshots','sevennet_final_eval'))
tree(E12,'SevenNet main training code, histories, predictions',skip=('smoke',))
# Old EqV3 decoders and earlier pilot r2SCAN cohorts are deliberately not mixed into current evidence.
for sub in ['code','runs','r2scan10k_full','unified']:
 tree(E14+'/'+sub,'learning-rate and trainable-reference controls / full r2SCAN')
for f in ['launch_controls.sh','launch_r2scan_full.sh','HISTORICAL_ONLY_NOTICE.md','EQV3_BOUNDARY_WITHDRAWAL_20261001.md','UNIFIED_ANALYSIS_REPORT.md']:
 add(E14+'/'+f,'historical boundary and launch configuration')
for f in ['eqv3_fitted_reference.csv','eqv3_fitted_offset_coefficients.npz']:
 add(E14+'/'+f,'EqV3 fitted reference')
tree(E17,'corrected physical EqV3 experiment',skip=('archive','gates'))
for sub in ['code','inputs','results/controls_export_v1','results/force100_v1','results/parameter_displacement_v1','results/trajectory_independent_v2/final_cohort_v3','results/trajectory_independent_v2/model_evaluation_v2','results/trajectory_independent_v2/certification_prefix5000']:
 tree(ROB+'/'+sub,'robustness, 100-structure forces and fixed MP trajectories')
for f in ['COMPLETE_HISTORY_CERTIFICATE.json','NATIVE_GRAPH_VERIFICATION.json','MP_TRAJECTORY_INDEPENDENT_PROTOCOL_v2.json','TRAIN_COMPOSITION_OVERLAP.json','LOCAL_ARRAY_SCHEMA_AUDIT.json']:
 add(ROB+'/results/trajectory_independent_v2/'+f,'trajectory certification summary')
tree('audit_energy_allocation_20261003','energy definition and provenance audit')
tree('experiment15_prediction_audit_20261001/code','EqV3 initialization and evaluation dependencies')
tree('experiment15_prediction_audit_20261001/support','cohort and physical decoder initialization metadata')
tree('experiment11_20260928/confirmation','fixed absolute and relative evaluation cohort',weights=True,skip=('fp32_diagnostic',))
medium='runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision/fair_finetune_293/graph_cache/medium'
tree(medium,'original 8192/1024/2048 split and native SevenNet graphs',weights=True)
tree('runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269','original split CSVs',skip=('cifs','cif'))
add('runs/rclfa_v3/real_fegx_foundation_replacement_272/reference_conversion/sevennet_reference/calibration_coefficients.csv','train-fitted 118 element coefficients and bias')
# Add selected model states, not every intermediate epoch.
for mf in [MAIN+'/sevennet_final_eval_v2/input_manifest.csv',MAIN+'/synthetic_reference_control/final_energy/input_manifest.csv',E17+'/analysis/selected_checkpoints.csv']:
 for row in csv.DictReader((ROOT/mf).open()):
  for key in ['path','checkpoint']:
   val=row.get(key,'')
   if val.startswith(str(ROOT)+'/'):add(val,'selected artifact referenced by '+mf)
for folder in [ROOT/E14/'runs',ROOT/E14/'r2scan10k_full/runs']:
 for p in folder.glob('*/best.pt'):add(p,'validation-selected control checkpoint')
# Include graph artifacts necessary to rerun training/evaluation with the historical definitions.
for rel in [E17+'/cache/graphs.pt',E14+'/r2scan10k_full/graph_cache/train_graphs.pt',E14+'/r2scan10k_full/graph_cache/val_graphs.pt',E14+'/r2scan10k_full/graph_cache/test_graphs.pt']:
 add(rel,'native cached model input graphs')
# Root-local Python import closure of custom model utilities.
queue=['rclfa_v3/eqv3_native.py','rclfa_v3/scripts/run_eqv3_native_upper_bound_smoke_20260725.py','rclfa_v3/scripts/run_sevennet_upper_bound_279.py']
seen=set()
while queue:
 rel=queue.pop()
 if rel in seen:continue
 seen.add(rel);p=ROOT/rel
 if not p.exists():continue
 add(p,'transitive local Python dependency')
 for parent in p.relative_to(ROOT).parents:
  if str(parent)!='.':
   init=ROOT/parent/'__init__.py'
   if init.exists():add(init,'package initializer')
 for n in ast.walk(ast.parse(p.read_text())):
  mods=[]
  if isinstance(n,ast.ImportFrom) and n.module:mods=[n.module]
  elif isinstance(n,ast.Import):mods=[a.name for a in n.names]
  for mod in mods:
   if mod.startswith('rclfa_v3'):
    q=mod.replace('.','/')+'.py'
    if (ROOT/q).exists():queue.append(q)
# All CIFs specifically referenced by the frozen datasets, not unrelated datasets.
metadata=[medium+'/graph_metadata.csv','experiment11_20260928/confirmation/manifest.csv',E14+'/r2scan10k_full/graph_cache/graph_metadata.csv',E17+'/cache/metadata.csv']
for rel in metadata:
 for row in csv.DictReader((ROOT/rel).open()):
  value=row.get('cif_path','')
  if value:add(value,'structure referenced by '+rel)
add('external_datasets/functional_protocol_round2_large10k/meta_gga_r2scan/cleaned/meta_gga_r2scan.csv','r2SCAN upstream labels and provenance')
# Retain exact third-party source tree used for EqV3, with its licenses.
tree('external_repos/equiformer_v3-main','fixed EqV3 source and third-party notices',skip=('fig','.github','tests','docs'))
# These pretrained weights are archived for preservation; notices keep their original provenance.
add('checkpoints/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt','EqV3 original pretrained checkpoint')
seven_candidates=[Path('/home/yxzhang/envs/pet_forme/lib/python3.11/site-packages/sevenn'),Path('/home/yxzhang/envs/foundation_teacher/lib/python3.11/site-packages/sevenn')]
for base in seven_candidates:
 for p in base.glob('pretrained_potentials/SevenNet_0__11Jul2024/*'):
  if p.is_file():add(p,'SevenNet original 7net-0 checkpoint and metadata')
 if any(str(base) in x for x in selected):break
# Original environment metadata (no environment variables or shell histories).
for name in ['pet_forme','foundation_teacher','sevennet_cuda']:
 py=Path('/home/yxzhang/envs')/name/'bin/python'
 if not py.exists():continue
 probe='import sys,json,importlib.metadata as m; names=["torch","sevenn","ase","scikit-learn","numpy","torch-geometric","e3nn","fairchem-core","pymatgen","pandas"]; d={"python":sys.version,"executable":sys.executable}; exec("for n in names:\\n try:d[n]=m.version(n)\\n except m.PackageNotFoundError:d[n]=None"); print(json.dumps(d,indent=2))'
 r=subprocess.run([str(py),'-c',probe],capture_output=True,text=True)
 (OUT/'environment'/f'{name}-installed.json').write_text(r.stdout or r.stderr)
 r=subprocess.run([str(py),'-m','pip','freeze'],capture_output=True,text=True)
 (OUT/'environment'/f'{name}-pip-freeze.txt').write_text(r.stdout)
# Read known API secret only for non-disclosing exclusion/redaction checks.
secrets=[]
key=Path('/home/yxzhang/.mp_api_key')
if key.exists():
 secret=key.read_bytes().strip()
 if len(secret)>=16:secrets.append(secret)
records=[];redactions=[];bytes_total=0
print('SELECTED',len(selected),'files',sum(Path(x).stat().st_size for x in selected),'bytes',flush=True)
for i,(src,reason) in enumerate(sorted(selected.items())):
 p=Path(src)
 if p.is_relative_to(ROOT):dest=Path('energy')/p.relative_to(ROOT)
 else:dest=Path('third_party/sevennet')/p.relative_to(p.parents[2])
 target=OUT/dest;target.parent.mkdir(parents=True,exist_ok=True)
 raw=p.read_bytes();orig=hashlib.sha256(raw).hexdigest()
 for secret in secrets:
  if secret in raw:
   if p.suffix not in ('.py','.json','.csv','.md','.log','.txt','.sh','.yaml','.yml'):raise RuntimeError('Secret detected in binary artifact: '+str(dest))
   raw=raw.replace(secret,b'REDACTED_MP_API_KEY');redactions.append(str(dest))
 target.write_bytes(raw)
 records.append({'source':src,'release_path':str(dest),'bytes':len(raw),'source_sha256':orig,'release_sha256':hashlib.sha256(raw).hexdigest(),'reason':reason,'redacted':orig!=hashlib.sha256(raw).hexdigest()})
 bytes_total+=len(raw)
 if i%3000==0:print('COPIED',i,'/',len(selected),flush=True)
(OUT/'provenance/source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(OUT/'provenance/missing_sources.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2))
(OUT/'provenance/redactions.json').write_text(json.dumps(redactions,indent=2))
(OUT/'provenance/packaging_scope.json').write_text(json.dumps({'created':'2026-10-16','source_root':str(ROOT),'source_files':len(records),'bytes':bytes_total,'excluded':['unrelated experiments','all virtual environments and caches','private credentials and shell histories','intermediate epoch states except selected checkpoints','interrupted / obsolete EqV3 runs','obsolete SevenNet v1 evaluation','smoke runs','full server-wide multi-gigabyte geometry inventories'],'preserved_original_paths':True,'original_files_unchanged':True},indent=2))
print('STAGED',len(records),'files',bytes_total,'bytes','MISSING',len(missing),'REDACTED',len(redactions),flush=True)
