"""Read-only inventory of explicitly selected current-paper history roots."""
import argparse,json,os
from pathlib import Path

def roots(base):
    local=base.name=='pcg'
    prefixes=('experiment','formation_energy_protocol','paper_results_repro','methods_repro','methods_protocol','reproducibility_configs','FormDelta_publication','remote_formdelta','invdesform_github_repro') if local else ('experiment','formation_energy_protocol','audit_energy_allocation','paper_revision')
    selected=[p for p in base.iterdir() if p.name.startswith(prefixes)]
    if not local:
        selected += [base/'runs/rclfa_v3',base/'external_benchmarks/matbench_mp_e_form',base/'external_datasets/matbench_discovery/mp']
    else:
        project=base/'reference_allocation_robustness_20261005'
        selected += [project/n for n in ('inputs','inputs_epoch','results','phase2_inventory','phase3_feasibility')]
    return sorted(set(p for p in selected if p.exists()))

def inventory(base):
    selected=roots(base);files=[]
    for root in selected:
        candidates=[root] if root.is_file() else (Path(d)/n for d,dirs,names in os.walk(root) for n in names)
        for p in candidates:
            if any(x in {'.git','node_modules','__pycache__','.venv'} for x in p.parts):continue
            if p.is_symlink():
                files.append({'path':str(p),'symlink_target':os.readlink(p),'status':'requires_resolution'});continue
            st=p.stat();files.append({'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
    return {'roots':[str(p) for p in selected],'files':files,'role':'pre-acquisition current-paper development scope inventory','complete':False}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('base',type=Path);a.add_argument('output',type=Path);p=a.parse_args()
    if p.output.exists():raise FileExistsError(p.output)
    out=inventory(p.base);p.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'roots':len(out['roots']),'files':len(out['files'])}))
