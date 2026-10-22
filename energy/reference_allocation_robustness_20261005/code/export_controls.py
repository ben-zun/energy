"""Read frozen LR/T checkpoints, export predictions; writes only new --output."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TIERS=[('T',3e-4),('U',1e-4),('A',1e-4),('U',3e-4),('A',3e-4),('U',1e-3),('A',1e-3)]
SEEDS=[20260528,20260529,20260530]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.output.exists():raise FileExistsError('refusing output overwrite')
    import torch
    src=Path('/home/yxzhang/energy/experiment14_alignment_controls_20260929/code')
    sys.path.insert(0,str(src))
    import evaluate_controls as controls
    from alignment_common import build_model
    graphs,manifest,fractions,offset=controls.confirmation_setup()
    if manifest.id.duplicated().any() or len(manifest)!=2436:raise ValueError('unexpected confirmation manifest')
    model=build_model(torch.device(args.device))
    provenance=[];frames=[];histories=[];hashes={str(src/name):sha(src/name) for name in ['evaluate_controls.py','train_controls.py']}
    for route,lr in TIERS:
        for seed in SEEDS:
            if lr==3e-4 and route in ['U','A']:
                run=controls.exp12_run(route,seed)
                selected=pd.read_csv(controls.EXP12/'results/selected_checkpoints.csv',keep_default_na=False)
                pick=selected[(selected.comparison=='final_best')&(selected.route==route)&(selected.seed==seed)]
                if len(pick)!=1:raise ValueError('ambiguous frozen checkpoint choice')
                epoch=int(pick.iloc[0].epoch);checkpoint=run/f'epoch{epoch:03d}.pt'
            else:
                run=controls.exp14_run(route,seed,lr)
                status=json.loads((run/'status.json').read_text())
                epoch=int(status['best_epoch']);checkpoint=run/'best.pt'
            manifest_path=run/'run_manifest.json'
            meta=json.loads(manifest_path.read_text())
            if meta['seed']!=seed or meta['route']!=route or meta['max_epochs']!=120 or meta['learning_rate']!=lr:
                raise ValueError('run configuration mismatch')
            hashes[str(manifest_path)]=sha(manifest_path);hashes[str(checkpoint)]=sha(checkpoint)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload['seed']!=seed or payload['route']!=route or payload['epoch']!=epoch:raise ValueError('checkpoint identity mismatch')
            del payload
            native,calibrated,info=controls.load_run(route,seed,lr,model,graphs,offset,fractions,torch.device(args.device))
            if info['epoch']!=epoch:raise ValueError('selection mismatch')
            frame=pd.DataFrame(dict(id=manifest.id,composition_key=['|'.join(map(str,np.rint(row*n).astype(int))) for row,n in zip(fractions,manifest.natoms)],truth_ev_atom=manifest.y_true,native_ev_atom=native,calibrated_ev_atom=calibrated,route=route,seed=seed,learning_rate=lr,selected_epoch=epoch))
            if not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all():raise ValueError('nonfinite predictions')
            frames.append(frame)
            provenance.append(dict(route=route,seed=seed,learning_rate=lr,selected_epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=hashes[str(checkpoint)],max_epochs=meta['max_epochs'],initial_hash=meta['init_hash'],reference_initialisation=meta.get('reference_initialisation','frozen_or_none')))
            history_path=run/'history.csv';hashes[str(history_path)]=sha(history_path)
            history=pd.read_csv(history_path,keep_default_na=False)
            if len(history)!=120 or set(history.epoch)!=set(range(1,121)):raise ValueError('incomplete training history')
            history['route']=route;history['seed']=seed;history['learning_rate']=lr;histories.append(history)
            for path in [run/'calibration_history.csv']:
                hashes[str(path)]=sha(path)
            print(json.dumps(dict(route=route,seed=seed,lr=lr,selected_epoch=epoch,status='exported')),flush=True)
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:raise ValueError('source artifact changed')
    args.output.mkdir(parents=True)
    pd.concat(frames,ignore_index=True).to_csv(args.output/'predictions.csv',index=False)
    pd.DataFrame(provenance).to_csv(args.output/'checkpoint_provenance.csv',index=False)
    pd.concat(histories,ignore_index=True).to_csv(args.output/'histories.csv',index=False)
    (args.output/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps({'status':'PASS','runs':21,'confirmation_n':2436,'selection':'original frozen validation selection; no confirmation tuning','writes':'new output directory only'},indent=2)+'\n')

if __name__=='__main__':main()
