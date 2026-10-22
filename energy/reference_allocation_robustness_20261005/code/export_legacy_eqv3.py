"""Faithful legacy residual-learning control; never relabelled physical-aligned."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def legacy_prediction(delta_total,natoms,route,reference):
    delta_total=np.asarray(delta_total,float);natoms=np.asarray(natoms,float);reference=np.asarray(reference,float)
    if delta_total.shape!=natoms.shape or reference.shape!=natoms.shape or np.any(natoms<=0):raise ValueError('legacy decoding shape/count mismatch')
    if route not in ['direct','residual']:raise ValueError('unknown legacy route')
    output=delta_total/natoms+(reference if route=='residual' else 0)
    if not np.isfinite(output).all():raise ValueError('nonfinite legacy prediction')
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=8);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    import torch
    sys.path.insert(0,'/home/yxzhang/energy/experiment14_alignment_controls_20260929/code')
    import eqv3_symmetric as old
    from alignment_common import atomic_fractions
    torch.cuda.set_device(a.device)
    manifest_path=old.CONF/'manifest.csv';base_path=old.CONF/'eqv3_base.csv'
    manifest=pd.read_csv(manifest_path,keep_default_na=False)
    base=pd.read_csv(base_path,keep_default_na=False)
    if manifest.id.duplicated().any() or base.id.duplicated().any():raise ValueError('duplicate IDs')
    base=base.set_index('id').loc[manifest.id]
    metadata=pd.read_csv(old.MEDIUM_META,keep_default_na=False)
    train_ids=set(metadata.loc[metadata.split=='train','id'])
    if len(train_ids)!=8192 or len(manifest)!=2436:raise ValueError('cohort counts mismatch')
    fractions=atomic_fractions(manifest)
    medium_fractions=pd.DataFrame(atomic_fractions(metadata),index=metadata.id)
    hashes={str(path):sha(path) for path in [manifest_path,base_path,old.MEDIUM_META,Path(old.__file__)]}
    calculator,model=old.build_model(a.device)
    from fairchem.core.datasets import data_list_collater
    from ase.io import read
    graphs=[]
    for row in manifest.itertuples():
        atoms=read(row.cif_path)
        if len(atoms)!=row.natoms:raise ValueError('atom count mismatch')
        graphs.append(calculator.a2g.convert(atoms))
    attached=False;rank_alpha=None;frames=[];provenance=[]
    for route in ['residual','direct']:
        for seed in [20260528,20260529,20260530]:
            checkpoint=old.checkpoint_path(route,seed);predpath=old.predictions_path(route,seed)
            hashes[str(checkpoint)]=sha(checkpoint);hashes[str(predpath)]=sha(predpath)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            config=payload['config'];current=(int(config['lora_rank']),float(config['lora_alpha']))
            from rclfa_v3.eqv3_native import configure_eqv3_tuning_mode
            if not attached:
                configure_eqv3_tuning_mode(model,tuning_type='lora',lora_rank=current[0],lora_alpha=current[1]);attached=True;rank_alpha=current
            if current!=rank_alpha:raise ValueError('incompatible legacy LoRA architectures')
            model.load_state_dict(payload['model'],strict=True)
            scale=float(payload.get('scale_divisor',1.0));model.avg_num_nodes=scale;model.eval();model.regress_forces=False;model.regress_stress=False
            model.gradient_checkpointing_block_list=[0]*len(model.blocks)
            totals=[]
            with torch.no_grad():
                for start in range(0,len(graphs),a.batch):
                    batch=data_list_collater([g.clone() for g in graphs[start:start+a.batch]],otf_graph=True).to(a.device)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):energy=model(batch)['energy'].reshape(-1)
                    totals.extend((energy.float()*scale).cpu().tolist())
            native=legacy_prediction(totals,manifest.natoms,route,base.native_reference)
            preds=pd.read_csv(predpath,keep_default_na=False)
            train=preds[preds.split=='train'].set_index('id')
            if train.index.duplicated().any() or set(train.index)!=train_ids:raise ValueError('legacy train calibration coverage mismatch')
            residual=train.y_true_per_atom_ev.to_numpy(float)-train.y_pred_per_atom_ev.to_numpy(float)
            if not np.isfinite(residual).all():raise ValueError('invalid archived train predictions')
            from sklearn.linear_model import Ridge
            ridge=Ridge(alpha=1e-6,fit_intercept=True).fit(medium_fractions.loc[train.index],residual)
            calibrated=native+ridge.predict(fractions)
            keys=['|'.join(map(str,np.rint(row*n).astype(int))) for row,n in zip(fractions,manifest.natoms)]
            frames.append(pd.DataFrame(dict(id=manifest.id,composition_key=keys,truth_ev_atom=manifest.y_true,native_ev_atom=native,calibrated_ev_atom=calibrated,route='legacy_'+route,seed=seed)))
            provenance.append(dict(route='legacy_'+route,seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=hashes[str(checkpoint)],scale_divisor=scale,reference='archived native_reference(X)' if route=='residual' else 'none',calibration_train_n=len(train),contract='D*q/N + reference(X) for residual, D*q/N for direct; task-reinterpreted native output, not physical decoder'))
            print(json.dumps(dict(route=route,seed=seed,status='exported')),flush=True)
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:raise ValueError('legacy input changed during evaluation')
    a.output.mkdir(parents=True)
    pd.concat(frames,ignore_index=True).to_csv(a.output/'predictions.csv',index=False)
    pd.DataFrame(provenance).to_csv(a.output/'provenance.csv',index=False)
    (a.output/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output/'audit.json').write_text(json.dumps({'status':'PASS','runs':6,'confirmation_n':2436,'classification':'historical legacy task-output residual-learning control; not current physical-aligned','calibration':'archived best-checkpoint train-only predictions, 8192 each; Ridge alpha1e-6','source_reference':'original frozen legacy reference; no repair/imputation performed'},indent=2)+'\n')

if __name__=='__main__':main()
