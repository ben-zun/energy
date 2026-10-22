"""Experiment 13: actual net network displacement from frozen theta0, no training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def network_displacement(initial, values):
    initial=np.asarray(initial,dtype=float);values=np.asarray(values,dtype=float)
    if initial.ndim!=1 or initial.shape!=values.shape or not np.isfinite(initial).all() or not np.isfinite(values).all():
        raise ValueError("invalid parameter vectors")
    norm=float(np.linalg.norm(initial))
    if norm==0:raise ValueError("zero initial parameter norm")
    absolute=float(np.linalg.norm(values-initial))
    return absolute, absolute/norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    import torch
    sys.path.insert(0,'/home/yxzhang/energy/experiment12_20260929_alignment/code')
    from alignment_common import build_model
    model=build_model(torch.device('cpu'))
    names=sorted(name for name,param in model.named_parameters() if param.requires_grad)
    initial_state=model.state_dict()
    initial=np.concatenate([initial_state[name].detach().cpu().numpy().astype(np.float64).ravel() for name in names])
    if initial.size!=301300 or not np.isfinite(initial).all():raise ValueError('initial parameter contract mismatch')
    norm=float(np.linalg.norm(initial))
    if norm==0:raise ValueError('zero initial parameter norm')
    rows=[];hashes={}
    for route in ['U','A']:
        for seed in [20260528,20260529,20260530]:
            run=Path('/home/yxzhang/energy/experiment12_20260929_alignment/runs')/f'{route}_{seed}'
            for epoch in range(1,121):
                path=run/f'epoch{epoch:03d}.pt'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                payload=torch.load(path,map_location='cpu',weights_only=False)
                if payload['seed']!=seed or payload['route']!=route or payload['epoch']!=epoch:raise ValueError('checkpoint metadata mismatch')
                values=np.concatenate([payload['model'][name].detach().cpu().numpy().astype(np.float64).ravel() for name in names])
                if values.shape!=initial.shape or not np.isfinite(values).all():raise ValueError('parameter vector contract mismatch')
                displacement, normalized=network_displacement(initial,values)
                rows.append(dict(route=route,seed=seed,epoch=epoch,network_parameter_count=len(initial),initial_network_norm=norm,net_displacement_from_pretrained=displacement,normalized_net_displacement=normalized))
            print(json.dumps(dict(route=route,seed=seed,epochs=120,status='computed')),flush=True)
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output/'per_epoch_parameter_displacement.csv',index=False)
    (a.output/'checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output/'audit.json').write_text(json.dumps({'status':'PASS','epoch_vectors':720,'definition':'L2 displacement of 301300 trainable network parameters from original pretrained initialization; excludes composition reference parameters','initial_parameter_names':names,'initial_vector_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),'gradient_batch_median':'unavailable historical artifact'},indent=2)+'\n')

if __name__=='__main__':main()
