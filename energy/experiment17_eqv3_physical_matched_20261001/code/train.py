from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from common import ROOT,OUT,make_model,weight_hash,sha,physical_energy,physical_loss,group_metrics


def atomic_torch_save(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def atomic_csv(frame,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--route',choices=['direct','clean'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--gate-only',action='store_true')
    args=parser.parse_args()
    lock=json.loads((OUT/'LOCK.json').read_text())
    assert sha(OUT/'cache/metadata.csv')==lock['metadata_sha256']
    run=OUT/('gates' if args.gate_only else 'runs')/args.route/f'seed{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'status.json').exists():
        raise FileExistsError('Run already completed; existing results retained')
    torch.cuda.set_device(args.device)
    calculator,model=make_model(args.seed,args.device)
    from fairchem.core.datasets import data_list_collater
    from rclfa_v3.scripts.run_eqv3_native_upper_bound_smoke_20260725 import build_optimizer
    metadata=pd.read_csv(OUT/'cache/metadata.csv',keep_default_na=False)
    graphs=torch.load(OUT/'cache/graphs.pt',map_location='cpu',weights_only=False)
    if len(graphs)!=len(metadata):
        raise ValueError('Graph coverage mismatch')
    sigma=lock['sigma']; device=torch.device(args.device)
    count=torch.tensor(metadata.natoms.to_numpy(),device=device,dtype=torch.float64)
    target=torch.tensor(metadata.y_true.to_numpy(),device=device,dtype=torch.float64)*count
    physical_offset=torch.tensor(metadata.physical_offset_total.to_numpy(),device=device,dtype=torch.float64)
    h=torch.tensor(metadata.h.to_numpy(),device=device,dtype=torch.float64)*count
    applied=h if args.route=='clean' else torch.zeros_like(h)
    indices={split:np.flatnonzero(metadata.split.eq(split)) for split in ('train','val','confirmation')}
    parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters)!=860417:
        raise ValueError('Unexpected trainable parameter count')
    initial_hash=weight_hash(model)
    def forward(selected):
        batch=data_list_collater([graphs[int(index)].clone() for index in selected],otf_graph=True).to(device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            raw=model(batch)['energy'].reshape(-1)
        return raw
    probes=metadata[metadata.split.eq('val') & metadata.natoms.le(64)]
    group_sizes=probes.groupby('group').size()
    probe_groups=group_sizes[group_sizes>=2].index[:2]
    probe_indices=probes[probes.group.isin(probe_groups)].index.to_numpy()[:8]
    if len(probe_indices)<2:
        raise ValueError('No repeated-composition initialization probe')
    model.train()
    raw=forward(probe_indices)
    gate_rows=[]
    for route,reference in [('direct',torch.zeros_like(h)),('clean',h)]:
        decoded=physical_loss(raw,target[probe_indices],physical_offset[probe_indices],reference[probe_indices],sigma)
        encoded=physical_loss(raw,target[probe_indices],physical_offset[probe_indices],reference[probe_indices],sigma,encoded=True)
        gradient_decoded=torch.autograd.grad(decoded,parameters,retain_graph=True,allow_unused=True)
        gradient_encoded=torch.autograd.grad(encoded,parameters,retain_graph=True,allow_unused=True)
        raw_gradient_decoded=torch.autograd.grad(decoded,raw,retain_graph=True)[0]
        raw_gradient_encoded=torch.autograd.grad(encoded,raw,retain_graph=True)[0]
        gradient_difference=torch.autograd.grad(decoded-encoded,parameters,retain_graph=True,allow_unused=True)
        gradient_repeat=torch.autograd.grad(decoded,parameters,retain_graph=True,allow_unused=True)
        difference_sq=0.; norm_sq=0.; maximum=0.
        for left,right in zip(gradient_decoded,gradient_encoded):
            if left is None and right is None:
                continue
            if left is None or right is None:
                raise AssertionError('Gradient coverage differs')
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise FloatingPointError('Nonfinite gradient in equivalence check')
            difference_sq+=float(((left.double()-right.double())**2).sum())
            norm_sq+=float((left.double()**2).sum())
            maximum=max(maximum,float((left-right).abs().max()))
        relative=np.sqrt(difference_sq)/max(np.sqrt(norm_sq),1e-30)
        repeat_sq=sum(float(((left.double()-right.double())**2).sum()) for left,right in zip(gradient_decoded,gradient_repeat) if left is not None)
        residual_sq=sum(float((gradient.double()**2).sum()) for gradient in gradient_difference if gradient is not None)
        raw_difference=float((raw_gradient_decoded-raw_gradient_encoded).abs().max())
        residual_relative=np.sqrt(residual_sq)/max(np.sqrt(norm_sq),1e-30)
        repeat_relative=np.sqrt(repeat_sq)/max(np.sqrt(norm_sq),1e-30)
        gate_rows.append({'route':route,'loss_decoded':float(decoded.detach()),'loss_encoded':float(encoded.detach()),
            'gradient_relative_l2_error':relative,'gradient_max_abs_error':maximum,
            'same_loss_repeated_backward_relative_error':repeat_relative,'raw_output_gradient_max_difference':raw_difference,
            'gradient_of_loss_difference_relative_norm':residual_relative})
        print(json.dumps(gate_rows[-1]),flush=True)
        if raw_difference!=0 or residual_relative>1e-6 or relative>torch.finfo(torch.bfloat16).eps or abs(float(decoded-encoded))>1e-8:
            raise AssertionError(f'Physical/encoded gradient mismatch: {gate_rows[-1]}')
    direct0=(physical_energy(raw,sigma,physical_offset[probe_indices],torch.zeros_like(h[probe_indices]))/count[probe_indices]).detach().cpu().numpy()
    clean0=(physical_energy(raw,sigma,physical_offset[probe_indices],h[probe_indices])/count[probe_indices]).detach().cpu().numpy()
    groups=metadata.loc[probe_indices,'group'].to_numpy()
    centered=lambda values:values-pd.Series(values).groupby(groups).transform('mean').to_numpy()
    group_error=float(np.max(abs(centered(direct0)-centered(clean0))))
    if group_error>1e-10 or weight_hash(model)!=initial_hash:
        raise AssertionError('Initialization identity failed')
    gate={'gradient_checks':gate_rows,'within_composition_prediction_difference_max_ev_atom':group_error,
        'initial_weight_hash':initial_hash,'parameter_updates_during_gate':0,'probe_ids':metadata.loc[probe_indices,'id'].tolist(),
        'training_mode':True,'network_precision':'bfloat16','physical_loss_precision':'float64'}
    (run/'gate.json').write_text(json.dumps(gate,indent=2))
    del raw,decoded,encoded,gradient_decoded,gradient_encoded
    model.zero_grad(set_to_none=True)
    if args.gate_only:
        (run/'status.json').write_text(json.dumps({'status':'gate_passed',**gate},indent=2))
        print(json.dumps(gate,indent=2)); return
    manifest_path=run/'run_manifest.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({'route':args.route,'seed':args.seed,'started_utc':datetime.now(timezone.utc).isoformat(),
            'lock_sha256':sha(OUT/'LOCK.json'),'code_sha256':sha(__file__),'initial_weight_hash':initial_hash,
            'trainable_parameters':sum(parameter.numel() for parameter in parameters)},indent=2))
    def evaluate(split):
        model.eval(); predictions=[]; loss_sum=0.; saturated=0
        with torch.no_grad():
            for start in range(0,len(indices[split]),8):
                selected=indices[split][start:start+8]
                output=forward(selected)
                total=physical_energy(output,sigma,physical_offset[selected],applied[selected])
                residual=total-target[selected]
                loss_sum+=float(torch.nn.functional.smooth_l1_loss(total,target[selected],beta=.05,reduction='sum'))
                saturated+=int((abs(residual)>=.05).sum())
                predictions.extend((total/count[selected]).cpu().tolist())
        selected=indices[split]
        frame=metadata.iloc[selected][['id','split','formula','natoms','group','y_true','h']].copy()
        frame['prediction']=predictions
        scores=group_metrics(frame.y_true.to_numpy(),frame.prediction.to_numpy(),frame.group.to_numpy())
        scores.update(loss=loss_sum/len(frame),saturated_fraction=saturated/len(frame))
        return scores,frame
    optimizer=build_optimizer(model,lr_head=lock['lr_head'],lr_lora=lock['lr_lora'],lr_backbone=5e-5,weight_decay=lock['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=20)
    latest_path=run/'latest.pt'
    history_path=run/'history.csv'
    best=float('inf'); best_epoch=0; history=[]; start_epoch=1; resumed_from_epoch=0
    if latest_path.exists():
        checkpoint=torch.load(latest_path,map_location='cpu',weights_only=False)
        required={'model','optimizer','scheduler','epoch','route','seed','lock_sha256'}
        if not required.issubset(checkpoint):
            raise ValueError('Checkpoint predates exact-resume support; preserve it and restart this run from initialization')
        if checkpoint['route']!=args.route or checkpoint['seed']!=args.seed or checkpoint['lock_sha256']!=sha(OUT/'LOCK.json'):
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'],device=device)
        resumed_from_epoch=int(checkpoint['epoch']); start_epoch=resumed_from_epoch+1
        if history_path.exists():
            history_frame=pd.read_csv(history_path)
            history_frame=history_frame[history_frame.epoch.le(resumed_from_epoch)].copy()
            history=history_frame.to_dict('records')
        if len(history)!=resumed_from_epoch or (history and int(history[-1]['epoch'])!=resumed_from_epoch):
            raise ValueError('History/checkpoint resume boundary mismatch')
        if history:
            best_row=min(history,key=lambda row:row['val_mae_mev'])
            best=float(best_row['val_mae_mev']); best_epoch=int(best_row['epoch'])
        print(json.dumps({'route':args.route,'seed':args.seed,'resume_from_epoch':resumed_from_epoch}),flush=True)
    else:
        epoch0=[]
        for split in ('train','val'):
            scores,frame=evaluate(split); atomic_csv(frame,run/f'epoch0_{split}.csv')
            epoch0.append({'split':split,**scores})
        atomic_csv(pd.DataFrame(epoch0),run/'epoch0_metrics.csv')
    started=time.perf_counter()
    for epoch in range(start_epoch,41):
        model.train(); order=indices['train'].tolist(); random.Random(args.seed+epoch).shuffle(order)
        training_loss=0.; training_abs=0.; batches=0; epoch_start=time.perf_counter()
        for start in range(0,len(order),8):
            selected=order[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            output=forward(selected)
            loss=physical_loss(output,target[selected],physical_offset[selected],applied[selected],sigma)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite physical loss')
            loss.backward()
            gradient_norm=torch.nn.utils.clip_grad_norm_(parameters,2.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                energy=physical_energy(output,sigma,physical_offset[selected],applied[selected])
                training_abs+=float(torch.sum(abs(energy/count[selected]-target[selected]/count[selected])))
            training_loss+=float(loss.detach())*len(selected); batches+=1
            if start==0 or start%1024==0:
                print(json.dumps({'route':args.route,'seed':args.seed,'epoch':epoch,'rows_done':start+len(selected),'elapsed_epoch_s':time.perf_counter()-epoch_start}),flush=True)
        validation,frame=evaluate('val')
        atomic_csv(frame,run/f'val_epoch{epoch:03d}.csv')
        entry={'epoch':epoch,'train_loss':training_loss/len(order),'train_online_mae_mev':1000*training_abs/len(order),
            **{f'val_{key}':value for key,value in validation.items()},'epoch_seconds':time.perf_counter()-epoch_start,
            'lr_head':optimizer.param_groups[0]['lr'],'last_gradient_norm':float(gradient_norm)}
        history.append(entry); atomic_csv(pd.DataFrame(history),history_path)
        scheduler.step(validation['mae_mev'])
        checkpoint={'model':{name:value.detach().cpu() for name,value in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state(device),
            'epoch':epoch,'route':args.route,'seed':args.seed,'lock_sha256':sha(OUT/'LOCK.json')}
        if validation['mae_mev']<best:
            best=validation['mae_mev']; best_epoch=epoch
            atomic_torch_save({'model':checkpoint['model'],'epoch':epoch,'route':args.route,'seed':args.seed,
                'lock_sha256':checkpoint['lock_sha256']},run/'best.pt')
        atomic_torch_save(checkpoint,latest_path)
        print(json.dumps({'route':args.route,'seed':args.seed,**entry,'best_epoch':best_epoch}),flush=True)
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    for split in ('train','val','confirmation'):
        _,frame=evaluate(split); atomic_csv(frame,run/f'best_{split}_predictions.csv')
    (run/'status.json').write_text(json.dumps({'status':'complete','epochs':40,'best_epoch':best_epoch,'best_val_mae_mev':best,
        'elapsed_seconds':time.perf_counter()-started,'checkpoint_sha256':sha(run/'best.pt'),
        'initial_weight_hash':initial_hash,'no_result_based_expansion':True,
        'exact_resume_supported':True,'resumed_from_epoch':resumed_from_epoch},indent=2))


if __name__=='__main__':
    main()
