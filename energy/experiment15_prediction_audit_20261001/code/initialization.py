from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read

ROOT=Path('/home/yxzhang/energy')
EXP=ROOT/'experiment14_alignment_controls_20260929'
OUT=ROOT/'experiment15_prediction_audit_20261001'
sys.path.insert(0,str(ROOT))
from rclfa_v3.eqv3_native import (ensure_eqv3_imports,configure_eqv3_energy_scaling,
    configure_eqv3_tuning_mode,configure_eqv3_gradient_checkpointing,SplitStructureRecord,collate_graph_records,seed_everything)


def weight_hash(model,head_only=False):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if head_only and not name.startswith('energy_block.'):
            continue
        digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(config,device):
    ensure_eqv3_imports()
    from fairchem.core import OCPCalculator
    seed_everything(int(config['seed']))
    calculator=OCPCalculator(checkpoint_path=config['checkpoint'],cpu=False,seed=config['seed'],only_output=['energy'])
    calculator.trainer.scaler=None
    model=getattr(calculator.trainer,'_unwrapped_model',None) or calculator.trainer.model
    scale=configure_eqv3_energy_scaling(model)
    configure_eqv3_gradient_checkpointing(model,enabled=not config['disable_gradient_checkpointing'])
    model.to(device)
    configure_eqv3_tuning_mode(model,tuning_type=config['tuning_type'],lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'])
    model.eval()
    return calculator,model,scale


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','val'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--finalize-existing',action='store_true')
    parser.add_argument('--weights-only',action='store_true')
    parser.add_argument('--seed',type=int,default=20260528)
    args=parser.parse_args()
    (OUT/'support').mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(args.device)
    configs={}
    for route in ('direct','clean'):
        path=EXP/f'eqv3_final40/{route}/seed{args.seed}/final40_{route}_s{args.seed}_best.pt'
        configs[route]=torch.load(path,map_location='cpu',weights_only=False)['config']
    before=[]
    for route in ('direct','clean'):
        calculator,model,scale=build(configs[route],args.device)
        before.append({'route':route,'weight_hash':weight_hash(model),'head_hash':weight_hash(model,True),
            'scale':scale,'lora_B_abs_max':max(float(parameter.detach().abs().max()) for name,parameter in model.named_parameters() if name.endswith('lora_B'))})
        if route=='direct':
            del calculator,model
    if before[0]['weight_hash']!=before[1]['weight_hash'] or before[0]['head_hash']!=before[1]['head_hash']:
        raise AssertionError('Route initialization differs')
    if args.weights_only:
        (OUT/'support'/f'initialization_weights_seed{args.seed}.json').write_text(json.dumps({'seed':args.seed,'routes':before,
            'same_initial_weights':True,'same_output_head':True},indent=2))
        return
    model_hash=weight_hash(model)
    config=configs['clean']
    normalization={'energy_rmsd':float(calculator.trainer.normalizers['energy'].rmsd.cpu()),
                   'energy_mean':float(calculator.trainer.normalizers['energy'].mean.cpu()),
                   'element_references':calculator.trainer.elementrefs['energy'].element_references.detach().cpu().tolist()}
    if args.finalize_existing:
        existing=pd.read_csv(OUT/'support'/f'initialization_rows_{args.split}.csv')
        expected=8192 if args.split=='train' else 1024
        if len(existing[existing['mode'].eq('eval')])!=expected:
            raise ValueError('Incomplete saved inference')
        (OUT/'support'/f'initialization_state_{args.split}.json').write_text(json.dumps({
            'split':args.split,'route_states':before,'no_parameter_updates':True,'beta':float(config['huber_beta']),
            'loss_units':'scaled total energy (E_total / scale_divisor), not per atom','config':config,
            'drop_path_rate':getattr(model,'drop_path_rate',None),'rows':expected,'seed':config['seed'],'normalization':normalization,
            'metadata_recovery':'forward and unchanged-weight checks completed before original Path JSON serialization failure; inference CSV retained'},indent=2,default=str))
        return
    dataset=Path(config['dataset_root'])
    data=pd.read_csv(dataset/args.split/f'{args.split}.csv',keep_default_na=False)
    reference=pd.read_csv(config['reference_csv'],keep_default_na=False).set_index('id')
    archive=pd.read_csv(ROOT/'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration/calibrated_predictions.csv',keep_default_na=False).set_index('id')
    expected=data.id.to_list()
    records=[]
    for row in data.itertuples(index=False):
        atoms=read(row.cif_path); natoms=len(atoms)
        value=float(reference.loc[row.id,config['reference_column']])
        target=float(archive.loc[row.id,'y_true'])
        if str(reference.loc[row.id,'split'])!=args.split:
            raise ValueError('Reference split mismatch')
        records.append(SplitStructureRecord(split=args.split,sample_id=row.id,cif_path=row.cif_path,
            natoms=natoms,target_per_atom_ev=target,target_total_ev=target*natoms,
            reference_per_atom_ev=value,reference_total_ev=value*natoms,data_object=calculator.a2g.convert(atoms)))
    rows=[]
    batch_size=int(config['batch_size']); beta=float(config['huber_beta'])
    criterion=torch.nn.SmoothL1Loss(beta=beta,reduction='none')
    modes=['eval']+(['train_no_update'] if args.split=='train' else [])
    for mode in modes:
        model.train(mode=='train_no_update')
        seed_everything(config['seed'])
        with torch.no_grad():
            for start in range(0,len(records),batch_size):
                bundle=collate_graph_records(records[start:start+batch_size],scale_divisor=scale,residual_mode=True)
                batch=bundle.batch.to(args.device)
                clean_target=bundle.target_scaled.to(args.device)
                direct_target=torch.tensor(bundle.y_true_total_ev/scale,device=args.device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=config['amp_dtype']=='bfloat16'):
                    output=model(batch)['energy'].reshape(-1)
                    direct_loss=criterion(output,direct_target)
                    clean_loss=criterion(output,clean_target)
                prediction_total=output.float().cpu().numpy()*scale
                for index,sample_id in enumerate(bundle.ids):
                    direct=float(prediction_total[index]/bundle.natoms[index])
                    offset=float(bundle.reference_total_ev[index]/bundle.natoms[index])
                    rows.append({'id':sample_id,'split':args.split,'mode':mode,'natoms':int(bundle.natoms[index]),
                        'truth_ev_atom':float(bundle.y_true_per_atom_ev[index]),'native_direct_ev_atom':direct,
                        'native_clean_ev_atom':direct+offset,'offset_ev_atom':offset,
                        'direct_loss':float(direct_loss[index].float().cpu()),'clean_loss':float(clean_loss[index].float().cpu()),
                        'direct_loss_residual':float(output[index].float().cpu())-float(direct_target[index].cpu()),
                        'clean_loss_residual':float(output[index].float().cpu())-float(clean_target[index].cpu()),
                        'archive_energy_ev_atom':float(archive.loc[sample_id,'model_energy_per_atom'])})
                if start%512==0:
                    print(args.split,mode,start,len(records),flush=True)
    if weight_hash(model)!=model_hash:
        raise AssertionError('Parameters changed in read-only audit')
    pd.DataFrame(rows).to_csv(OUT/'support'/f'initialization_rows_{args.split}.csv',index=False)
    (OUT/'support'/f'initialization_state_{args.split}.json').write_text(json.dumps({
        'split':args.split,'route_states':before,'no_parameter_updates':True,'beta':beta,
        'loss_units':'scaled total energy (E_total / scale_divisor), not per atom',
        'config':config,'drop_path_rate':getattr(model,'drop_path_rate',None),'normalization':normalization,
        'rows':len(records),'seed':config['seed'],'same_output_reused_for_both_identical_route_initializations':True},indent=2,default=str))


if __name__=='__main__':
    main()
