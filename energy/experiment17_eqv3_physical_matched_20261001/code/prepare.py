import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch
from ase.io import read

from common import ROOT,OUT,make_model,fractions,sha


def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    if (OUT/'LOCK.json').exists():
        raise FileExistsError('Protocol already frozen')
    state=json.loads((ROOT/'experiment16_eqv3_output_contract_20261001/forward_train.json').read_text())
    data=[]
    for split in ('train','val'):
        frame=pd.read_csv(ROOT/f'experiment16_eqv3_output_contract_20261001/live_{split}.csv',keep_default_na=False)
        metadata=pd.read_csv(ROOT/f'runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269/{split}/{split}.csv',keep_default_na=False)
        frame=frame.merge(metadata[['id','cif_path']],on='id',validate='one_to_one')
        data.append(frame[['id','split','cif_path','formula','natoms','y_true']])
    confirmation=pd.read_csv(ROOT/'experiment11_20260928/confirmation/manifest.csv',keep_default_na=False)
    confirmation['split']='confirmation'
    data.append(confirmation[['id','split','cif_path','formula','natoms','y_true']])
    frame=pd.concat(data,ignore_index=True)
    if frame.id.duplicated().any():
        raise ValueError('Fit/select/confirmation overlap')
    xvalues=fractions(frame)
    counts=xvalues*frame.natoms.to_numpy()[:,None]
    np.testing.assert_allclose(counts,np.rint(counts),atol=1e-6,rtol=0)
    frame['group']=['|'.join(map(str,row)) for row in np.rint(counts).astype(int)]
    coefficients=np.load(ROOT/'experiment16_eqv3_output_contract_20261001/reference_physical_fp32.npz')
    frame['h']=xvalues@coefficients['w']+float(coefficients['b'])
    frame['physical_offset_total']=counts@np.array(state['element_references'])[1:119]+state['mean']
    if frame.groupby('group').h.agg(lambda values:values.max()-values.min()).max()>1e-9:
        raise AssertionError('Composition reference not group constant')
    frame.to_csv(OUT/'cache/metadata.csv',index=False)
    np.save(OUT/'cache/fractions.npy',xvalues)
    calculator,model=make_model(20260528,'cuda:0')
    graphs=[]
    for index,row in enumerate(frame.itertuples(index=False)):
        atoms=read(row.cif_path)
        if len(atoms)!=row.natoms:
            raise ValueError('CIF atom-count mismatch')
        graph=calculator.a2g.convert(atoms)
        graph.audit_row=torch.tensor([index],dtype=torch.long)
        graphs.append(graph)
    torch.save(graphs,OUT/'cache/graphs.pt')
    sources=[ROOT/'experiment16_eqv3_output_contract_20261001/reference_physical_fp32.npz',
        ROOT/'experiment15_prediction_audit_20261001/support/cleaning_mask.csv',
        ROOT/'experiment15_prediction_audit_20261001/support/cohort.csv',
        ROOT/'checkpoints/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt']
    lock={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':[20260528,20260529,20260530],
        'routes':['direct','clean'],'epochs':40,'early_stopping':False,'batch_size':8,
        'decoder':'E_total=sigma*q+mean+sum(n_Z*epsilon_Z)+N*h; h=0 for direct, frozen physical reference for clean',
        'sigma':state['rmsd'],'mean':state['mean'],'loss':'physical total SmoothL1 beta=0.05 eV',
        'precision':'network bfloat16 autocast; physical decode and scalar loss float64 for cancellation stability',
        'lora_rank':8,'lora_alpha':16.,'lr_head':2e-4,'lr_lora':2e-4,'weight_decay':1e-4,'clip_grad_norm':2.,
        'lr_schedule':'ReduceLROnPlateau val native per-atom MAE; factor .5 patience20; same rule both routes',
        'checkpoint_selection':'minimum validation native per-atom MAE over exactly40 epochs',
        'post_calibration':'best-checkpoint train predictions only; Ridge alpha1e-6 on 118 fractions + intercept, applied equally to both routes',
        'confirmation':'same previously inspected holdout, cleaned190groups401structures primary; no tuning; not claimed novel blind data',
        'first_seed_gate':'runtime/gradient/unit/coverage only, never effect direction; remaining seeds mandatory after operational gate',
        'trainable_device':'all LoRA and energy head parameters on GPU after attachment; no CPU LoRA offload',
        'input_sha256':{str(path):sha(path) for path in sources},
        'metadata_sha256':sha(OUT/'cache/metadata.csv'),'graphs_sha256':sha(OUT/'cache/graphs.pt')}
    (OUT/'LOCK.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock,indent=2))


if __name__=='__main__':
    main()
