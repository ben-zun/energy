from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition,Element

ROOT=Path('/home/yxzhang/energy')
OUT=ROOT/'experiment17_eqv3_physical_matched_20261001'
sys.path.insert(0,str(ROOT/'experiment15_prediction_audit_20261001/code'))
from initialization import build,weight_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fractions(frame):
    return np.array([[Composition(str(formula)).get_atomic_fraction(Element.from_Z(number)) for number in range(1,119)] for formula in frame.formula])


def make_model(seed,device):
    config=json.loads((ROOT/'experiment15_prediction_audit_20261001/support/initialization_state_val.json').read_text())['config']
    config['seed']=seed
    calculator,model,scale=build(config,device)
    model.to(device)
    if scale!=1.:
        raise ValueError('This locked physical decoder assumes checkpoint avg_num_nodes=1')
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.device!=torch.device(device):
            raise AssertionError('Trainable parameters are not on the selected GPU')
    return calculator,model


def physical_energy(raw,sigma,offset,composition_total):
    return raw.double()*sigma+offset+composition_total


def physical_loss(raw,target_total,offset,composition_total,sigma,encoded=False):
    if encoded:
        target=(target_total-offset-composition_total)/sigma
        return sigma*torch.nn.functional.smooth_l1_loss(raw.double(),target,beta=.05/sigma)
    energy=physical_energy(raw,sigma,offset,composition_total)
    return torch.nn.functional.smooth_l1_loss(energy,target_total,beta=.05)


def group_metrics(truth,prediction,groups):
    data=pd.DataFrame({'error':prediction-truth,'group':groups,'truth':truth,'prediction':prediction})
    repeated=data[data.group.map(data.group.value_counts())>=2]
    centered=repeated.error-repeated.groupby('group').error.transform('mean')
    costs=[]
    for _,block in repeated.groupby('group'):
        costs.append(float(block.loc[block.prediction.idxmin(),'truth']-block.truth.min()))
    return {'mae_mev':1000*float(abs(data.error).mean()),
        'ec_mae_mev':1000*float(abs(centered).mean()) if len(repeated) else None,
        'major_wrong_rate':float(np.mean(np.array(costs)>.01)) if costs else None,
        'groups':len(costs),'relative_n':len(repeated)}
