from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from common import OUT,sha


def check_run(route,seed):
    import numpy as np
    import pandas as pd
    run=OUT/'runs'/route/f'seed{seed}'
    status=json.loads((run/'status.json').read_text())
    history=pd.read_csv(run/'history.csv')
    if status['status']!='complete' or len(history)!=40 or status['epochs']!=40:
        raise ValueError('Incomplete fixed budget')
    if int(history.loc[history.val_mae_mev.idxmin(),'epoch'])!=status['best_epoch']:
        raise ValueError('Checkpoint selection mismatch')
    if not np.isfinite(history.select_dtypes(include=[np.number])).all().all():
        raise ValueError('Invalid training trajectory')
    if sha(run/'best.pt')!=status['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    return status


def run_job(route,seed,gpu):
    run=OUT/'runs'/route/f'seed{seed}'
    if (run/'status.json').exists():
        return check_run(route,seed)
    command=[sys.executable,str(OUT/'code/train.py'),'--route',route,'--seed',str(seed),'--device',f'cuda:{gpu}']
    with (OUT/'logs'/f'{route}_{seed}.log').open('a') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return check_run(route,seed)


def phase(jobs,name):
    (OUT/'DRIVER_STATUS.json').write_text(json.dumps({'phase':name,'started_utc':datetime.now(timezone.utc).isoformat(),
        'jobs':jobs,'effect_direction_is_not_a_gate':True},indent=2))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures={pool.submit(run_job,*job):job for job in jobs}
        for future in as_completed(futures):
            result=future.result()
            print(json.dumps({'completed':futures[future],'best_epoch':result['best_epoch']}),flush=True)


def main():
    (OUT/'logs').mkdir(exist_ok=True)
    phase([('direct',20260528,0),('clean',20260528,1)],'first_matched_seed')
    first=[check_run(route,20260528) for route in ('direct','clean')]
    if first[0]['initial_weight_hash']!=first[1]['initial_weight_hash']:
        raise ValueError('Initial weights differ across routes')
    (OUT/'FIRST_PAIR_GATE.json').write_text(json.dumps({'operational_gate':'passed','effect_direction_not_inspected':True,
        'criteria':['40 epochs each','finite trajectories','val-only checkpoint choice','identical initialization','prediction artifacts'],
        'timestamp_utc':datetime.now(timezone.utc).isoformat()},indent=2))
    phase([('direct',20260529,0),('clean',20260529,1),('direct',20260530,2),('clean',20260530,3)],'remaining_two_matched_seeds')
    subprocess.run([sys.executable,str(OUT/'code/evaluate.py')],check=True)
    (OUT/'DRIVER_STATUS.json').write_text(json.dumps({'phase':'complete','runs':6,'epochs_each':40},indent=2))


if __name__=='__main__':
    main()
