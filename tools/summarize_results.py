#!/usr/bin/env python3
"""Offline numerical cross-checks; Python standard library only. No training/API use."""
from pathlib import Path
import csv, json, statistics as st, itertools, collections, argparse
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'energy'; M=E/'experiment_reference_allocation_final_20261004'
R=E/'reference_allocation_robustness_20261005'
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def stats(vals):
    vals=list(map(float,vals));return {'n':len(vals),'mean':st.mean(vals),'sample_std':st.stdev(vals) if len(vals)>1 else 0.0}
def main():
    out={'note':'Generated at packaging time from archived CSVs; not a new model experiment.', 'tables':{}, 'checks':[]}
    def put(name, data, field, expected=None, places=3):
        s=stats(x[field] for x in data);out['tables'][name]=s
        if expected is not None:
            ok=round(s['mean'],places)==round(expected,places)
            out['checks'].append({'name':name,'expected_mean':expected,'computed_mean':s['mean'],'decimal_places':places,'passed':ok})
    for model,folder,cohort,targets in [
        ('SevenNet','sevennet_final_eval_v2','common401',{'direct':(69.911,67.171,.719),'aligned':(21.474,23.379,.844)}),
        ('EqV3','eqv3_final_eval','relative401',{'direct':(62.384,54.569,.684),'aligned':(17.789,17.154,.840)})]:
        data=rows(M/folder/'metrics/per_seed_metrics.csv')
        for route,expected in targets.items():
            a=[x for x in data if x['route']==route and x['calibration']=='calibrated']
            put(f'{model}/{route}/absolute_mae_mev_atom',[x for x in a if x['cohort']=='overall2048'],'mae_mev',expected[0])
            rel=[x for x in a if x['cohort']==cohort]
            put(f'{model}/{route}/pair_delta_mae_mev_atom',rel,'pair_delta_mae_mev',expected[1])
            put(f'{model}/{route}/top1_accuracy',rel,'top1_accuracy',expected[2])
    data=rows(M/'required_vs_induced/summary_with_group_bootstrap.csv')
    for route,ex in [('direct',(2.641,.498,2.593)),('aligned',(.741,.358,.649))]:
        a=[x for x in data if x['model']=='sevennet' and x['route']==route]
        for field,expected in zip(['norm_ratio','projection_alpha','orthogonal_norm_ratio'],ex):put(f'SevenNet/{route}/{field}',a,field,expected)
    data=rows(M/'synthetic_reference_control/final_energy/metrics/per_seed_metrics.csv')
    force=rows(M/'synthetic_reference_control/final_force/force_drift_summary.csv')
    f100=rows(R/'results/force100_v1/force_drift_summary.csv')
    grouped=collections.defaultdict(list)
    for x in rows(R/'results/force100_v1/force_vectors.csv'):
        grouped[(x['condition'],x['seed'],x['id'])].extend((float(x['adapted_f'+a])-float(x['base_f'+a]))**2 for a in 'xyz')
    structure_rmse=collections.defaultdict(list)
    for (condition,seed,ident),errors in grouped.items():structure_rmse[(condition,seed)].append(st.mean(errors)**.5)
    f100_structure=[{'condition':c,'seed':s,'structure_mean_rmse_ev_ang':st.mean(v)} for (c,s),v in structure_rmse.items()]
    for cond,ex in [('zero_direct',(.902,.207,.00415,.00235)),('reference_direct',(53.990,18.770,.74877,.4169)),('reference_aligned',(.995,.210,.00382,.00247))]:
        a=[x for x in data if x['condition']==cond and x['calibration']=='calibrated']
        put(f'synthetic/{cond}/absolute_mae_mev_atom',a,'synthetic_mae_mev',ex[0])
        put(f'synthetic/{cond}/pair_drift_mev_atom',a,'drift_pair_delta_mae_mev',ex[1])
        put(f'force12/{cond}/rmse_ev_ang',[x for x in force if x['condition']==cond],'force_drift_rmse_ev_ang',ex[2],5)
        put(f'force100/{cond}/structure_mean_rmse_ev_ang',[x for x in f100_structure if x['condition']==cond],'structure_mean_rmse_ev_ang',ex[3],4 if cond=='reference_direct' else 5)
    data=rows(M/'cross_protocol_summary/summary.csv')
    for route,ex in [('direct',(83.667,24.175,.463)),('aligned',(23.943,12.979,.744))]:
        a=[x for x in data if x['route']==route]
        for field,expected in zip(['absolute_mae_full_scope_mev_atom','pair_delta_mae_mev_atom','top1_accuracy'],ex):put(f'r2SCAN/{route}/{field}',a,field,expected)
    # Fixed task/snapshot pairs, not pairs across different tasks.
    t=R/'results/trajectory_independent_v2'
    cohort=rows(t/'final_cohort_v3/FINAL_TRAJECTORY_COHORT.csv')
    pred=rows(t/'model_evaluation_v2/PREDICTIONS.csv')
    tasks=collections.defaultdict(list)
    for x in cohort:tasks[x['task_id']].append(x)
    lookup={(x['snapshot_id'],x['route']):float(x['native_energy_per_atom']) for x in pred}
    routes=sorted({x['route'] for x in pred});task_metrics=[];pair_rows=[]
    for task,group in sorted(tasks.items()):
        pairs=list(itertools.combinations(group,2))
        for i,j in pairs:pair_rows.append({'task_id':task,'snapshot_i':i['snapshot_id'],'snapshot_j':j['snapshot_id']})
        metrics={}
        for route in routes:
            errors=[]
            for i,j in pairs:
                actual=float(i['DFT_energy_per_atom'])-float(j['DFT_energy_per_atom'])
                estimate=lookup[(i['snapshot_id'],route)]-lookup[(j['snapshot_id'],route)]
                errors.append(abs(estimate-actual)*1000)
            metrics[route]=st.mean(errors)
        metrics['Direct_seed_mean']=st.mean(metrics[x] for x in routes if x.startswith('Direct'))
        metrics['Aligned_seed_mean']=st.mean(metrics[x] for x in routes if x.startswith('Aligned'))
        task_metrics.append({'task_id':task,'n_structures':len(group),'n_pairs':len(pairs),**metrics})
    traj={'n_tasks':len(tasks),'n_structures':len(cohort),'n_pairs':len(pair_rows),'run_types':dict(collections.Counter(x[0]['run_type'] for x in tasks.values()))}
    for route in ['Frozen','Frozen+h','Direct_seed_mean','Aligned_seed_mean']:
        traj[route+'_task_median_mev_atom']=st.median(x[route] for x in task_metrics)
    traj['aligned_better_than_direct_tasks']=sum(x['Aligned_seed_mean']<x['Direct_seed_mean'] for x in task_metrics)
    out['trajectory']=traj
    for route,expected in [('Frozen',54.4),('Aligned_seed_mean',84.9),('Direct_seed_mean',137.3)]:
        actual=traj[route+'_task_median_mev_atom'];out['checks'].append({'name':'trajectory/'+route,'expected_mean':expected,'computed_mean':actual,'decimal_places':1,'passed':round(actual,1)==expected})
    out['checks'].append({'name':'trajectory/41_of_50','passed':traj['aligned_better_than_direct_tasks']==41})
    out['passed']=all(x['passed'] for x in out['checks'])
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    for name,data in [('trajectory_fixed_pairs.csv',pair_rows),('trajectory_task_metrics.csv',task_metrics)]:
        with (checks/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (checks/'paper_number_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':out['passed'],'checks':len(out['checks']),'failures':[x for x in out['checks'] if not x['passed']],'trajectory':traj},indent=2))
    return 0 if out['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
