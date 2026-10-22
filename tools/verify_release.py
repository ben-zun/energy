#!/usr/bin/env python3
"""Verify archive integrity and frozen experiment definitions, without model imports."""
from pathlib import Path
import ast, csv, hashlib, itertools, json, math, re, sys, collections
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'energy';M=E/'experiment_reference_allocation_final_20261004';R=E/'reference_allocation_robustness_20261005'
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    errors=[];checks={}
    def check(name,ok,detail=None):
        checks[name]={'passed':bool(ok),'detail':detail}
        if not ok:errors.append(name)
    records=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    bad=[]
    for x in records:
        p=ROOT/x['release_path']
        if not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['release_sha256']:bad.append(x['release_path'])
    check('original_artifact_hashes',not bad,{'files':len(records),'bad':bad})
    check('no_missing_sources',json.loads((ROOT/'provenance/missing_sources.json').read_text())==[])
    split=E/'runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269'
    sets={k:{x['id'] for x in rows(split/k/(k+'.csv'))} for k in ['train','val','test']}
    check('medium_split_counts',{k:len(v) for k,v in sets.items()}=={'train':8192,'val':1024,'test':2048},{k:len(v) for k,v in sets.items()})
    check('medium_split_disjoint',all(not(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)))
    pairs=rows(M/'sevennet_final_eval_v2/metrics/common401_pairs.csv')
    ids={x[k] for x in pairs for k in ['id_i','id_j']}
    check('fixed_relative_cohort',len(pairs)==238 and len(ids)==401 and len({x['group_id'] for x in pairs})==190,{'structures':len(ids),'groups':len({x['group_id'] for x in pairs}),'pairs':len(pairs)})
    # Independently recompute pair errors from actual per-structure predictions.
    pp=rows(M/'sevennet_final_eval_v2/predictions/sevennet_test_predictions.csv')
    lookup={(x['id'],x['seed'],x['route']):x for x in pp}
    discrepancies=[]
    for metric in rows(M/'sevennet_final_eval_v2/metrics/per_seed_metrics.csv'):
        if metric['cohort']!='common401':continue
        field='y_pred_calibrated_mev_atom' if metric['calibration']=='calibrated' else 'y_pred_raw_mev_atom'
        errs=[]
        for pair in pairs:
            a=lookup[(pair['id_i'],metric['seed'],metric['route'])];b=lookup[(pair['id_j'],metric['seed'],metric['route'])]
            errs.append(abs((float(a[field])-float(b[field]))-(float(a['y_target_mev_atom'])-float(b['y_target_mev_atom']))))
        if abs(sum(errs)/len(errs)-float(metric['pair_delta_mae_mev']))>1e-8:discrepancies.append([metric['seed'],metric['route'],metric['calibration']])
    check('raw_prediction_pair_error_recomputed',not discrepancies,discrepancies)
    sel=rows(R/'inputs/force100_selection.csv')
    check('force100_selection',len(sel)==100 and len({x['id'] for x in sel})==100 and {x['id'] for x in sel}<=sets['test'])
    force100=rows(R/'results/force100_v1/force_drift_summary.csv')
    check('force100_nine_selected_runs',len(force100)==9 and all(int(x['n_structures'])==100 for x in force100))
    for name,path in [('force12',M/'synthetic_reference_control/final_force/finite_difference_checks.csv'),('force100',R/'results/force100_v1/finite_difference_checks.csv')]:
        data=rows(path);check(name+'_finite_difference_records',bool(data),{'rows':len(data)})
        # Explicitly validate the saved best-step acceptance rule for force100.
        if name=='force100':
            groups=collections.defaultdict(list)
            for x in data:groups[(x['model'],x['seed'],x['id'],x['atom_index'],x['axis'])].append(x)
            passed=sum(any(float(x['absolute_error_ev_ang'])<=max(.02,.10*abs(float(x['analytic_force_ev_ang']))) for x in group) for group in groups.values())
            check('force100_best_step_finite_difference',passed==len(groups),{'passed_components':passed,'components':len(groups)})
    t=R/'results/trajectory_independent_v2'
    co=rows(t/'final_cohort_v3/FINAL_TRAJECTORY_COHORT.csv');pred=rows(t/'model_evaluation_v2/PREDICTIONS.csv')
    counts=collections.Counter(x['task_id'] for x in co)
    check('trajectory_counts',len(counts)==50 and len(co)==171 and sum(n*(n-1)//2 for n in counts.values())==434)
    check('trajectory_prediction_coverage',len(pred)==171*8 and len({(x['snapshot_id'],x['route']) for x in pred})==171*8)
    # Validate each explicitly selected artifact path in final checkpoint manifests.
    paths=[]
    for p in [M/'sevennet_final_eval_v2/input_manifest.csv',M/'synthetic_reference_control/final_energy/input_manifest.csv',E/'experiment17_eqv3_physical_matched_20261001/analysis/selected_checkpoints.csv']:
        for x in rows(p):
            for k in ['path','checkpoint']:
                if x.get(k,'').startswith('/home/yxzhang/energy/'):
                    q=E/x[k].removeprefix('/home/yxzhang/energy/');paths.append(str(q.relative_to(ROOT)))
    check('selected_checkpoint_manifest_paths',all((ROOT/p).is_file() for p in paths),{'references':len(paths)})
    # Detect accidental private files / obvious embedded credentials; never log values.
    forbidden=[];suspect=[];syntax=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        if p.name.startswith('.env') or p.name in ['id_rsa','id_ed25519'] or '.git' in p.parts:forbidden.append(str(p.relative_to(ROOT)))
        if p.suffix in {'.py','.sh','.json','.yaml','.yml','.txt','.csv','.md'} and p.stat().st_size<30_000_000:
            data=p.read_text(errors='replace')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',data) or re.search(r'(?<![\w-])sk-[A-Za-z0-9]{24,}',data) or re.search(r'https?://[^/\s:@]+:[^/\s@]+@',data):suspect.append(str(p.relative_to(ROOT)))
            if p.suffix=='.py' and 'external_repos' not in p.parts:
                try:ast.parse(data,filename=str(p))
                except SyntaxError:syntax.append(str(p.relative_to(ROOT)))
    check('no_private_files',not forbidden,forbidden);check('no_obvious_embedded_credentials',not suspect,suspect);check('project_python_syntax',not syntax,syntax)
    check('paper_number_check',json.loads((ROOT/'checks/paper_number_check.json').read_text())['passed'])
    result={'passed':not errors,'checks':checks,'failed_checks':errors,'note':'Integrity/statistics verification only. Does not rerun GPU training or establish historical environment compatibility.'}
    (ROOT/'checks/verification_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':not errors,'checks':len(checks),'failed_checks':errors},indent=2))
    return 0 if not errors else 1
if __name__=='__main__':raise SystemExit(main())
