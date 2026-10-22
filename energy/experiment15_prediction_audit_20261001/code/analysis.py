from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path('/home/yxzhang/energy')
EXP=ROOT/'experiment14_alignment_controls_20260929'
OUT=ROOT/'experiment15_prediction_audit_20261001'
SEEDS=(20260528,20260529,20260530)
DRAWS=10000
SEED=20261001
TARGET_RMS_FLOOR=1e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cohort():
    manifest=pd.read_csv(EXP/'unified/evaluation_manifest.csv',keep_default_na=False)
    mask=manifest.relative_cohort & manifest.keep
    counts=manifest.loc[mask].groupby('exact_group').size()
    mask &= manifest.exact_group.isin(counts[counts>=2].index)
    manifest['analysis_keep']=mask
    manifest.to_csv(OUT/'support/cleaning_mask.csv',index=False)
    cohort=manifest[mask].sort_values(['exact_group','id']).reset_index(drop=True)
    if len(cohort)!=401 or cohort.exact_group.nunique()!=190:
        raise ValueError('Not the requested 190-group/401-structure cohort')
    if cohort.id.duplicated().any() or cohort.label_conflict.any():
        raise ValueError('Duplicate ID or unresolved label conflict in primary cohort')
    source=ROOT/'runs/rclfa_v3/official_benchmark_214/matbench_e_form_5fold/fold0/dataset/test/test.csv'
    labels=pd.read_csv(source,keep_default_na=False).set_index('id').loc[cohort.id]
    np.testing.assert_allclose(cohort.y_true,labels.formation_energy_per_atom_ev,rtol=0,atol=1e-12)
    if not np.array_equal(cohort.cif_path.to_numpy(),labels.cif_path.to_numpy()):
        raise ValueError('Structure source mismatch')
    cohort['cif_sha256']=[digest(path) for path in cohort.cif_path]
    cohort['label_source']=str(source)
    cohort.to_csv(OUT/'support/cohort.csv',index=False)
    return cohort


def aligned(block,cohort):
    if block.id.duplicated().any():
        raise ValueError('Duplicate prediction IDs')
    values=block.set_index('id').loc[cohort.id,'prediction'].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    return values


def predictions(cohort):
    seven=pd.read_csv(EXP/'unified/predictions_matbench.csv')
    eqv=pd.read_csv(EXP/'unified/predictions_eqv3_final40.csv')
    eqv=eqv[eqv.checkpoint_selection.eq('best')]
    base=pd.read_csv(ROOT/'experiment11_20260928/confirmation/eqv3_base.csv').rename(columns={'base':'prediction'})
    values={}; provenance=[]
    for seed in SEEDS:
        values['SevenNet','base',seed]=aligned(seven[(seven.method=='base')&(seven.seed==seed)],cohort)
        for route in ('U','A'):
            candidates=[]
            for lr in (1e-4,3e-4,1e-3):
                run=(ROOT/'experiment12_20260929_alignment/runs'/f'{route}_{seed}' if lr==3e-4 else EXP/'runs'/f'controls_{route}_{seed}_lr{lr:g}')
                history=pd.read_csv(run/'history.csv')
                best=history.loc[history.val_calibrated_mae.idxmin()]
                candidates.append((float(best.val_calibrated_mae),lr,int(best.epoch),run))
            score,lr,epoch,run=min(candidates,key=lambda item:(item[0],item[1]))
            checkpoint=(run/f'epoch{epoch:03d}.pt') if lr==3e-4 else run/'best.pt'
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            method=f'{route}@{lr:g}'
            values['SevenNet',route,seed]=aligned(seven[(seven.method==method)&(seven.seed==seed)],cohort)
            provenance.append({'model':'SevenNet','route':route,'seed':seed,'selected_lr':lr,'epoch':epoch,
                'val_selection_mae':score,'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
                'predictions':str(EXP/'unified/predictions_matbench.csv'),'method_key':method,
                'post_calibration':'train-only composition ridge; no scalar energy slope'})
        values['EqV3','base',seed]=aligned(base,cohort)
        for route in ('direct','clean'):
            run=EXP/'eqv3_final40'/route/f'seed{seed}'
            prefix=f'final40_{route}_s{seed}'
            history=pd.read_csv(run/f'{prefix}_history.csv')
            best=history.loc[history.val_mae_mev_atom.idxmin()]
            summary=json.loads((run/f'{prefix}_summary.json').read_text())
            if int(best.epoch)!=summary['best_epoch']:
                raise ValueError('EqV3 checkpoint does not match minimum validation MAE')
            checkpoint=run/f'{prefix}_best.pt'
            values['EqV3',route,seed]=aligned(eqv[(eqv.method==route)&(eqv.seed==seed)],cohort)
            provenance.append({'model':'EqV3','route':route,'seed':seed,'selected_lr':None,'epoch':int(best.epoch),
                'val_selection_mae':float(best.val_mae_mev_atom),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
                'predictions':str(EXP/'unified/predictions_eqv3_final40.csv'),'method_key':route,
                'post_calibration':'best-checkpoint train-only composition ridge; latest excluded'})
    pd.DataFrame(provenance).to_csv(OUT/'support/selected_checkpoints.csv',index=False)
    return values


def group_values(truth,base,prediction,groups):
    rows=[]
    for group in sorted(set(groups)):
        mask=groups==group
        target=truth[mask]-base[mask]; target-=target.mean()
        actual=prediction[mask]-base[mask]; actual-=actual.mean()
        error=prediction[mask]-truth[mask]; within=error-error.mean()
        cost=truth[mask][np.argmin(prediction[mask])]-truth[mask].min()
        rows.append([mask.sum(),np.sum(target**2),np.sum(actual**2),np.dot(target,actual),np.sum((actual-target)**2),
            np.sum(abs(within)),float(cost>.01),np.sum(error**2),mask.sum()*error.mean()**2,np.sum(within**2),np.sum(abs(error))])
    return np.array(rows,float)


def metrics(sums,group_count):
    values=np.asarray(sums)
    count,target_sq,actual_sq,dot,diff_sq,ec_abs,major,total_sq,bias_sq,within_sq,abs_error=np.moveaxis(values,-1,0)
    target_norm=np.sqrt(target_sq); actual_norm=np.sqrt(actual_sq)
    usable=target_norm/np.sqrt(count)>TARGET_RMS_FLOOR
    ratio=np.divide(actual_norm,target_norm,out=np.full_like(actual_norm,np.nan),where=usable)
    cosine=np.divide(dot,target_norm*actual_norm,out=np.full_like(actual_norm,np.nan),where=usable&(actual_norm>1e-14))
    correction={'target_norm_ev_atom':target_norm,'actual_norm_ev_atom':actual_norm,'norm_ratio':ratio,
        'direction_cosine':cosine,'correction_error_norm_ev_atom':np.sqrt(diff_sq),
        'target_rms_mev_atom':1000*target_norm/np.sqrt(count),'actual_rms_mev_atom':1000*actual_norm/np.sqrt(count),
        'ec_mae_mev_atom':1000*ec_abs/count,'major_wrong_rate':major/group_count,
        'mae_mev_atom':1000*abs_error/count}
    decomposition={'mse_total_ev2_atom2':total_sq/count,'mse_group_bias_ev2_atom2':bias_sq/count,
        'mse_within_ev2_atom2':within_sq/count,'identity_residual_ev2_atom2':(total_sq-bias_sq-within_sq)/count,
        'mae_mev_atom':1000*abs_error/count,'ec_mae_mev_atom':1000*ec_abs/count}
    return correction,decomposition


def row_with_intervals(identity,point,boot):
    result=dict(identity)
    for metric,value in point.items():
        array=np.asarray(boot[metric]); finite=array[np.isfinite(array)]
        result[metric]=float(value)
        result[f'{metric}_ci95_low']=float(np.quantile(finite,.025)) if len(finite) else None
        result[f'{metric}_ci95_high']=float(np.quantile(finite,.975)) if len(finite) else None
        result[f'{metric}_defined_bootstrap_fraction']=len(finite)/len(array)
    return result


def analyze(cohort,values):
    groups=cohort.exact_group.to_numpy(); truth=cohort.y_true.to_numpy()
    count=cohort.exact_group.nunique()
    draws=np.random.default_rng(SEED).integers(count,size=(DRAWS,count))
    rows_c,rows_d=[],[]; computed={}
    for (model,route,seed),prediction in values.items():
        table=group_values(truth,values[model,'base',seed],prediction,groups)
        point_c,point_d=metrics(table.sum(axis=0),count)
        boot_c,boot_d=metrics(table[draws].sum(axis=1),count)
        if abs(point_d['identity_residual_ev2_atom2'])>1e-12:
            raise AssertionError('MSE decomposition failed')
        identity={'row_type':'per_seed','model':model,'route':route,'seed':seed,'n_structures':len(cohort),'n_groups':count,
            'target_interpretable':bool(point_c['target_rms_mev_atom']>1000*TARGET_RMS_FLOOR),
            'direction_status':'defined' if np.isfinite(point_c['direction_cosine']) else 'undefined_zero_or_negligible_norm'}
        rows_c.append(row_with_intervals(identity,point_c,boot_c)); rows_d.append(row_with_intervals(identity,point_d,boot_d))
        computed[model,route,seed]=(point_c,point_d,boot_c,boot_d)
    for model,route in sorted(set((model,route) for model,route,_ in values)):
        packs=[computed[model,route,seed] for seed in SEEDS]
        means=[]
        for index in range(4):
            means.append({metric:np.mean([pack[index][metric] for pack in packs],axis=0) for metric in packs[0][index]})
        identity={'row_type':'seed_mean','model':model,'route':route,'seed':'all','n_structures':401,'n_groups':190}
        correction_row=row_with_intervals(identity,means[0],means[2])
        decomposition_row=row_with_intervals(identity,means[1],means[3])
        for metric in means[0]:
            correction_row[f'{metric}_seed_sd']=float(np.std([pack[0][metric] for pack in packs],ddof=1))
        for metric in means[1]:
            decomposition_row[f'{metric}_seed_sd']=float(np.std([pack[1][metric] for pack in packs],ddof=1))
        rows_c.append(correction_row); rows_d.append(decomposition_row)
    for model,left,right in [('SevenNet','U','A'),('EqV3','direct','clean')]:
        paired_points=[]; paired_boot=[]
        for seed in SEEDS:
            lhs= computed[model,left,seed]; rhs=computed[model,right,seed]
            points={metric:lhs[0][metric]-rhs[0][metric] for metric in lhs[0]}
            boot={metric:lhs[2][metric]-rhs[2][metric] for metric in lhs[2]}
            paired_points.append(points); paired_boot.append(boot)
            identity={'row_type':'paired_difference','model':model,'route':f'{left}-{right}','seed':seed,'n_structures':401,'n_groups':190}
            rows_c.append(row_with_intervals(identity,points,boot))
        points={metric:np.mean([point[metric] for point in paired_points]) for metric in paired_points[0]}
        boot={metric:np.mean([block[metric] for block in paired_boot],axis=0) for metric in paired_boot[0]}
        rows_c.append(row_with_intervals({'row_type':'paired_seed_mean','model':model,'route':f'{left}-{right}','seed':'all','n_structures':401,'n_groups':190},points,boot))
    pd.DataFrame(rows_c).to_csv(OUT/'correction_analysis.csv',index=False)
    pd.DataFrame(rows_d).to_csv(OUT/'error_decomposition.csv',index=False)


def configuration_cases(cohort,values):
    rows=[]
    for seed in SEEDS:
        for group,block in cohort.groupby('exact_group',sort=True):
            indices=block.index.to_numpy(); truth=block.y_true.to_numpy()
            row={'seed':seed,'group':group,'n':len(block),'formula':block.formula.iloc[0],
                 'true_minimum_ids':json.dumps(block.loc[np.abs(block.y_true-block.y_true.min())<=1e-12,'id'].tolist()),
                 'group_spread_mev':1000*float(np.ptp(truth)),
                 'structure_and_label_records':json.dumps(block[['id','cif_path','cif_sha256','y_true','source','label_source']].to_dict('records'))}
            for model,routes in [('SevenNet',('base','U','A')),('EqV3',('base','direct','clean'))]:
                for route in routes:
                    prediction=values[model,route,seed][indices]
                    chosen=int(np.argmin(prediction)); prefix=f'{model}_{route}'
                    cost=1000*float(truth[chosen]-truth.min())
                    row[f'{prefix}_chosen_id']=block.id.iloc[chosen]
                    row[f'{prefix}_cost_mev']=cost
                    row[f'{prefix}_exact_correct']=cost<=1e-9
                    row[f'{prefix}_major_wrong']=cost>10
            wrong_u=not row['SevenNet_U_exact_correct']; wrong_a=not row['SevenNet_A_exact_correct']
            row.update({'U_wrong_A_right':wrong_u and not wrong_a,'both_wrong':wrong_u and wrong_a,
                'A_wrong_U_right':wrong_a and not wrong_u,'both_right':not wrong_u and not wrong_a})
            rows.append(row)
    result=pd.DataFrame(rows); result['illustrative_example']=False
    for category in ('U_wrong_A_right','both_wrong','A_wrong_U_right'):
        candidates=result[(result.seed==SEEDS[0])&result[category]].copy()
        candidates['ranking_cost']=candidates[['SevenNet_U_cost_mev','SevenNet_A_cost_mev']].max(axis=1)
        if len(candidates):
            chosen=candidates.sort_values(['ranking_cost','group'],ascending=[False,True]).index[0]
            result.loc[chosen,'illustrative_example']=True
    result['case_selection_rule']='seed20260528; one group per discordance category with largest max(U,A) cost, tie by group; descriptive only'
    result.to_csv(OUT/'configuration_cases.csv',index=False)


def main():
    (OUT/'support').mkdir(parents=True,exist_ok=True)
    cohort=load_cohort(); values=predictions(cohort)
    analyze(cohort,values); configuration_cases(cohort,values)
    input_files=[EXP/'unified/evaluation_manifest.csv',EXP/'unified/predictions_matbench.csv',EXP/'unified/predictions_eqv3_final40.csv',
        ROOT/'experiment11_20260928/confirmation/eqv3_base.csv',EXP/'unified/lr_selection_by_validation.csv',
        EXP/'eqv3_fitted_reference.csv',EXP/'eqv3_fitted_offset_coefficients.npz']
    settings={'draws':DRAWS,'seed':SEED,'bootstrap_unit':'whole cleaned exact-composition group; identical draws for all methods/seeds',
        'seed_mean':'arithmetic mean of per-seed statistics, group bootstrap conditional on three archived seeds; not pooled independent samples',
        'energy_unit':'eV/atom; norms in eV/atom; MSE in (eV/atom)^2; MAE in meV/atom',
        'major_wrong_threshold_mev':10,'exact_tie_cost_tolerance_mev':1e-9,'target_rms_floor_ev_atom':TARGET_RMS_FLOOR,
        'mask':'relative_cohort & keep, then remove post-cleaning singleton groups, then center',
        'reference_models':'frozen calibrated base for each backbone; EqV3 historical adapted residual is NOT base',
        'training':'none; no optimization updates or overwriting reference fits',
        'inputs':{str(path):digest(path) for path in input_files}}
    (OUT/'support/provenance.json').write_text(json.dumps(settings,indent=2))
    print(pd.read_csv(OUT/'correction_analysis.csv').query("row_type=='seed_mean'")[['model','route','norm_ratio','direction_cosine','ec_mae_mev_atom','major_wrong_rate']].to_string(index=False))


if __name__=='__main__':
    main()
