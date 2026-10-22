from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import ROOT,OUT,sha,group_metrics


def cluster_bootstrap(errors,groups,draws):
    sums=pd.DataFrame({'e':errors,'g':groups}).groupby('g').e.agg(['sum','count'])
    return sums['sum'].to_numpy()[draws].sum(axis=1)/sums['count'].to_numpy()[draws].sum(axis=1)


def main():
    (OUT/'analysis').mkdir(exist_ok=True)
    lock=json.loads((OUT/'LOCK.json').read_text())
    metadata=pd.read_csv(OUT/'cache/metadata.csv',keep_default_na=False).set_index('id')
    fractions=np.load(OUT/'cache/fractions.npy')
    fraction_frame=pd.DataFrame(fractions,index=metadata.index)
    primary=pd.read_csv(ROOT/'experiment15_prediction_audit_20261001/support/cohort.csv',keep_default_na=False).sort_values(['exact_group','id'])
    manifest=pd.read_csv(ROOT/'experiment11_20260928/confirmation/manifest.csv',keep_default_na=False)
    scopes={'relative_clean401':primary.id.tolist(),'overall2048':manifest[manifest.overall_cohort].id.tolist()}
    rows=[]; predictions={}; selection=[]; native_invariance=[]
    for seed in lock['seeds']:
        hashes=[]
        for route in lock['routes']:
            run=OUT/'runs'/route/f'seed{seed}'
            status=json.loads((run/'status.json').read_text())
            history=pd.read_csv(run/'history.csv')
            assert len(history)==40 and status['epochs']==40
            assert int(history.loc[history.val_mae_mev.idxmin(),'epoch'])==status['best_epoch']
            hashes.append(status['initial_weight_hash'])
            train=pd.read_csv(run/'best_train_predictions.csv').set_index('id')
            confirmation=pd.read_csv(run/'best_confirmation_predictions.csv').set_index('id')
            assert len(train)==8192 and len(confirmation)==2436
            assert not set(train.index)&set(confirmation.index)
            fit=Ridge(alpha=1e-6,fit_intercept=True).fit(fraction_frame.loc[train.index],train.y_true-train.prediction)
            joblib.dump(fit,run/'post_calibration.joblib')
            calibrated=confirmation.prediction+fit.predict(fraction_frame.loc[confirmation.index])
            result=confirmation.copy(); result['native_prediction']=confirmation.prediction
            result['calibrated_prediction']=calibrated; result.to_csv(run/'confirmation_calibrated.csv')
            selection.append({'seed':seed,'route':route,'best_epoch':status['best_epoch'],'val_native_mae_mev':status['best_val_mae_mev'],
                'checkpoint':str(run/'best.pt'),'checkpoint_sha256':sha(run/'best.pt'),'post_calibration_fit_n':len(train)})
            for calibration,column in [('native','native_prediction'),('calibrated','calibrated_prediction')]:
                for scope,ids in scopes.items():
                    block=result.loc[ids]
                    truth=metadata.loc[ids,'y_true'].to_numpy(float)
                    np.testing.assert_allclose(block.y_true.to_numpy(),truth,atol=1e-12)
                    prediction=block[column].to_numpy(float)
                    groups=metadata.loc[ids,'group'].to_numpy()
                    metrics=group_metrics(truth,prediction,groups)
                    if scope=='overall2048':
                        metrics['ec_mae_mev']=None; metrics['major_wrong_rate']=None
                    rows.append({'seed':seed,'route':route,'calibration':calibration,'scope':scope,'n':len(ids),**metrics})
                    predictions[seed,route,calibration,scope]=(truth,prediction,groups)
            native=predictions[seed,route,'native','relative_clean401']
            corrected=predictions[seed,route,'calibrated','relative_clean401']
            native_error=pd.Series(native[1]-native[0]); corrected_error=pd.Series(corrected[1]-corrected[0]); groups=native[2]
            difference=(native_error-native_error.groupby(groups).transform('mean'))-(corrected_error-corrected_error.groupby(groups).transform('mean'))
            assert np.max(abs(difference))<1e-10
            native_invariance.append({'seed':seed,'route':route,'centered_calibration_effect_max':float(np.max(abs(difference)))})
        assert hashes[0]==hashes[1]
    summary=pd.DataFrame(rows); summary.to_csv(OUT/'analysis/per_seed_metrics.csv',index=False)
    aggregate=summary.groupby(['scope','calibration','route'])[['mae_mev','ec_mae_mev','major_wrong_rate']].agg(['mean','std'])
    aggregate.columns=['_'.join(names) for names in aggregate.columns]
    aggregate=aggregate.reset_index(); aggregate.to_csv(OUT/'analysis/summary.csv',index=False)
    pd.DataFrame(selection).to_csv(OUT/'analysis/selected_checkpoints.csv',index=False)
    pd.DataFrame(native_invariance).to_csv(OUT/'analysis/post_calibration_invariance.csv',index=False)
    paired=[]
    for calibration in ('native','calibrated'):
        for scope in scopes:
            all_boot={}; all_point={}
            for seed in lock['seeds']:
                truth,left,groups=predictions[seed,'clean',calibration,scope]
                _,right,other=predictions[seed,'direct',calibration,scope]
                assert np.array_equal(groups,other)
                unique_groups=np.unique(groups)
                draws=np.random.default_rng(20261001).integers(len(unique_groups),size=(10000,len(unique_groups)))
                error_left=left-truth; error_right=right-truth
                candidates={'absolute_mae':(np.abs(error_left)-np.abs(error_right))*1000}
                if scope=='relative_clean401':
                    center=lambda values:values-pd.Series(values).groupby(groups).transform('mean').to_numpy()
                    candidates['ec_mae']=(np.abs(center(error_left))-np.abs(center(error_right)))*1000
                for metric,delta in candidates.items():
                    distribution=cluster_bootstrap(delta,groups,draws)
                    point=float(delta.mean())
                    paired.append({'seed':seed,'scope':scope,'calibration':calibration,'metric':metric,
                        'clean_minus_direct':point,'ci95_low':float(np.quantile(distribution,.025)),
                        'ci95_high':float(np.quantile(distribution,.975)),'groups':len(unique_groups)})
                    all_boot.setdefault(metric,[]).append(distribution); all_point.setdefault(metric,[]).append(point)
                if scope=='relative_clean401':
                    flags=[]
                    for group in unique_groups:
                        mask=groups==group
                        cost_left=truth[mask][np.argmin(left[mask])]-truth[mask].min()
                        cost_right=truth[mask][np.argmin(right[mask])]-truth[mask].min()
                        flags.append(float(cost_left>.01)-float(cost_right>.01))
                    flags=np.array(flags); distribution=flags[draws].mean(axis=1); point=float(flags.mean())
                    paired.append({'seed':seed,'scope':scope,'calibration':calibration,'metric':'major_wrong_rate',
                        'clean_minus_direct':point,'ci95_low':float(np.quantile(distribution,.025)),
                        'ci95_high':float(np.quantile(distribution,.975)),'groups':len(unique_groups)})
                    all_boot.setdefault('major_wrong_rate',[]).append(distribution); all_point.setdefault('major_wrong_rate',[]).append(point)
            for metric in all_boot:
                distribution=np.mean(all_boot[metric],axis=0)
                paired.append({'seed':'mean_of_three','scope':scope,'calibration':calibration,'metric':metric,
                    'clean_minus_direct':float(np.mean(all_point[metric])),'ci95_low':float(np.quantile(distribution,.025)),
                    'ci95_high':float(np.quantile(distribution,.975)),'groups':len(unique_groups)})
    pd.DataFrame(paired).to_csv(OUT/'analysis/paired_bootstrap.csv',index=False)
    report=f'''# EqV3完整物理解码匹配训练结果

六个全新训练：direct/clean × 三seed，均为40 epochs；没有按结果方向追加或删减训练。
两路线相同预训练权重和LoRA初始化，physical total SmoothL1，只有clean加入固定组成参考。
模型选择只用原validation原生每原子MAE；最佳模型的后处理组成校准只拟合train。
物理损失使用float64标量运算以降低总能量相消误差；网络两路线均bfloat16。
梯度等价检查、重复反向数值噪声和初始化组内不变检查见各run/gate.json。

## 三seed均值与标准差

{aggregate.to_markdown(index=False,floatfmt='.5f')}

## 配对组成组bootstrap

差值为clean−direct。负值表示clean更好。按整组成组重采样10000次；
总体MAE包含单例组，relative只用已清洗190组/401结构。均值CI对三个已运行seed条件化，不将seed×结构当独立样本。

{pd.DataFrame(paired).to_markdown(index=False,floatfmt='.5f')}

## 解释边界

不沿用旧EqV3负结果。新比较是否支持对齐必须分别看绝对、相对指标和配对CI。
第0轮更好不保证最终EC更好；总能量损失的饱和率不能直接与SevenNet每原子损失比较。
不因结果方向追加epochs、模型或人工组成偏移训练。
确认样本已在先前研究中查看，本轮冻结用于评价且不参与拟合选择，不宣称全新盲测。
'''
    (OUT/'FINAL_REPORT.md').write_text(report)
    print(aggregate.to_string(index=False))


if __name__=='__main__':
    main()
