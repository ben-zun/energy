from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure,Composition

from analysis import ROOT,EXP,OUT,digest


def main():
    initialization=pd.read_csv(OUT/'initialization_audit.csv')
    correction=pd.read_csv(OUT/'correction_analysis.csv')
    decomposition=pd.read_csv(OUT/'error_decomposition.csv')
    cases=pd.read_csv(OUT/'configuration_cases.csv')
    cohort=pd.read_csv(OUT/'support/cohort.csv',keep_default_na=False)
    metadata=json.loads((OUT/'support/provenance.json').read_text())
    for path,checksum in metadata['inputs'].items():
        assert digest(path)==checksum
    assert len(cohort)==401 and cohort.exact_group.nunique()==190
    assert (cohort.groupby('exact_group').size()>=2).all()
    assert not cohort.id.duplicated().any()
    assert len(initialization)==6
    assert initialization.no_parameter_updates.all()
    assert initialization.weights_equal_between_routes.all() and initialization.output_heads_equal.all()
    assert initialization.offset_group_spread_ev_atom.max()<1e-9
    assert initialization.identity_max_ev_atom.max()<1e-10
    assert initialization.reference_train_only_archive_fit_verified.all()
    assert not initialization.reference_fitted_to_native_initialization.any()
    train_state=json.loads((OUT/'support/initialization_state_train.json').read_text())
    val_state=json.loads((OUT/'support/initialization_state_val.json').read_text())
    assert train_state['route_states'][0]['weight_hash']==val_state['route_states'][0]['weight_hash']
    config=train_state['config']
    initialization_inputs=[Path(config['checkpoint']),Path(config['reference_csv']),
        ROOT/'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration/calibrated_predictions.csv',
        Path(config['dataset_root'])/'train/train.csv',Path(config['dataset_root'])/'val/val.csv',
        ROOT/'rclfa_v3/scripts/run_eqv3_native_upper_bound_smoke_20260725.py',
        ROOT/'rclfa_v3/eqv3_native.py',EXP/'code/p06_eqv3_alignment_gate.py',EXP/'code/p08_error_decomposition.py',EXP/'code/p09_alignment_start.py']
    (OUT/'support/initialization_input_hashes.json').write_text(json.dumps({str(path):digest(path) for path in initialization_inputs},indent=2))
    for seed in (20260529,20260530):
        state=json.loads((OUT/'support'/f'initialization_weights_seed{seed}.json').read_text())
        assert state['same_initial_weights'] and state['same_output_head']
        assert all(route['lora_B_abs_max']==0 for route in state['routes'])
    assert len(cases)==570 and set(cases.groupby('seed').size())=={190}
    assert len(correction[correction.row_type.eq('per_seed')])==18
    assert set(correction.n_structures)=={401} and set(correction.n_groups)=={190}
    np.testing.assert_allclose(decomposition.mse_total_ev2_atom2,
        decomposition.mse_group_bias_ev2_atom2+decomposition.mse_within_ev2_atom2,atol=1e-12,rtol=0)
    point=correction[correction.row_type.eq('per_seed')]
    for row in point.itertuples(index=False):
        rates=cases[cases.seed.eq(int(row.seed))][f'{row.model}_{row.route}_major_wrong'].mean()
        assert abs(rates-row.major_wrong_rate)<1e-12
    for model,left,right in [('SevenNet','U','A'),('EqV3','direct','clean')]:
        for seed in (20260528,20260529,20260530):
            selection=point[(point.model==model)&(point.seed.astype(str)==str(seed))].set_index('route')
            paired=correction[(correction.row_type=='paired_difference')&(correction.model==model)&(correction.seed.astype(str)==str(seed))].iloc[0]
            assert abs(paired.ec_mae_mev_atom-(selection.loc[left,'ec_mae_mev_atom']-selection.loc[right,'ec_mae_mev_atom']))<1e-10
    source_rows=[]
    for row in cases[cases.illustrative_example].itertuples(index=False):
        for entry in json.loads(row.structure_and_label_records):
            structure=Structure.from_file(entry['cif_path'])
            record=cohort.set_index('id').loc[entry['id']]
            assert len(structure)==int(record.natoms)
            assert structure.composition.reduced_composition==Composition(record.formula).reduced_composition
            assert digest(entry['cif_path'])==entry['cif_sha256']
            source_rows.append({'id':entry['id'],'group':row.group,'natoms':len(structure),
                'cif_sha256':entry['cif_sha256'],'label_source':entry['label_source'],'verified':True})
    pd.DataFrame(source_rows).to_csv(OUT/'support/case_source_verification.csv',index=False)
    report_init=initialization[initialization['mode'].eq('eval')][['route','split','n','native_mae_mev_atom','native_rmse_mev_atom','actual_smooth_l1','residual_mean_ev_atom','saturated_fraction']]
    summary=correction[correction.row_type.eq('seed_mean')][['model','route','target_norm_ev_atom','actual_norm_ev_atom','norm_ratio','direction_cosine','correction_error_norm_ev_atom','mae_mev_atom','ec_mae_mev_atom','major_wrong_rate']]
    paired=correction[correction.row_type.eq('paired_seed_mean')][['model','route','correction_error_norm_ev_atom','correction_error_norm_ev_atom_ci95_low','correction_error_norm_ev_atom_ci95_high','ec_mae_mev_atom','ec_mae_mev_atom_ci95_low','ec_mae_mev_atom_ci95_high','major_wrong_rate','major_wrong_rate_ci95_low','major_wrong_rate_ci95_high']]
    mse=decomposition[decomposition.row_type.eq('seed_mean')][['model','route','mse_total_ev2_atom2','mse_group_bias_ev2_atom2','mse_within_ev2_atom2']]
    contract=json.loads((OUT/'support/initialization_contract.json').read_text())
    report=f'''# 已有预测与checkpoint核查（不新增训练）

## 决定

本轮完成要求的三项检查，四份主CSV已生成。**暂不启动组成偏移训练。**
SevenNet未对齐U比A有更大的最终组内修正误差，整组配对区间支持这一方向；
但EqV3 clean的参考没有对应训练器实际原生初始化输出，有效性门槛未通过。
应先独立修复并复核这个输出契约；不能将现有clean结果解释为“更好起点反而适配更差”。
原参考CSV、系数与旧预测均未覆盖，也没有以重新拟合的新参考替换旧值。

## 1. EqV3第0轮

完整8192 train与1024 val前向，seed20260528；另外两个种子验证路线间初始参数、输出头一致且LoRA B=0，因而初始函数一致。
另保存train模式、无参数更新的完整8192样本损失。这里不是用反归一化归档列替代live模型。

{report_init.to_markdown(index=False,floatfmt='.6g')}

Smooth L1沿用原训练器beta=0.05，作用在**总能量/scale_divisor**上；scale=1，
不是直接对每原子能量误差使用0.05阈值。饱和率指实际损失残差绝对值≥beta。
绝大多数样本已在Huber线性区，不能仅凭误差增大推断参数梯度按比例增大。

参数和输出头哈希通过；单位N转换、clean−direct=d以及组内d恒定通过。
参考确实由归档train残差拟合：岭回归正规方程最大残差{contract['archived_train_fit_normal_equation_max']:.3g}。
但归档e0是经过rmsd与元素参考反归一化的能量，原生训练器直接使用未反归一化的network输出。
两者混用，导致clean起点显著恶化。归档还包含均值填充的缺失预测：
train {contract['mappings']['train']['archive_filled_prediction_unique_ids']}个，val {contract['mappings']['val']['archive_filled_prediction_unique_ids']}个。
这些不能当成真实初始化预测。反归一化回放的剩余数值差异和填充影响在support/initialization_contract.json中保留。

## 2. 同一清洗队列上的实际修正

先取relative_cohort&keep（411行），再剔除清洗后单例组，最终190组/401行；随后才组内中心化。
SevenNet U/A由各自val校准MAE选学习率，均选0.001；EqV3只使用final40的val-selected best，不混latest。
EqV3基准是真正冻结基准，不是历史已适配的residual模型。

{summary.to_markdown(index=False,floatfmt='.6g')}

范数单位eV/atom（401维向量的L2范数），不是每结构RMS；后者另有列。
目标RMS≤1e−6 eV/atom时比例与方向不可解释；零实际修正的方向同样为空。
所有相对指标在同一401行、同一分组上计算。重大选错率定义为选择代价>10 meV/atom。

配对结果为左方法−右方法，10000次完整组成组重采样，各方法、各seed使用同一draw。
三seed汇总是逐seed统计量均值，不将570个组×seed观测当作独立样本。

{paired.to_markdown(index=False,floatfmt='.6g')}

SevenNet U的范数比更大且方向一致性更低，但余弦仍为正，不能简单写成“方向反了”。
最终修正误差与材料选错率的配对差异共同支持U偏离目标更多。
A仍未在所有相对指标上优于base；这里不能将较小范数自动当作机制成功。
EqV3 direct/clean结果可描述，但在第0轮失配排除之前，不能用来证明预对齐无效。

## 3. MSE可加分解

同一队列、同一误差定义，单位(eV/atom)^2：

{mse.to_markdown(index=False,floatfmt='.8g')}

逐seed验证total=bias+within；MAE不使用此可加分解。L2/MSE改善也不保证EC-MAE改善。

## 4. 材料后果与来源

configuration_cases.csv含190组×3seed=570行，每行列出六种方案选择的ID、真实最低能ID（含并列）、
选择代价、重大选错标记和SevenNet U错/A对、两者错、A错/U对。
“对”用真实最低能精确命中（1e−9 meV数值容差）；重大选错另用10 meV阈值。
公开案例规则：主seed每个不一致类别选max(U,A)代价最大的一个组，并以组名打破并列。
案例只是图示，不推断总体发生率。选中结构的CIF原子数、组成、文件hash和原始标签文件已核对。
磁态等缺失元数据仍标为UNAVAILABLE，不虚构材料机理。

## 文件与复现

- initialization_audit.csv：第0轮误差、真实损失输入、参考来源和检查标志。
- correction_analysis.csv：每seed、seed均值、配对差值及组bootstrap区间。
- error_decomposition.csv：MSE组偏差/组内分解与恒等式残差。
- configuration_cases.csv：逐组选择与案例。
- support/selected_checkpoints.csv：实际checkpoint路径、hash、val选择规则与预测源。
- support/cleaning_mask.csv、cohort.csv：全清洗mask与最终401个ID。
- support/provenance.json：输入hash、单位、统计设置。
- support/initialization_rows_*.csv：全部原生前向和逐样本损失。

## 代码核查结论

p08不能原样复用：清洗前中心化、保留清洗后单例、把EqV3 residual当base；本轮已纠正。
p09仅重算归档输出，无法代表实际第0轮。本轮用未更新参数的live输出核查。
p10默认只查1024训练样本，其live输出与归档输出不符时会抛异常，不能提供完整失败核查记录；
本轮覆盖全部8192训练样本、1024验证样本，使用原collate的真实target_scaled，并保留失败事实。

证据自审：主要发现有对应输入与恒等式；参考有效性失败如实报告；保留阴性结果；
不将案例当统计证据；不启动任何新的训练。
'''
    (OUT/'AUDIT_REPORT.md').write_text(report)
    checks={'execution_checks':'passed','scientific_initialization_validity':'failed','no_new_training':True,
        'initialization_rows':len(initialization),'correction_rows':len(correction),'decomposition_rows':len(decomposition),
        'configuration_rows':len(cases),'groups':190,'structures':401,
        'mse_identity_max_abs':float(abs(decomposition.identity_residual_ev2_atom2).max()),
        'reference_preserved_sha256':digest(EXP/'eqv3_fitted_reference.csv'),
        'coefficient_preserved_sha256':digest(EXP/'eqv3_fitted_offset_coefficients.npz'),
        'outputs':{name:digest(OUT/name) for name in ['initialization_audit.csv','correction_analysis.csv','error_decomposition.csv','configuration_cases.csv']}}
    (OUT/'support/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
