"""Build PROTOCOL.md and FINAL_COMPARISON.md from the recorded results.

Deliberately free of torch/pymatgen imports so the report logic can be rehearsed offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# mirrored from alignment_common to keep this module import-light
ENERGY = Path('/home/yxzhang/energy')
OUT = ENERGY / 'experiment12_20260929_alignment'
CALIBRATION_CACHE = (ENERGY / 'runs/rclfa_v3/real_fegx_foundation_replacement_272'
                     / 'reference_conversion/sevennet_reference/calibration_coefficients.csv')
PREFIXES = ('3_', '4_', 'reduce_input_to_hidden', 'reduce_hidden_to_energy',
            'rescale_atomic_energy')
TRAINABLE_PARAMETERS = 301300
ROUTES = ('U', 'A')
SEEDS = (20260528, 20260529, 20260530)
BATCH_TRAIN, BATCH_EVAL = 16, 32
LR, WEIGHT_DECAY, HUBER_BETA = 3e-4, 1e-5, 0.05
MAX_EPOCHS = 120
CALIBRATION_ALPHA = 1e-6
BUDGETS = (20, 40, 60, 80, 120)


def run_dir(route, seed, root=None):
    return (Path(root) if root else OUT) / 'runs' / f'{route}_{seed}'


def table(header, body):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
                     + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body])


def label(row):
    if row.comparison == 'same_budget':
        return f'截至 {int(row.budget)} epoch 最佳'
    if row.comparison == 'final_best':
        return '120 epoch 内最佳'
    return f'首次达到 {float(row.threshold_mev):.3f} meV'


def fmt(value, digits=2):
    return '—' if value is None or (isinstance(value, float) and not np.isfinite(value)) else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(OUT))
    args = parser.parse_args()
    root = Path(args.root)
    results = root / 'results'
    history = pd.read_csv(results / 'training_history.csv')
    selected = pd.read_csv(results / 'selected_checkpoints.csv')
    metrics = pd.read_csv(results / 'final_metrics.csv')
    paired = pd.read_csv(results / 'paired_differences.csv')
    tail = pd.read_csv(results / 'tail20_trend.csv')

    key = ['seed', 'comparison', 'criterion', 'route', 'epoch']
    selected = selected.drop_duplicates(subset=key)
    metrics = metrics.drop_duplicates(subset=key)
    merged = selected.merge(metrics, on=key, how='left', validate='one_to_one')
    merged['label'] = merged.apply(label, axis=1)

    run_info = {f'{route}_{seed}': json.loads((run_dir(route, seed, root) / 'status.json').read_text())
                for route in ROUTES for seed in SEEDS}

    # ---- required final table -------------------------------------------------
    rows = []
    for _, r in merged[merged.comparison.isin(['same_budget', 'final_best'])].iterrows():
        rows.append([r.seed, r.label, r.route, int(r.epoch),
                     fmt(r.overall_abs_mae_calibrated), fmt(r.relative_abs_mae_calibrated),
                     fmt(r.relative_ec_mae)])
    main_table = table(['seed', '预算／匹配方式', '路线', '选中epoch', '总体绝对MAE',
                        '相对队列绝对MAE', '相对队列EC-MAE'], rows)

    native_rows = []
    for _, r in merged[merged.comparison == 'final_best'].iterrows():
        native_rows.append([r.seed, r.route, int(r.epoch), fmt(r.overall_abs_mae_native),
                            fmt(r.overall_abs_mae_calibrated),
                            fmt(r.relative_abs_mae_native), fmt(r.relative_abs_mae_calibrated),
                            fmt(r.relative_ec_mae), f"{r.ec_mae_calibration_shift_mev:.2e}"])
    native_table = table(['seed', '路线', 'epoch', '总体MAE(原生)', '总体MAE(后校准)',
                          '相对绝对MAE(原生)', '相对绝对MAE(后校准)', '相对EC-MAE', 'EC 校准偏移'],
                         native_rows)

    # ---- class 3 --------------------------------------------------------------
    class3 = merged[merged.comparison == 'matched_accuracy']
    class3_rows = []
    for (seed, criterion), block in class3.groupby(['seed', 'criterion']):
        if set(block.route) != set(ROUTES):
            continue
        u = block[block.route == 'U'].iloc[0]
        a = block[block.route == 'A'].iloc[0]
        gap = abs(float(u.val_calibrated_mae) - float(a.val_calibrated_mae))
        verdict = '可称相近' if gap <= 1.0 else ('差距明显' if gap > 2.0 else '略差')
        class3_rows.append([seed, f"{float(u.threshold_mev):.3f}",
                            f"{int(u.epoch)} / {fmt(u.val_calibrated_mae, 3)}",
                            f"{int(a.epoch)} / {fmt(a.val_calibrated_mae, 3)}",
                            fmt(gap, 3), verdict])
    class3_table = table(['seed', '阈值 (meV)', 'U 首次达到 (epoch / val MAE)', 'A 首次达到 (epoch / val MAE)',
                          '实际验证MAE差距', '判定'], class3_rows)

    # ---- paired intervals -----------------------------------------------------
    metric_label = {'overall_abs_mae_calibrated': '总体绝对MAE',
                    'relative_abs_mae_calibrated': '相对队列绝对MAE',
                    'relative_ec_mae': '相对队列EC-MAE'}
    def criterion_label(comparison, criterion):
        if comparison == 'same_budget':
            return f"截至 {int(criterion.split('_')[1])} epoch 最佳"
        if comparison == 'final_best':
            return '120 epoch 内最佳'
        return f"首次达到 {float(criterion.split('_')[-1]):.3f} meV"
    paired_rows = []
    for _, r in paired.sort_values(['seed', 'comparison', 'criterion', 'metric']).iterrows():
        paired_rows.append([r.seed, criterion_label(r.comparison, r.criterion),
                            metric_label.get(r.metric, r.metric),
                            fmt(r.mean_u_mev, 3), fmt(r.mean_a_mev, 3),
                            fmt(r.difference_u_minus_a_mev, 3),
                            f"[{fmt(r.ci95_low, 3)}, {fmt(r.ci95_high, 3)}]",
                            '不含 0' if (r.ci95_low > 0 or r.ci95_high < 0) else '含 0'])
    paired_table = table(['seed', '匹配方式', '指标', 'U 均值', 'A 均值', 'U − A (meV)', '95% CI', '区间'],
                         paired_rows)

    tail_rows = [[r.route, r.seed, fmt(r.tail20_head_mean, 3), fmt(r.tail20_tail_mean, 3),
                  fmt(r.change_mev, 3), '仍在下降' if r.still_descending else '已走平/回升']
                 for _, r in tail.iterrows()]
    tail_table = table(['路线', 'seed', '最近20轮前5轮均值', '最近20轮后5轮均值', '变化', '判定'], tail_rows)

    init_checks = run_info['A_20260528']['init_checks']
    protocol = f"""# PROTOCOL — U vs A 匹配长训练

工程：`{root}`；数据/模型/校准来源均只读引用既有实验，不覆盖任何已有产物。

## 1. 两条路线

令 $f_\\theta(X)$ 为 SevenNet 输出总能量，$N_X$ 为原子数，$d(z)$ 为冻结的元素参考校准。

| 路线 | 进入训练损失的预测 | 脚本参数 |
|---|---|---|
| **U**（训练前不对齐） | $\\widehat y_U = f_\\theta(X)/N_X$ | `--alignment none` |
| **A**（训练前对齐） | $\\widehat y_A = f_\\theta(X)/N_X + d(z)$ | `--alignment element_reference` |

两者都与同一个单位原子形成能标签 $y$（`target_per_atom`，eV/atom）计算 Smooth L1 损失。

$d(z) = -(\\mathbf{{x}}^\\mathsf{{T}}\\mu_Z + b)$，其中 $\\mu_Z$ 与 $b$ 取自
`{CALIBRATION_CACHE}` 的 `element_reference` 块（118 个元素系数 + 1 个截距，**无尺度项**，
原 train 拟合、冻结不更新）。A 不需要额外加载冻结模型——它只在预测式里加一个常数偏移。

## 2. 训练配置（两条路线完全一致）

| 项目 | 设置 |
|---|---|
| 模型 | 同一个 `7net-0` checkpoint（`load_sevennet_model(remove_force_output=True)`） |
| 数据 | 原 train 8192、val 1024；test/确认集不参与选择 |
| 可训练参数 | `{list(PREFIXES)}`，断言 {TRAINABLE_PARAMETERS} 个 |
| 优化器 | AdamW，lr {LR}，weight decay {WEIGHT_DECAY} |
| 损失 | Smooth L1，beta={HUBER_BETA}，单位 eV/atom |
| batch | 训练 {BATCH_TRAIN}，评估 {BATCH_EVAL} |
| 最大 epoch | {MAX_EPOCHS}，**不启用提前停止** |
| 随机种子 | {', '.join(map(str, SEEDS))} |
| 学习率调度 | 无 |
| 初始化与 batch 顺序 | 同 seed 下 U/A 完全一致（同一 `torch.Generator().manual_seed(seed)`） |

## 3. 统一后校准与 checkpoint 选择

每个 epoch 保存两套预测：

- **原生预测**：上表中实际进入训练损失的输出；
- **统一后校准预测**：$\\widehat y^{{cal}}_t = \\widehat y_t + q_t(z)$，
  $q_t(z) = \\mathbf{{x}}^\\mathsf{{T}}\\mathbf{{w}}_t + b_t$。

$q_t$ 的拟合规则（U/A 完全相同）：

- 每个 epoch 只用 **train** 的残差 $y - \\widehat y_t$ 拟合；
- 特征 = 118 维原子分数 + 截距；Ridge $\\alpha = {CALIBRATION_ALPHA}$，截距不惩罚；
- 不额外拟合能量尺度；**不参与反向传播**（在 `torch.no_grad()` 之外用 sklearn 拟合，梯度图上不存在它）。

主 checkpoint 统一选取 **后校准 val 绝对 MAE 最低** 的 epoch，并列时选较早者。
原生 val MAE 同时保存用于解释训练过程。线性组成后校准不改变同组成组内的相对能量，
因此 `val_ec_mae` 与 `val_native_ec_mae` 应当相等（实测最大偏移见 `final_metrics.csv` 的
`ec_mae_calibration_shift_mev`）。

## 4. 数据用途

| 集合 | 用途 |
|---|---|
| 原 train（8192） | 训练 + 后校准拟合 |
| 原 val（1024） | 唯一的 checkpoint 选择依据 |
| 原 test（2048） | 不参与选择；本轮不用于报告 |
| 冻结确认集（2436 = 总体 2048 + 相对 200 组/422） | 仅在选点冻结后统一预测，用于最终比较 |

## 5. 初始化不变量（每次运行开始时断言并记录）

| 检查 | 结果 |
|---|---|
| A 初始预测 == exp11 归档 base，medium 全 11264 行最大差 | {init_checks['A_initial_vs_exp11_base_max_abs_ev_atom']:.3e} eV/atom |
| A 初始预测 == 确认集归档 base，2436 行最大差 | {init_checks['A_initial_vs_confirmation_base_max_abs_ev_atom']:.3e} eV/atom |
| A − U == d(z) 最大差 | {init_checks['route_A_minus_route_U_equals_d_max_abs']:.3e} eV/atom |
| `route_prediction('U')` vs 原生输出最大差 | {init_checks['route_U_call_equals_raw_max_abs']:.3e} |
| `route_prediction('A')` vs 原生+d 最大差 | {init_checks['route_A_call_equals_raw_plus_d_max_abs']:.3e} |
| 总能量只除一次原子数（float32 舍入内） | max abs diff {init_checks['per_atom_single_division_max_abs_ev_atom']:.3e} eV/atom |
| d(z) 组内最大差（medium） | {init_checks['medium_d_within_group_max_spread_ev_atom']:.3e} eV/atom |
| d(z) 组内最大差（确认集） | {init_checks['confirmation_d_within_group_max_spread_ev_atom']:.3e} eV/atom |

## 6. 文件

```text
{root}/
  code/alignment_common.py           共享定义（d(z)、组成分数、per_atom、后校准）
  code/train_alignment_matched.py     U/A 统一训练脚本
  code/evaluate_alignment.py         三类比较 + 确认集评估 + 配对 bootstrap
  code/build_reports.py              生成本文件与 FINAL_COMPARISON.md
  runs/{{U,A}}_{{seed}}/history.csv              逐 epoch 曲线
  runs/{{U,A}}_{{seed}}/calibration_history.csv  逐 epoch 后校准参数（118 w + b）
  runs/{{U,A}}_{{seed}}/epoch*.pt                逐 epoch checkpoint
  runs/{{U,A}}_{{seed}}/val_epoch*.npz           逐 epoch val 原生/后校准预测
  runs/{{U,A}}_{{seed}}/status.json              状态与初始化不变量
  results/training_history.csv       六次运行合并曲线
  results/selected_checkpoints.csv   三类比较选出的 epoch
  results/final_metrics.csv          选中 checkpoint 的确认集指标
  results/paired_differences.csv     U−A 配对区间
  results/tail20_trend.csv           最近 20 轮趋势
```
"""

    # ---- conclusion summary ---------------------------------------------------
    final = merged[merged.comparison == 'final_best']
    timing_rows = []
    for route in ROUTES:
        for seed in SEEDS:
            block = history[(history.route == route) & (history.seed == seed)]
            timing_rows.append([route, seed, len(block),
                                f'{block.training_time_seconds.median():.1f}',
                                f'{block.evaluation_time_seconds.median():.1f}',
                                f'{(block.training_time_seconds.sum() + block.evaluation_time_seconds.sum()) / 60:.1f}'])
    timing_table = table(['路线', 'seed', 'epochs', 'per-epoch 训练 (s)', 'per-epoch 评估 (s)',
                          '总耗时 (min)'], timing_rows)
    lines = []
    for seed in SEEDS:
        u = final[(final.seed == seed) & (final.route == 'U')].iloc[0]
        a = final[(final.seed == seed) & (final.route == 'A')].iloc[0]
        lines.append(f"- seed {seed}: U 选 epoch {int(u.epoch)}（val {u.val_calibrated_mae:.2f}），"
                     f"A 选 epoch {int(a.epoch)}（val {a.val_calibrated_mae:.2f}）；"
                     f"确认集总体 MAE U {u.overall_abs_mae_calibrated:.2f} vs A {a.overall_abs_mae_calibrated:.2f}")
    still = tail[tail.still_descending]
    trend_text = (f"{len(still)}/6 条曲线在最近 20 轮仍明显下降"
                  if len(still) else "六条曲线在最近 20 轮均已走平或回升")
    paired_final = paired[paired.criterion == 'final_best']
    overall = paired_final[paired_final.cohort == 'overall']
    sig = overall[(overall.ci95_low > 0) | (overall.ci95_high < 0)]
    paired_text = (f"最终最佳比较下，总体队列的 U−A 配对区间有 {len(sig)}/{len(overall)} 个 seed 不含 0"
                   if len(overall) else "无配对结果")

    comparison = f"""# FINAL_COMPARISON — U vs A 匹配长训练

数据来源：`results/` 下的 CSV；所有数值由 `code/evaluate_alignment.py` 与
`code/build_reports.py` 从产物直接读出。协议见同目录 `PROTOCOL.md`。

## 0. 结论摘要

{chr(10).join(lines)}

- {trend_text}（见第 5 节）。
- {paired_text}（见第 4 节）。

## 1. 训练是否仍在下降（最近 20 轮）

{tail_table}

## 2. 第一类与第二类：相同预算与最终最佳

最终结果表（绝对误差单位为 meV/atom；"总体"= 确认集总体队列 2048 个结构，
"相对队列"= 冻结相对能量队列 200 组 / 422 个结构。表中 MAE 均为**统一后校准**预测）：

{main_table}

第二类（最终最佳）同时给出原生与后校准口径，便于判断后校准是否改变了结论：

{native_table}

## 3. 第三类：相近绝对精度

阈值取每个 seed 内两条路线第 20 轮的 val 后校准 MAE 与各自 120 轮最优 val MAE，
仅保留两条路线都能达到的阈值；每个阈值取各路线**首次达到**该阈值的 epoch：

{class3_table}

判据：实际验证 MAE 差距 ≤ 1.0 meV 记为"可称相近"，> 2.0 meV 记为"差距明显"。
若无法匹配，此处会直接写明而不是回到确认集上补挑点。

## 4. 配对区间（U − A，负值表示 U 更好）

总体队列按精确组成组做簇 bootstrap（单例组成各自成簇）；
相对队列按 200 个完整组成组做配对 bootstrap。
"相对队列绝对MAE"用未中心化的绝对误差，"相对队列EC-MAE"用组内中心化后的绝对误差；
后者在数学上不受线性组成后校准影响，因此原生与后校准给出同一数值。

{paired_table}

## 5. 逐 epoch 耗时

{timing_table}

## 6. 限制

- 120 epoch 是**预定上限**，不自动代表收敛；第 1 节给出各曲线的末段趋势供判断。
- 后校准是线性组成校正，不改变组内相对能量，故相对队列 EC-MAE 与是否后校准无关。
- 确认集在选点冻结后统一预测；后校准参数取自训练期已拟合的该 epoch 参数，
  未使用任何确认集标签。
- 本文件只比较 U 与 A 两条路线，不含 C 组、非线性组成预处理或新的 alpha 搜索。
"""
    (root / 'PROTOCOL.md').write_text(protocol)
    (root / 'FINAL_COMPARISON.md').write_text(comparison)
    print('wrote', root / 'PROTOCOL.md')
    print('wrote', root / 'FINAL_COMPARISON.md')


if __name__ == '__main__':
    main()
