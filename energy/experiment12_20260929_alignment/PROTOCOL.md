# PROTOCOL — U vs A 匹配长训练

工程：`/home/yxzhang/energy/experiment12_20260929_alignment`；数据/模型/校准来源均只读引用既有实验，不覆盖任何已有产物。

## 1. 两条路线

令 $f_\theta(X)$ 为 SevenNet 输出总能量，$N_X$ 为原子数，$d(z)$ 为冻结的元素参考校准。

| 路线 | 进入训练损失的预测 | 脚本参数 |
|---|---|---|
| **U**（训练前不对齐） | $\widehat y_U = f_\theta(X)/N_X$ | `--alignment none` |
| **A**（训练前对齐） | $\widehat y_A = f_\theta(X)/N_X + d(z)$ | `--alignment element_reference` |

两者都与同一个单位原子形成能标签 $y$（`target_per_atom`，eV/atom）计算 Smooth L1 损失。

$d(z) = -(\mathbf{x}^\mathsf{T}\mu_Z + b)$，其中 $\mu_Z$ 与 $b$ 取自
`/home/yxzhang/energy/runs/rclfa_v3/real_fegx_foundation_replacement_272/reference_conversion/sevennet_reference/calibration_coefficients.csv` 的 `element_reference` 块（118 个元素系数 + 1 个截距，**无尺度项**，
原 train 拟合、冻结不更新）。A 不需要额外加载冻结模型——它只在预测式里加一个常数偏移。

## 2. 训练配置（两条路线完全一致）

| 项目 | 设置 |
|---|---|
| 模型 | 同一个 `7net-0` checkpoint（`load_sevennet_model(remove_force_output=True)`） |
| 数据 | 原 train 8192、val 1024；test/确认集不参与选择 |
| 可训练参数 | `['3_', '4_', 'reduce_input_to_hidden', 'reduce_hidden_to_energy', 'rescale_atomic_energy']`，断言 301300 个 |
| 优化器 | AdamW，lr 0.0003，weight decay 1e-05 |
| 损失 | Smooth L1，beta=0.05，单位 eV/atom |
| batch | 训练 16，评估 32 |
| 最大 epoch | 120，**不启用提前停止** |
| 随机种子 | 20260528, 20260529, 20260530 |
| 学习率调度 | 无 |
| 初始化与 batch 顺序 | 同 seed 下 U/A 完全一致（同一 `torch.Generator().manual_seed(seed)`） |

## 3. 统一后校准与 checkpoint 选择

每个 epoch 保存两套预测：

- **原生预测**：上表中实际进入训练损失的输出；
- **统一后校准预测**：$\widehat y^{cal}_t = \widehat y_t + q_t(z)$，
  $q_t(z) = \mathbf{x}^\mathsf{T}\mathbf{w}_t + b_t$。

$q_t$ 的拟合规则（U/A 完全相同）：

- 每个 epoch 只用 **train** 的残差 $y - \widehat y_t$ 拟合；
- 特征 = 118 维原子分数 + 截距；Ridge $\alpha = 1e-06$，截距不惩罚；
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
| A 初始预测 == exp11 归档 base，medium 全 11264 行最大差 | 8.109e-06 eV/atom |
| A 初始预测 == 确认集归档 base，2436 行最大差 | 5.722e-06 eV/atom |
| A − U == d(z) 最大差 | 6.464e-06 eV/atom |
| `route_prediction('U')` vs 原生输出最大差 | 9.537e-07 |
| `route_prediction('A')` vs 原生+d 最大差 | 1.145e-06 |
| 总能量只除一次原子数（float32 舍入内） | max abs diff 9.537e-07 eV/atom |
| d(z) 组内最大差（medium） | 0.000e+00 eV/atom |
| d(z) 组内最大差（确认集） | 0.000e+00 eV/atom |

## 6. 文件

```text
/home/yxzhang/energy/experiment12_20260929_alignment/
  code/alignment_common.py           共享定义（d(z)、组成分数、per_atom、后校准）
  code/train_alignment_matched.py     U/A 统一训练脚本
  code/evaluate_alignment.py         三类比较 + 确认集评估 + 配对 bootstrap
  code/build_reports.py              生成本文件与 FINAL_COMPARISON.md
  runs/{U,A}_{seed}/history.csv              逐 epoch 曲线
  runs/{U,A}_{seed}/calibration_history.csv  逐 epoch 后校准参数（118 w + b）
  runs/{U,A}_{seed}/epoch*.pt                逐 epoch checkpoint
  runs/{U,A}_{seed}/val_epoch*.npz           逐 epoch val 原生/后校准预测
  runs/{U,A}_{seed}/status.json              状态与初始化不变量
  results/training_history.csv       六次运行合并曲线
  results/selected_checkpoints.csv   三类比较选出的 epoch
  results/final_metrics.csv          选中 checkpoint 的确认集指标
  results/paired_differences.csv     U−A 配对区间
  results/tail20_trend.csv           最近 20 轮趋势
```
