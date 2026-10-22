# Energy-reference alignment

预训练原子势微调中元素参考能差异对相对能量的影响。

这里保存了论文使用的代码、数据划分和实验结果。实验比较了直接微调（direct）与先处理元素参考能差异再微调（aligned）两条路线，使用 SevenNet 和 EquiformerV3，评价形成能、同组成结构间的相对能量、结构排序和原子力变化。

## 文件说明

| 目录 | 内容 |
|---|---|
| `energy/experiment_reference_allocation_final_20261004/` | 主要实验、预测结果、固定结构对、bootstrap 和合成参考能对照 |
| `energy/experiment17_eqv3_physical_matched_20261001/` | EquiformerV3 配对实验、训练记录和评价结果 |
| `energy/reference_allocation_robustness_20261005/` | 100 个结构的原子力评价、有限差分检查和 MP 弛豫轨迹评价 |
| `energy/experiment14_alignment_controls_20260929/` | 学习率、参考能及跨计算方法对照 |
| `energy/runs/`、`energy/rclfa_v3/` | 数据划分、结构文件和相关运行记录 |
| `environment/` | 实验环境和依赖版本 |
| `manuscript/` | 论文 LaTeX 源文件 |
| `checks/`、`provenance/` | 数值核对、文件来源和校验记录 |
| `tools/` | 结果汇总和归档检查脚本 |

## 查看结果

用 Python 3 运行以下命令即可重新汇总保存的结果，不需要 GPU：

```bash
python tools/summarize_results.py
```

输出保存在 `checks/paper_number_check.json`，轨迹结构对及逐任务指标保存在同目录的 CSV 文件中。

主要结果也可以直接查看 [FINAL_REPORT.md](energy/experiment_reference_allocation_final_20261004/FINAL_REPORT.md)。训练和模型评价代码在各实验目录内，依赖版本见 `environment/`。重新运行 GPU 实验前，需要按本机环境调整脚本中的数据和模型路径。

## 完整数据与模型

代码仓库包含结构、数据划分、预测结果和统计表。模型权重、图缓存及序列化模型文件放在 [v1.0.0 完整归档](https://github.com/ben-zun/energy/releases/tag/v1.0.0) 中，文件名为 `energy-reference-alignment-20261016.zip`。仓库中未直接存放的文件列在 `EXTERNAL_ARTIFACTS.json`。

下载并解压完整归档后，在解压目录运行：

```bash
python tools/verify_release.py
```

该命令检查原始文件哈希、数据划分、固定结构对、预测结果和有限差分记录；输出写入 `checks/verification_report.json`。

第三方代码和模型保留原有许可证：[EquiformerV3](energy/external_repos/equiformer_v3-main/LICENSE)、[SevenNet](third_party/sevennet/LICENSE)。
