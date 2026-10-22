# EqV3“提前对齐适用边界”解释撤回

2026-10-01 的 live 第0轮检查发现：旧 clean 参考项用反归一化归档能量拟合，
却加到了训练器另一种原生输出解释上。该比较未实现宣称的提前对齐。
因此旧 direct/clean 结果不能用于证明 EqV3 是提前对齐策略的适用边界，
也不能据此声称更好的初始校准导致了更差适配。

训练曲线、checkpoint、参考CSV和旧数值全部保留；撤回的是科学解释。
未仅凭这一问题认定旧direct训练错误：其原生任务输出和评价链需独立判断。

完整输出契约、真实缺失预测补齐及新参考核查见：
/home/yxzhang/energy/experiment16_eqv3_output_contract_20261001/OUTPUT_CONTRACT_REPORT.md

本轮不启动新匹配训练或组成偏移训练；修正后的clean需要新训练，不能用旧checkpoint代替。
