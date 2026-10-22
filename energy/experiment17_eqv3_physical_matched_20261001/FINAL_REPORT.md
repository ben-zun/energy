# EqV3完整物理解码匹配训练结果

六个全新训练：direct/clean × 三seed，均为40 epochs；没有按结果方向追加或删减训练。
两路线相同预训练权重和LoRA初始化，physical total SmoothL1，只有clean加入固定组成参考。
模型选择只用原validation原生每原子MAE；最佳模型的后处理组成校准只拟合train。
物理损失使用float64标量运算以降低总能量相消误差；网络两路线均bfloat16。
梯度等价检查、重复反向数值噪声和初始化组内不变检查见各run/gate.json。

## 三seed均值与标准差

| scope             | calibration   | route   |   mae_mev_mean |   mae_mev_std |   ec_mae_mev_mean |   ec_mae_mev_std |   major_wrong_rate_mean |   major_wrong_rate_std |
|:------------------|:--------------|:--------|---------------:|--------------:|------------------:|-----------------:|------------------------:|-----------------------:|
| overall2048       | calibrated    | clean   |       17.78865 |       0.45526 |         nan       |        nan       |               nan       |              nan       |
| overall2048       | calibrated    | direct  |       62.38394 |       1.68054 |         nan       |        nan       |               nan       |              nan       |
| overall2048       | native        | clean   |       19.46541 |       0.62254 |         nan       |        nan       |               nan       |              nan       |
| overall2048       | native        | direct  |       67.70223 |       3.44606 |         nan       |        nan       |               nan       |              nan       |
| relative_clean401 | calibrated    | clean   |       17.03643 |       1.46438 |           9.10276 |          0.88070 |                 0.02807 |                0.00804 |
| relative_clean401 | calibrated    | direct  |       58.63458 |       4.62675 |          27.87931 |          0.87362 |                 0.16316 |                0.01898 |
| relative_clean401 | native        | clean   |       18.70645 |       1.34575 |           9.10276 |          0.88070 |                 0.02807 |                0.00804 |
| relative_clean401 | native        | direct  |       62.34685 |       6.41319 |          27.87931 |          0.87362 |                 0.16316 |                0.01898 |

## 配对组成组bootstrap

差值为clean−direct。负值表示clean更好。按整组成组重采样10000次；
总体MAE包含单例组，relative只用已清洗190组/401结构。均值CI对三个已运行seed条件化，不将seed×结构当独立样本。

| seed          | scope             | calibration   | metric           |   clean_minus_direct |   ci95_low |   ci95_high |   groups |
|:--------------|:------------------|:--------------|:-----------------|---------------------:|-----------:|------------:|---------:|
| 20260528      | relative_clean401 | native        | absolute_mae     |            -40.97140 |  -47.95892 |   -34.51230 |      190 |
| 20260528      | relative_clean401 | native        | ec_mae           |            -18.96923 |  -24.62523 |   -14.06611 |      190 |
| 20260528      | relative_clean401 | native        | major_wrong_rate |             -0.13158 |   -0.18421 |    -0.08421 |      190 |
| 20260529      | relative_clean401 | native        | absolute_mae     |            -40.10420 |  -49.40334 |   -32.14656 |      190 |
| 20260529      | relative_clean401 | native        | ec_mae           |            -18.94447 |  -24.02803 |   -14.25187 |      190 |
| 20260529      | relative_clean401 | native        | major_wrong_rate |             -0.11053 |   -0.16842 |    -0.05789 |      190 |
| 20260530      | relative_clean401 | native        | absolute_mae     |            -49.84560 |  -61.21094 |   -40.27801 |      190 |
| 20260530      | relative_clean401 | native        | ec_mae           |            -18.41596 |  -22.67195 |   -14.26795 |      190 |
| 20260530      | relative_clean401 | native        | major_wrong_rate |             -0.16316 |   -0.22105 |    -0.10526 |      190 |
| mean_of_three | relative_clean401 | native        | absolute_mae     |            -43.64040 |  -51.28913 |   -37.11688 |      190 |
| mean_of_three | relative_clean401 | native        | ec_mae           |            -18.77655 |  -22.85662 |   -15.13244 |      190 |
| mean_of_three | relative_clean401 | native        | major_wrong_rate |             -0.13509 |   -0.17895 |    -0.09298 |      190 |
| 20260528      | overall2048       | native        | absolute_mae     |            -43.89473 |  -46.82635 |   -40.99256 |     2045 |
| 20260529      | overall2048       | native        | absolute_mae     |            -50.97382 |  -54.13578 |   -47.84789 |     2045 |
| 20260530      | overall2048       | native        | absolute_mae     |            -49.84193 |  -52.86273 |   -46.84798 |     2045 |
| mean_of_three | overall2048       | native        | absolute_mae     |            -48.23683 |  -50.71396 |   -45.76986 |     2045 |
| 20260528      | relative_clean401 | calibrated    | absolute_mae     |            -40.51719 |  -47.94599 |   -33.70152 |      190 |
| 20260528      | relative_clean401 | calibrated    | ec_mae           |            -18.96923 |  -24.62523 |   -14.06611 |      190 |
| 20260528      | relative_clean401 | calibrated    | major_wrong_rate |             -0.13158 |   -0.18421 |    -0.08421 |      190 |
| 20260529      | relative_clean401 | calibrated    | absolute_mae     |            -38.28810 |  -46.97232 |   -31.00425 |      190 |
| 20260529      | relative_clean401 | calibrated    | ec_mae           |            -18.94447 |  -24.02803 |   -14.25187 |      190 |
| 20260529      | relative_clean401 | calibrated    | major_wrong_rate |             -0.11053 |   -0.16842 |    -0.05789 |      190 |
| 20260530      | relative_clean401 | calibrated    | absolute_mae     |            -45.98917 |  -56.38491 |   -37.06100 |      190 |
| 20260530      | relative_clean401 | calibrated    | ec_mae           |            -18.41596 |  -22.67195 |   -14.26795 |      190 |
| 20260530      | relative_clean401 | calibrated    | major_wrong_rate |             -0.16316 |   -0.22105 |    -0.10526 |      190 |
| mean_of_three | relative_clean401 | calibrated    | absolute_mae     |            -41.59816 |  -49.25755 |   -35.10549 |      190 |
| mean_of_three | relative_clean401 | calibrated    | ec_mae           |            -18.77655 |  -22.85662 |   -15.13244 |      190 |
| mean_of_three | relative_clean401 | calibrated    | major_wrong_rate |             -0.13509 |   -0.17895 |    -0.09298 |      190 |
| 20260528      | overall2048       | calibrated    | absolute_mae     |            -42.35771 |  -45.29963 |   -39.50910 |     2045 |
| 20260529      | overall2048       | calibrated    | absolute_mae     |            -46.06490 |  -48.95161 |   -43.15851 |     2045 |
| 20260530      | overall2048       | calibrated    | absolute_mae     |            -45.36326 |  -48.27673 |   -42.53335 |     2045 |
| mean_of_three | overall2048       | calibrated    | absolute_mae     |            -44.59529 |  -46.99811 |   -42.23194 |     2045 |

## 解释边界

不沿用旧EqV3负结果。新比较是否支持对齐必须分别看绝对、相对指标和配对CI。
第0轮更好不保证最终EC更好；总能量损失的饱和率不能直接与SevenNet每原子损失比较。
不因结果方向追加epochs、模型或人工组成偏移训练。
确认样本已在先前研究中查看，本轮冻结用于评价且不参与拟合选择，不宣称全新盲测。
