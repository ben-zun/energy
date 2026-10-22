可以。我建议这次不要再“边跑边想”，而是把实验协议先冻结。你现在真正需要做的事情其实不多，但每一项都必须保证定义一致，否则很容易又出现 EqV3 之前那种“参考项和输出标尺混用”的问题。

我把它整理成一份可以直接照着实施的实验清单。核心原则是：**不再扩模型，不再扩方法，只完成因果证据和跨模型收尾。**

---

## 一、先冻结所有实验共同协议

在任何新代码之前，先单独写一个 `PROTOCOL.md` 或 `experiment_config.yaml`，把下面这些东西锁死。后面任何脚本都从这里读取，不允许每个实验自己重新定义。

1. **能量输出定义必须统一**

   所有训练、验证、测试最终都回到同一个“物理总能量”定义。

   如果模型内部输出的是某种编码能量，必须明确写成：

   \[
   E_{\mathrm{phys}}
   =
   D(E_{\mathrm{raw}},z)
   \]

   其中 \(D\) 是唯一的物理解码函数。

   direct 和 aligned 唯一允许的差别是：

   \[
   E_U(X)=E_\theta(X)
   \]

   \[
   E_A(X)=E_\theta(X)+h(z)
   \]

   不能一条路线在 raw output 上算 loss，另一条路线在 decoded physical energy 上算 loss，除非已经严格证明两者的 forward 和 gradient 完全等价。

   EqV3 之前的问题就是这里，因此这个检查必须放在所有训练之前。

2. **单位只能有一种主单位**

   建议训练统一使用物理总能量 eV，评价统一换成 eV/atom 或 meV/atom。

   如果一个组成校正是每原子形式

   \[
   h_{\rm pa}(z)=\sum_i x_i\delta\mu_i,
   \]

   那么加入总能量时必须是

   \[
   h_{\rm tot}(z)=N h_{\rm pa}(z)
   =\sum_i n_i\delta\mu_i.
   \]

   这是最容易写错的地方之一。

3. **训练集、验证集、测试集 ID 冻结**

   不重新划分。

   direct/aligned 必须使用完全相同的结构 ID。

   所有参考能拟合、标准化统计、后处理校准只能使用 train。

   validation 只允许做 checkpoint selection。

   test 完全不能参与：
   - 元素参考能拟合；
   - 均值方差；
   - 超参数选择；
   - checkpoint 选择；
   - 阈值选择。

4. **同组成结构的定义不能临时改**

   你当前这篇主线里的 same-composition 分析，继续沿用原实验定义。

   如果之前是“元素计数向量完全一致”，就继续使用：

   ```text
   comp_key = ((Z1,n1),(Z2,n2),...)
   ```

   不要突然换成 `reduced_formula`。

   `reduced_formula` 可以用于防止 train/test chemistry leakage，但不是同一个概念。你以前已经构建过 formula-group-disjoint benchmark，这种划分方式本身已经审计过。粘贴的文本 (1)

5. **随机种子固定**

   SevenNet 和 EqV3 都继续使用原来的三个种子。

   synthetic control 也用同一组三个 seed，不另选“效果好”的 seed。

6. **每次训练必须保存完整状态**

   每个 epoch 至少保存：

   ```text
   model_state
   optimizer_state
   scheduler_state
   grad_scaler_state
   epoch
   global_step
   random_state_python
   random_state_numpy
   random_state_torch
   random_state_cuda
   ```

   自动重启时必须从完整 epoch 恢复，而不是只加载模型权重重新启动 optimizer。

7. **所有输出都带哈希和版本信息**

   每个实验目录保存：

   ```text
   git_commit
   config_sha256
   train_ids_sha256
   val_ids_sha256
   test_ids_sha256
   pretrained_checkpoint_sha256
   reference_coefficients
   reference_coefficients_sha256
   ```

你之前已经做过非常严格的一对一 ID 检查、train-only 标准化、重复预测一致性和数值容差审计，这套习惯应该原样继承。粘贴的文本 (1)

---

# 二、实验 1：EqV3 正确对照的最终测试集收尾

这是**必须完成**的，但不用重新训练。

你现在六条 40 epoch 训练都已经完成，所以只做 evaluation。

### 目的

回答：

> SevenNet 中观察到的 unaligned/aligned 差异是否能在不同架构 EqV3 上复现？

### 不能改的东西

direct 和 aligned 必须：

- 相同 pretrained checkpoint；
- 相同 LoRA 初始化；
- 相同训练数据；
- 相同 loss；
- 相同 optimizer；
- 相同训练轮数；
- 相同 precision；
- 相同 checkpoint selection；
- 相同训练后 calibration。

唯一差别：

```text
direct:
E_pred = E_model

aligned:
E_pred = E_model + h_reference(z)
```

### 具体执行顺序

对每个 seed：

```text
1. 读取所有 epoch validation log
2. 按预先规定的 validation rule 选 checkpoint
3. 记录被选 epoch
4. 加载 checkpoint
5. 在 train 上生成 prediction
6. 用 train-only 数据拟合后处理组成校准
7. 固定校准系数
8. 在 test 上生成 raw physical prediction
9. 应用固定 calibration
10. 保存 test prediction
```

不要在 direct 和 aligned 上分别临时发明不同的 calibration 方法。

### 必须保存的预测表

建议统一生成：

```text
eqv3_test_predictions.csv
```

字段：

```text
id
seed
route                 # direct / aligned
split
formula
composition_key
n_atoms
y_target_mev_atom
y_pretrained_mev_atom
y_pred_raw_mev_atom
y_pred_calibrated_mev_atom
selected_epoch
```

### 必须评价的指标

第一组，绝对能量：

\[
\mathrm{MAE}
=
\frac1N\sum_i|\hat E_i-E_i|
\]

同时报告：
- MAE
- RMSE
- bias
- p90/p95

第二组，同组成关系：

- EC-MAE
- pairwise \(\Delta E\)-MAE
- pair accuracy
- near-degenerate pair accuracy
- top-1
- major misselection rate

这些指标的定义全部复用 SevenNet 脚本，不重新写一套 EqV3 版本。

最好抽出一个公共函数：

```python
evaluate_absolute(...)
evaluate_same_composition(...)
```

SevenNet 和 EqV3 都调用同一个 evaluator。

### 统计方式

不要对三个 seed 做普通 t-test。

更合理的是：

- 每个 seed 给出指标；
- 报 mean ± std；
- direct vs aligned 做 paired bootstrap。

对于 absolute MAE，可以按 structure resample。

对于 EC / pair / top-1，建议按 composition group resample，不能直接把同一组里的 pair 当独立样本。

### 通过条件

这里不要提前规定“aligned 必须赢多少”。

只要求：

- 代码审计通过；
- 三个 seed 全部评价完成；
- direct/aligned 使用同一 evaluator；
- test 从未进入 calibration 或 checkpoint selection。

即便最后 test 没有 val 那么漂亮，也照实报告。

---

# 三、实验 2：SevenNet + EqV3 的 required-vs-induced 分析

这个**不用训练**，但我认为必须做。

这是现在新主线里最重要的机制分析。

## 目的

回答：

> 微调真正改变了多少同组成结构之间的能量关系？
>
> 这些变化是不是目标标签实际要求的？

对于同组成结构对 \(i,j\)，定义预训练模型的相对能量：

\[
\Delta E_{ij}^{0}
=
E_i^0-E_j^0.
\]

目标数据要求的相对能量是：

\[
\Delta E_{ij}^{\mathrm{tar}}
=
E_i^{\mathrm{tar}}-E_j^{\mathrm{tar}}.
\]

那么目标真正要求模型改变的量是：

\[
r_{ij}^{\mathrm{target}}
=
\Delta E_{ij}^{\mathrm{tar}}
-
\Delta E_{ij}^{0}.
\]

微调实际产生的改变是：

\[
r_{ij}^{\mathrm{FT}}
=
\Delta E_{ij}^{\mathrm{FT}}
-
\Delta E_{ij}^{0}.
\]

真正应该比较的是：

\[
r_{ij}^{\mathrm{FT}}
\quad\text{vs}\quad
r_{ij}^{\mathrm{target}}.
\]

不是简单看 \(r^{FT}\) 大不大。

---

## 具体代码流程

从同一个 prediction CSV 开始。

先构造 same-composition groups。

组内生成固定 pair list。

这个 pair list必须：
- direct/aligned 共用；
- 所有 seed 共用；
- 不因为某一路线缺数据而动态变化。

生成：

```text
same_composition_pairs.csv
```

字段：

```text
group_id
id_i
id_j
delta_pre
delta_target
delta_direct
delta_aligned
r_target
r_direct
r_aligned
```

其中：

```python
r_target  = delta_target  - delta_pre
r_direct  = delta_direct  - delta_pre
r_aligned = delta_aligned - delta_pre
```

### 必须计算的四个量

第一，最直接的 excess error：

\[
\epsilon^{FT}_{ij}
=
r^{FT}_{ij}-r^{target}_{ij}.
\]

报告：

\[
\mathrm{MAE}_{excess}
=
\langle |\epsilon_{ij}| \rangle.
\]

第二，全局 norm ratio：

\[
R_{\rm norm}
=
\frac{\|r^{FT}\|_2}
{\|r^{target}\|_2}.
\]

注意必须是**整体向量 norm 后再比**。

不要对每个 pair 算

\[
|r^{FT}_{ij}|/|r^{target}_{ij}|
\]

再平均，因为当 target 很小时会爆炸。

第三，方向一致性：

\[
\rho
=
\mathrm{corr}(r^{FT},r^{target})
\]

或者 cosine similarity。

第四，我很建议再加一个投影系数：

\[
\alpha
=
\frac{(r^{FT})^\top r^{target}}
{\|r^{target}\|_2^2}.
\]

它很直观：

- \(\alpha\approx1\)：修改幅度接近目标要求；
- \(\alpha>1\)：沿目标方向 overshoot；
- \(\alpha<1\)：改变不足；
- correlation 很低：修改方向本身就不对。

然后再报告正交残差：

\[
r_{\perp}
=
r^{FT}
-
\alpha r^{target}.
\]

\[
R_\perp
=
\frac{\|r_\perp\|}
{\|r^{target}\|}.
\]

这个比单纯“3.52 倍”更有解释力。

---

## 画什么图

只需要两张核心图。

第一张：

横轴：

\[
r^{target}
\]

纵轴：

\[
r^{FT}
\]

direct 一张，aligned 一张。

画 \(y=x\)。

第二张：

三个条形或点图：

```text
pretrained
direct
aligned
```

比较：
- excess MAE
- norm ratio
- correlation / projection alpha

### 这项实验结果怎么解释

不能说：

> direct 改了 PES，所以 direct 错。

应该说：

> direct 产生的结构相关变化与目标所需变化之间偏差更大。

如果 aligned 更接近 \(r^{target}\)，才是你真正需要的结论。

---

# 四、实验 3：Synthetic pure-reference control

这是**唯一一个我真正建议新增训练的实验**。

也是最重要的因果实验。

---

## 目的

现实目标数据里有一个永远无法完全排除的问题：

> 也许目标泛函本来就要求 PES 改变。

Synthetic control 就人为创造一个我们知道正确答案的世界：

> 目标与 pretrained model 的唯一差别就是元素参考能变化，PES 真值完全不变。

这样如果 direct 仍然改变同组成相对能量或力，我们才能非常干净地说：

> reference mismatch 本身足以诱导不必要的结构相关变化。

---

## 4.1 Synthetic target 怎么构造

这个地方一定不要用真实 DFT target 直接加 shift。

最干净的是用 pretrained model 自己做 teacher。

对每个结构 \(X\)，先固定预训练模型：

\[
E_0(X).
\]

然后构造：

\[
E_{\mathrm{syn}}(X)
=
E_0(X)+h(z).
\]

其中：

\[
h(z)
=
\sum_i n_i\delta\mu_i
\]

如果训练使用 total energy。

如果存储时需要 per atom：

\[
h_{\mathrm{pa}}(z)
=
\frac{1}{N}\sum_i n_i\delta\mu_i.
\]

### \(\delta\mu_i\) 从哪里来

不要自己随机编。

最好直接使用你真实 aligned 实验中、只从 training set 拟合得到的元素参考能差值。

这样 synthetic shift 的量级与真实任务一致。

但 synthetic dataset 的 label 是：

```python
y_syn = y_pretrained + known_reference_shift
```

而不是：

```python
y_syn = real_target + shift
```

前者才保证“真实 PES change = 0”。

---

## 4.2 三条路线

至少保留三个条件。

### Oracle reference

不训练模型：

\[
E_{\rm oracle}
=
E_0+h(z).
\]

这是理论参考。

它应该满足：

```text
energy error ≈ 0
EC drift = 0
force drift = 0
```

如果这一条不满足，说明数据构造或单位错了。

### Synthetic-direct

不提供显式 reference：

\[
E_{\rm pred}=E_\theta.
\]

让模型参数自己去拟合：

\[
E_0+h(z).
\]

初始化：

\[
\theta=\theta_0.
\]

### Synthetic-aligned

显式加入已知 reference：

\[
E_{\rm pred}
=
E_\theta+h(z).
\]

同样从：

\[
\theta_0
\]

开始。

理论上初始化时已经接近精确 synthetic target。

---

## 4.3 必须再加一个零 shift control

建议做：

\[
h(z)=0.
\]

也就是：

\[
E_{\rm syn}=E_0.
\]

作用是验证：

> 即使什么都不用学，训练循环本身会不会制造 drift？

如果 zero-shift direct 也产生很明显 EC drift，那先别解释科学问题，训练实现本身可能有问题。

所以 synthetic 最小矩阵：

| shift | direct | aligned |
|---|---|---|
| 0× | ✓ | 可省，因为二者相同 |
| 1×真实参考差值 | ✓ | ✓ |

这已经够了。

0.5× / 2× dose-response 可以以后再决定，我现在**不建议预先增加**。

---

## 4.4 训练协议

直接复用 SevenNet 主实验。

不要为了 synthetic control 再调一套超参数。

使用完全相同：
- seed；
- lr；
- optimizer；
- Smooth L1；
- batch size；
- epoch 数；
- scheduler；
- mixed precision；
- gradient clipping；
- checkpoint rule。

如果真实 SevenNet 主协议是 120 epoch，那 synthetic 也跑 120。

即使 aligned 在 epoch 0 就已经很好，也照固定预算走，不提前停止。

---

## 4.5 synthetic 必须评价什么

### Primary endpoint 1：EC drift

因为 synthetic target 只有 composition shift，所以正确真值是：

\[
\Delta E_{ij}^{target}
=
\Delta E_{ij}^{0}.
\]

于是：

\[
r_{ij}^{target}=0.
\]

定义：

\[
D_{\rm EC}
=
\operatorname{MAE}
\left[
\Delta E_{ij}^{FT}
-
\Delta E_{ij}^{0}
\right].
\]

这是 synthetic 最核心指标。

---

### Primary endpoint 2：force drift

因为：

\[
\nabla_Xh(z)=0,
\]

synthetic 真值下原子力应该完全等于 pretrained model：

\[
F_{\rm syn}=F_0.
\]

所以比较：

\[
D_F
=
\sqrt{
\frac{1}{3N}
\sum
\|F^{FT}-F^0\|^2
}.
\]

注意论文里叫：

> force change / force drift

不要叫：

> force error

因为你没有高精度目标力标签。

---

### Secondary endpoint

还可以记录：

- absolute synthetic MAE；
- parameter update norm；
- EC pair accuracy；
- top-1 relative to pretrained ranking；
- per-epoch EC drift。

其中我尤其建议记录每个 epoch：

```text
epoch
train_energy_loss
val_energy_mae
val_EC_drift
val_force_drift
```

如果出现：

> direct energy MAE 持续下降，同时 EC drift 上升

这个训练动力学图会非常有说服力。

---

## 4.6 Synthetic 代码必须做的单元测试

训练开始前全部通过。

### Test A：synthetic shift 数学正确

随机抽 100 个结构：

```python
assert abs(
    y_syn_total - y_pre_total - sum(n_i * delta_mu_i)
) < tolerance
```

### Test B：同组成 shift 完全一致

同组成 pair：

```python
shift_i_per_atom == shift_j_per_atom
```

容差建议：

```text
1e-8 eV/atom
```

如果 FP32，则可放宽到 `1e-6`.

### Test C：oracle EC 完全不变

```python
delta_oracle_pair == delta_pretrained_pair
```

### Test D：oracle force 完全不变

因为你不需要重新跑 reference term 的 gradient：

```text
∂h/∂R = 0
```

代码上必须确认 reference term 不依赖 position tensor。

特别注意：

```python
h = composition_features(...)
```

里面不能意外使用任何 coordinate-dependent embedding。

---

## 4.7 Synthetic 结果如何判断

### 如果得到

```text
direct:
energy MAE ↓
EC drift 大
force drift 大

aligned:
energy MAE ≈ 0
EC drift ≈ 0
force drift ≈ 0
```

这是最理想结果。

### 如果 direct 也几乎没有 EC drift

不要硬解释。

说明这个架构本身可能存在足够强的 composition-only degrees of freedom，可以把 reference shift 学进去而不改 PES。

那么结论应变成：

> excess PES change 并非 reference mismatch 的必然结果，而与模型参数化和优化路径有关。

这仍然是科学结果。

### 如果 aligned 也漂移

优先怀疑实现。

因为 synthetic aligned 初始化理论上已经匹配 target。

检查：
- reference sign；
- total/per-atom；
- decode；
- loss scale；
- optimizer；
- weight decay；
- scheduler；
- AMP。

特别注意 weight decay。

即便 loss=0，AdamW 的 decoupled weight decay 仍然可能改变参数！

这个地方非常关键。

如果 synthetic-aligned 的 loss 为 0，但仍然使用 AdamW weight decay，模型参数仍可能被更新。

所以 synthetic 实验里你必须确认现有 optimizer 是否有 weight decay。

如果有，不能为了让 aligned 不动而临时把它关掉，因为那会破坏 matched protocol。

正确做法是：
- 两条路线保持相同 optimizer；
- 同时增加一个 `no-training oracle`；
- 将 optimizer-induced drift 作为 background；
- zero-shift control 估计这种 background。

这就是为什么 0× shift control 很重要。

---

# 五、实验 4：把 Synthetic 的 force 定义审计好

这项不是独立训练，但建议写一个小测试脚本。

选 8–16 个结构。

对模型物理能量：

\[
E_{\mathrm{phys}}(R)
\]

用 autograd 得到：

\[
F=-\frac{\partial E}{\partial R}.
\]

然后检查：

- pretrained；
- synthetic-direct；
- synthetic-aligned。

不要从另一个 head 或另一个未经物理解码的 energy 去计算 force。

如果模型代码本身提供 `forces` 输出，也要验证它和你当前物理解码后的 energy gradient 一致。

这里尤其防止出现：

```text
训练 loss 用 decoded total energy
force 却从 raw normalized energy 求导
```

这种隐蔽错误。

---

# 六、实验 5：跨协议 MP→r²SCAN 只做统一整理，不重新训练

这部分已有结果够了。

不再训练。

只需要统一成和 SevenNet / EqV3 相同格式的结果表：

```text
model
protocol
seed
route
absolute_MAE
EC_MAE
pair_acc
top1
major_misselection
```

这样正文中可以直接比较：

```text
SevenNet / target A
EqV3 / target A
SevenNet / MP→r2SCAN
```

目的只是：

> 现象不是一个模型、一个目标能量定义的偶然结果。

---

# 七、实验 6：最后统一做一次 numerical audit

这个我认为必须单独成为一个脚本，例如：

```text
audit_alignment_experiments.py
```

最后只输出：

```text
AUDIT_OK
```

或者明确报错退出。

必须检查：

### 数据

```text
train ∩ val = 0
train ∩ test = 0
val ∩ test = 0
```

预测 ID 和 target ID 一一对应。

没有 duplicate IDs。

没有 NaN / inf。

你以前已经踩过真实材料化学式 `"NaN"` 被 pandas 当缺失值的问题，所以继续强制：

```python
pd.read_csv(..., keep_default_na=False)
```

这类数据清洗问题你以前已经专门修过。粘贴的文本 (1)

### 初始化

每个 seed：

```text
hash(init_direct) == hash(init_aligned)
```

### epoch 0 同组成关系

direct/aligned 的 epoch0 结构相关预测必须相同。

把 composition-only difference 消掉之后：

\[
\max_g
\operatorname{spread}
[
E_A^0-E_U^0
]
<10^{-6}\ {\rm eV/atom}
\]

### reference

reference coefficients 只由 train 得到。

保存系数。

test evaluation 时禁止 refit。

### loss

在关闭 autocast 的 FP32 下：

```text
physical-loss implementation
vs
encoded equivalent implementation
```

比较：
- loss value；
- `dL/draw_output`；
- parameter gradient。

不要再用 AMP 下 0.1% 左右的梯度随机差异来判断公式对错。你之前已经发现相同 loss 重算都会出现类似量级的 mixed-precision gradient 波动。

正确做法：

```text
数学等价测试：FP32/no autocast
生产训练：原 AMP 设置
```

### calibration cancellation

对于同组成 pair，如果后处理 calibration 是纯组成项，那么：

\[
(\hat E_i^{cal}-\hat E_j^{cal})
-
(\hat E_i^{raw}-\hat E_j^{raw})
\approx0.
\]

随机抽 pair 自动验证。

---

# 八、我建议最终目录固定成这样

```text
experiment_reference_allocation_final/
│
├── PROTOCOL.md
├── config.yaml
├── manifests/
│   ├── dataset_hashes.json
│   ├── checkpoint_hashes.json
│   └── reference_coefficients.json
│
├── eqv3_final_eval/
│   ├── predictions/
│   ├── metrics/
│   ├── bootstrap/
│   └── audit.json
│
├── required_vs_induced/
│   ├── sevennet_pairs.csv
│   ├── eqv3_pairs.csv
│   ├── summary.csv
│   └── figures/
│
├── synthetic_reference_control/
│   ├── synthetic_labels/
│   ├── seed_*/
│   ├── epoch_metrics.csv
│   ├── force_drift.csv
│   ├── pair_drift.csv
│   └── audit.json
│
├── cross_protocol_summary/
│   └── summary.csv
│
└── FINAL_AUDIT.md
```

---

# 九、哪些绝对不要再做

我建议现在明确停止下面这些方向：

- 不加第三个 backbone；
- 不再做更多 learning-rate sweep；
- 不再扩 loss；
- 不再做 ranking loss；
- 不再做 latent Δ-learning；
- 不再扩 WBM；
- 不再做 hull-aware training；
- 不再做 adapter bank；
- 不再做 residual-transfer 7×7；
- 不再为了 synthetic 结果不好临时换 optimizer。

因为这些已经不再增加当前主线的证明力。

你的旧工作里已经证明过：更强 latent representation 本身就能很好地处理 correction，而且额外 \(\Delta E\) loss 不一定继续带来收益。粘贴的文本 (1) 这本身就是一个很好的理由，不应该再往“多加一个 thermodynamic loss”方向扩。

---

# 十、最后给你一个最简执行顺序

真正执行时，就按这个顺序：

1. **写统一 protocol + audit 脚本。**
2. **恢复 GPU 后完成 EqV3 final test evaluation。**
3. **用 SevenNet 和 EqV3 prediction 做 required-vs-induced 后处理。**
4. **构造 synthetic pure-reference labels，先跑全部单元测试。**
5. **先跑一个 seed 做程序 smoke test，只检查程序，不看结果决定是否继续。**
6. **程序检查通过后固定跑满 3 seeds。**
7. **计算 synthetic EC drift + force drift + training trajectory。**
8. **整理 MP→r²SCAN 既有结果，不重新训练。**
9. **运行最终 audit。**
10. **停止实验，开始写论文。**

其中真正新增 GPU 训练只有第 6 步的 synthetic control。

如果这一套全部完成，你这篇论文最关键的四个层次就都有了：

> **真实现象 → 数学诊断 → 受控因果实验 → 跨模型/跨协议验证。**

从证据结构上已经足够完整，不需要再靠“多跑几个模型”堆可信度。