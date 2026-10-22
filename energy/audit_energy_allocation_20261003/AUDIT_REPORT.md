# Audit report: energy-only fine-tuning and correction allocation

## Scope and experiment lock

Audited only the fixed Experiment 3/6/8/9 chain. No training was launched, no original file was modified, and no paper result was rewritten. Input hashes and code versions are in `input_manifest.csv` and `environment.json`.

## 1. Confirmation-set independence — **manuscript wording error**

The 2,020-row downstream confirmation set is disjoint from downstream transfer fitting and alpha selection by structure ID, but it is **not independent of the earlier official model split**. Confirmation contains **1447 SevenNet official-train IDs** and **1447 EqV3 official-train IDs**. Those same rows entered the actual baseline element-reference fits and the post-adaptation train-only calibration whenever predictions were available.

Thus the statement that 'confirmation labels do not participate in calibration' is false if 'calibration' includes the baseline/post-adaptation official-train fits. It is true only for the later downstream composition fit. The set must not be described as a completely independent test set.

The actually unexposed remainder (excluding official train, official validation/model selection, baseline fit, post-adaptation fit, transfer fitting and alpha selection) is summarized below:

| model    |   n_fully_unexposed_structures |   n_repeated_exact_composition_groups |   n_ec_structures |   base_mae_mev_atom |   adapt_mae_mev_atom |   base_ec_mae_mev_atom |   adapt_ec_mae_mev_atom |
|:---------|-------------------------------:|--------------------------------------:|------------------:|--------------------:|---------------------:|-----------------------:|------------------------:|
| sevennet |                            362 |                                     8 |                23 |             123.772 |             122.923  |                8.93263 |                 15.8857 |
| eqv3     |                            362 |                                     8 |                23 |             123.955 |              10.7611 |                7.16404 |                  9.4789 |

Baseline refits reproduce stored predictions: True. This traces the fit to actual official-train rows, not names.

Detailed intersections by ID, reduced formula and chemical system are in `split_overlap.csv` and `intersections/intersection_ids.csv`.

## 2. Model-vs-protocol attribution — **implementation error plus insufficient architecture-only evidence**

The exact executable contracts are in `training_contracts.md`. SevenNet predicts direct per-atom formation energy with a per-atom SmoothL1 loss. EqV3 labels its route residual, but LoRA starts at zero while the pretrained absolute `energy_block` is left intact; the step-0 decoded prediction is therefore `E_ref + E_pretrained`, not `E_ref`. Treating the absolute pretrained branch as a residual is an **implementation/output-definition error** for a clean residual route. The saved 9.63 meV/atom result remains a valid measurement of the implemented route, but not of a correctly initialized residual route.

They also differ in trainable fraction, optimizer groups, epoch budget and precision. Matched within-SevenNet protocol contrasts are valid and summarized below; the full 21-row table is `matched_controls.csv`.

| checkpoint               | method              |   energy_mae_mev_atom |   exact_composition_mae_mev_atom |   absolute_mae_delta_vs_sevennet_base_mev_atom |   ec_mae_delta_vs_sevennet_base_mev_atom |
|:-------------------------|:--------------------|----------------------:|---------------------------------:|-----------------------------------------------:|-----------------------------------------:|
| sevennet_base            | base                |              113.744  |                          4.1896  |                                        0       |                              0           |
| last_two_blocks_lr0p0003 | vanilla             |               81.759  |                         17.0344  |                                      -31.9849  |                             12.8448      |
| l2sp                     | l2sp                |               81.6846 |                         17.0135  |                                      -32.0592  |                             12.8239      |
| replay_1                 | replay              |              105.31   |                          4.24777 |                                       -8.43355 |                              0.0581716   |
| recal_alpha_0            | recalibration_first |               38.3561 |                          4.1896  |                                      -75.3877  |                              1.33227e-14 |
| recal_alpha_0.05         | recalibration_first |               37.2524 |                          4.30141 |                                      -76.4915  |                              0.111809    |

There is no existing same-protocol different-model control matching physical output and loss definitions. The supported cross-model comparison is 'model + adaptation implementation', not architecture alone.

## 3. Energy/calibration/force consistency — **formula audit passed; requested new numerical cross-check incomplete**

All reported energies are per-atom eV in prediction tables; model forces must come from total energy `E=N e`. Element-reference and composition-only terms are position-independent at fixed composition. Therefore element-reference calibration leaves force unchanged, affine calibration scales force by its energy coefficient, and alpha interpolation gives `(1-alpha)F_base+alpha F_adapt`.

Existing raw-force magnitude statistics (not calibrated-gradient claims):

| model    |    n |   force_cosine_median |   force_difference_rmse_median_ev_ang |   base_force_rms_median_ev_ang |   adapt_force_rms_median_ev_ang |   force_norm_ratio_median |   force_norm_ratio_p05 |   force_norm_ratio_p95 |
|:---------|-----:|----------------------:|--------------------------------------:|-------------------------------:|--------------------------------:|--------------------------:|-----------------------:|-----------------------:|
| eqv3     |  525 |              0.999834 |                             0.0027895 |                      0.106537  |                        0.106907 |                  0.995584 |               0.964927 |                1.05452 |
| sevennet | 4250 |              0.425636 |                             0.463928  |                      0.0962367 |                        0.490909 |                  3.9736   |               1.04848  |               21.225   |

The old force probes used raw model calculator forces. They were not recomputed as gradients of every published calibrated/interpolated energy. The existing EqV3 finite-difference table has 100 structures at only one step (`1e-3 Å`); median relative error is 0.212. This does not meet the requested multi-step convergence check.

The NVIDIA driver is unavailable under the current kernel, so new real forward/autograd/multi-step finite differences were not fabricated. The missing checks remain explicitly unresolved. Local-perturbation provenance is: 500 stratified confirmation parents; EqV3 energies on 500 parents × 33 configurations = 16,500; forces on the first 75 parents × 7 retained configurations = 525 in the analyzed force table; random-displacement RMS values and exact generator are preserved in the cited probe script. The old gradient check used the first 100 parents, one deterministic zero-COM random direction per structure, and `eps=1e-3 Å`.

## 4. EqV3 'small error increase' versus actual change — **claim not supported by EC-MAE delta alone**

| model    |   n_groups |   n_structures |   T_rms_mev_atom |   C_rms_mev_atom |   T_norm_ev_atom |   C_norm_ev_atom |   C_over_T_norm |   pearson_T_C |   spearman_T_C |   alpha_EC_star |   scaled_R2_zero_centered |   base_ec_mae_mev_atom |   adapt_ec_mae_mev_atom |   ec_mae_delta_mev_atom |   base_pairwise_order_accuracy |   adapt_pairwise_order_accuracy |   base_lowest_structure_accuracy |   adapt_lowest_structure_accuracy |   C_over_T_norm_ci95_low |   C_over_T_norm_ci95_high |   pearson_T_C_ci95_low |   pearson_T_C_ci95_high |   spearman_T_C_ci95_low |   spearman_T_C_ci95_high |   alpha_EC_star_ci95_low |   alpha_EC_star_ci95_high |   scaled_R2_zero_centered_ci95_low |   scaled_R2_zero_centered_ci95_high |   base_ec_mae_mev_atom_ci95_low |   base_ec_mae_mev_atom_ci95_high |   adapt_ec_mae_mev_atom_ci95_low |   adapt_ec_mae_mev_atom_ci95_high |   ec_mae_delta_mev_atom_ci95_low |   ec_mae_delta_mev_atom_ci95_high |   base_pairwise_order_accuracy_ci95_low |   base_pairwise_order_accuracy_ci95_high |   adapt_pairwise_order_accuracy_ci95_low |   adapt_pairwise_order_accuracy_ci95_high |   base_lowest_structure_accuracy_ci95_low |   base_lowest_structure_accuracy_ci95_high |   adapt_lowest_structure_accuracy_ci95_low |   adapt_lowest_structure_accuracy_ci95_high |
|:---------|-----------:|---------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|--------------:|---------------:|----------------:|--------------------------:|-----------------------:|------------------------:|------------------------:|-------------------------------:|--------------------------------:|---------------------------------:|----------------------------------:|-------------------------:|--------------------------:|-----------------------:|------------------------:|------------------------:|-------------------------:|-------------------------:|--------------------------:|-----------------------------------:|------------------------------------:|--------------------------------:|---------------------------------:|---------------------------------:|----------------------------------:|---------------------------------:|----------------------------------:|----------------------------------------:|-----------------------------------------:|-----------------------------------------:|------------------------------------------:|------------------------------------------:|-------------------------------------------:|-------------------------------------------:|--------------------------------------------:|
| sevennet |         45 |            193 |          8.04644 |         28.2836  |        0.111785  |        0.392928  |        3.51504  |      0.33455  |     0.00156909 |       0.0951766 |                  0.111923 |                4.1896  |                17.0344  |               12.8448   |                       0.92542  |                        0.799895 |                         0.866667 |                          0.488889 |                 2.31597  |                   6.87906 |             -0.0958338 |                0.581357 |               -0.192969 |                 0.182335 |               -0.0179314 |                  0.229299 |                        0.000822954 |                            0.337976 |                         3.01423 |                          6.08189 |                         13.3331  |                          24.5184  |                        9.19605   |                          20.0587  |                                0.892422 |                                 0.939396 |                                 0.719251 |                                  0.8081   |                                  0.755556 |                                   0.955556 |                                   0.333333 |                                    0.644444 |
| eqv3     |         45 |            193 |          6.81055 |          2.69703 |        0.0946152 |        0.0374683 |        0.396007 |     -0.511439 |     0.137868   |      -1.29149   |                  0.26157  |                2.90974 |                 3.40221 |                0.492468 |                       0.941702 |                        0.93855  |                         0.955556 |                          0.911111 |                 0.258513 |                   0.77545 |             -0.859579  |                0.258219 |               -0.148786 |                 0.263765 |               -2.44736   |                  0.720335 |                        0.00016584  |                            0.738876 |                         1.71765 |                          4.48848 |                          2.16157 |                           5.67302 |                        0.0679611 |                           1.40365 |                                0.938136 |                                 0.977273 |                                 0.935402 |                                  0.966887 |                                  0.888889 |                                   1        |                                   0.822222 |                                    0.977778 |

`C_rms`, `C/T`, correlations and `alpha_EC*` directly quantify prediction change and alignment with the needed correction. EC-MAE delta instead quantifies only the change in error. They are not interchangeable. Ranking, lowest-energy selection, per-group values and complete-group bootstrap intervals are included in `within_group_summary.csv`, `within_group_group_metrics.csv`, and `within_group_corrections.csv`. Force cosine likewise does not replace force RMS/norm-ratio statistics; the existing force evidence is preserved without upgrading it to a calibrated-gradient claim.

## Classification summary

- Confirmation described as fully label-independent: **manuscript wording error**.
- EqV3 absolute pretrained output used as a residual at initialization: **implementation error**.
- Architecture-only attribution from SevenNet versus EqV3: **existing evidence insufficient**.
- Algebraic energy/calibration/force definitions: **passed**.
- Published raw-force metrics treated as gradients of every calibrated/interpolated energy: **existing evidence insufficient / wording must be qualified**.
- New multi-step finite-difference/autograd check: **unresolved due NVIDIA driver**, not passed.
- EqV3 actual-change inference from a small EC-MAE increase: **existing evidence insufficient**; use the new C/T/alignment statistics.

## Reproduction

```bash
/home/yxzhang/envs/fegx_benchmark214/bin/python /home/yxzhang/energy/audit_energy_allocation_20261003/code/audit_energy_allocation.py --out /home/yxzhang/energy/audit_energy_allocation_20261003
/home/yxzhang/envs/fegx_benchmark214/bin/python /home/yxzhang/energy/audit_energy_allocation_20261003/code/verify_audit.py /home/yxzhang/energy/audit_energy_allocation_20261003
```
