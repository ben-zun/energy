# Reference-allocation experiments: final report

This report summarizes the frozen protocol executed on 2026-10-04. No result-dependent runs, epoch extensions, route deletion, or test-set selection were used.

## Audit outcome

- Final numerical audit: `AUDIT_OK`.
- Local and server unit tests: 10/10 passed with warnings treated as errors.
- Synthetic runs: 9/9 completed exactly 120 epochs; all four systemd workers ended successfully with zero restarts.
- Synthetic split: 8192 train / 1024 validation / 2048 test; IDs are disjoint.
- Checkpoint selection and post-calibration used validation and train data only, respectively.

## 1. Corrected EqV3 matched comparison

All values below are mean ± sample standard deviation across the three frozen seeds.

| Route | Overall calibrated MAE (meV/atom) | Fixed confirmation EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |
|---|---:|---:|---:|---:|---:|
| direct | 62.384 ± 1.681 | 27.879 ± 0.874 | 54.569 ± 1.314 | 0.684 ± 0.027 | 0.163 ± 0.019 |
| aligned | 17.789 ± 0.455 | 9.103 ± 0.881 | 17.154 ± 1.958 | 0.840 ± 0.013 | 0.028 ± 0.008 |

Under the corrected full physical decoder, aligned is better than direct for every seed on both absolute and same-composition endpoints. The paired whole-group/bootstrap tables are retained in `eqv3_final_eval/bootstrap/`.

## 2. SevenNet matched evaluation

| Route | Overall calibrated MAE (meV/atom) | Fixed confirmation EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |
|---|---:|---:|---:|---:|---:|
| direct | 69.911 ± 0.503 | 35.598 ± 0.218 | 67.171 ± 0.563 | 0.719 ± 0.008 | 0.109 ± 0.006 |
| aligned | 21.474 ± 0.160 | 12.490 ± 0.042 | 23.379 ± 0.100 | 0.844 ± 0.006 | 0.037 ± 0.005 |

## 3. Required versus induced within-group corrections

The frozen 190-group/238-pair list is shared across models, routes, and seeds. Intervals in the machine-readable table use 10,000 whole-group bootstrap draws.

| Model | Route | Excess MAE (meV/atom) | Norm ratio | Pearson | Projection α | Orthogonal norm ratio |
|---|---|---:|---:|---:|---:|---:|
| sevennet | direct | 67.171 ± 0.563 | 2.641 ± 0.016 | 0.172 ± 0.012 | 0.498 ± 0.026 | 2.593 ± 0.021 |
| sevennet | aligned | 23.379 ± 0.100 | 0.741 ± 0.007 | 0.471 ± 0.005 | 0.358 ± 0.000 | 0.649 ± 0.008 |
| eqv3 | direct | 54.569 ± 1.314 | 1.670 ± 0.096 | 0.424 ± 0.065 | 0.685 ± 0.062 | 1.520 ± 0.133 |
| eqv3 | aligned | 17.154 ± 1.958 | 1.030 ± 0.119 | 0.766 ± 0.040 | 0.791 ± 0.065 | 0.657 ± 0.118 |

Aligned corrections are much closer to the corrections required by the target. This supports a statement about correction allocation; it does not imply that every change to a potential-energy surface is intrinsically wrong.

## 4. Synthetic pure-reference control

Synthetic targets are the frozen pretrained SevenNet physical energy plus a pure train-fitted composition reference. The reference has no coordinate dependence. The test set contains 2048 structures; same-composition endpoints use 22 complete groups, 52 structures, and 53 frozen pairs.

| Condition | Selected epochs | Calibrated MAE (meV/atom) | EC drift (meV/atom) | Pair drift (meV/atom) | Force drift RMSE (eV/Å) | Force cosine |
|---|---|---:|---:|---:|---:|---:|
| zero_direct | 13, 7, 15 | 0.902 ± 0.052 | 0.168 ± 0.035 | 0.207 ± 0.023 | 0.00415 ± 0.00037 | 0.99980 ± 0.00004 |
| reference_direct | 119, 119, 119 | 53.990 ± 0.046 | 15.389 ± 0.197 | 18.770 ± 0.304 | 0.74877 ± 0.00957 | 0.33194 ± 0.02660 |
| reference_aligned | 4, 12, 19 | 0.995 ± 0.052 | 0.163 ± 0.051 | 0.210 ± 0.041 | 0.00382 ± 0.00087 | 0.99983 ± 0.00009 |

The aligned condition stays at the same energy/EC/force-drift scale as the zero-shift optimizer background. Direct training leaves a roughly 54 meV/atom absolute residual, induces about 15.4 meV/atom EC drift, and changes forces by about 0.75 eV/Å RMS.

Force-definition verification used physical total energy, 12 SHA256-selected test structures, and central differences for 8 structures × 4 models × 4 steps. All 32 components passed; the best-step worst absolute error was 0.001636 eV/Å.

## 5. Synthetic paired uncertainty and composition-direction projection

Whole-group bootstrap gives direct − aligned EC drift = 15.226 meV/atom (95% CI 6.304 to 28.336); aligned − zero is -0.005 (95% CI -0.088 to 0.069).
Whole-structure bootstrap gives direct − aligned force drift = 0.74496 eV/Å (95% CI 0.52610 to 0.84493); aligned − zero is -0.00033 (95% CI -0.00129 to 0.00093).

The aligned-minus-zero intervals include zero; this is reported as no detected excess over background, not as proof of equivalence.

| Condition | Projection β | ||u⊥|| / ||h|| | corr(u,h) | Residual RMS (meV/atom) |
|---|---:|---:|---:|---:|
| reference_aligned | 1.000052 ± 0.000008 | 0.000481 ± 0.000045 | 0.999999 ± 0.000000 | 2.287 ± 0.215 |
| reference_direct | 0.998613 ± 0.001182 | 0.022250 ± 0.000140 | 0.997847 ± 0.000028 | 105.775 ± 0.666 |

Direct training learns nearly the full global reference direction (β≈1) but simultaneously leaves a much larger component orthogonal to the known reference. That orthogonal component is described only as unexplained by h: it can contain other composition directions as well as structure-dependent change.

## 6. Raw versus post-calibrated supplementary summary

| Model | Route | Raw MAE | Post-calibrated MAE | EC-MAE before/after |
|---|---|---:|---:|---:|
| EqV3 | aligned | 19.465 ± 0.623 | 17.789 ± 0.455 | 9.103 / 9.103 |
| EqV3 | direct | 67.702 ± 3.446 | 62.384 ± 1.681 | 27.879 / 27.879 |
| SevenNet | aligned | 21.523 ± 0.256 | 21.474 ± 0.160 | 12.490 / 12.490 |
| SevenNet | direct | 70.559 ± 0.587 | 69.911 ± 0.503 | 35.598 / 35.598 |
| SevenNet-synthetic | reference_aligned | 1.195 ± 0.117 | 0.995 ± 0.052 | 0.163 / 0.163 |
| SevenNet-synthetic | reference_direct | 54.744 ± 0.184 | 53.990 ± 0.046 | 15.389 / 15.389 |
| SevenNet-synthetic | zero_direct | 1.014 ± 0.050 | 0.902 ± 0.052 | 0.168 / 0.168 |

Pure composition post-calibration changes absolute MAE but leaves EC-MAE unchanged to numerical precision.

## 7. Existing MP→r²SCAN summary

No retraining was performed. The current hashed machine-readable artifacts contain 90 groups / 214 structures / 202 pairs; an older prose report saying 85/209 is stale.

| Route | Absolute MAE | EC-MAE | Pair ΔE-MAE | Top-1 | Major misselection |
|---|---:|---:|---:|---:|---:|
| direct | 83.667 ± 0.114 | 17.943 ± 0.579 | 24.175 ± 0.662 | 0.463 ± 0.006 | 0.185 ± 0.006 |
| aligned | 23.943 ± 0.092 | 8.754 ± 0.094 | 12.979 ± 0.071 | 0.744 ± 0.000 | 0.063 ± 0.006 |

## Interpretation and limits

The corrected EqV3 and matched SevenNet results both support aligned allocation. The synthetic experiment supplies the clean causal control: a composition-only target shift causes large unwanted same-composition and force changes under direct training, whereas aligned training remains at the zero-shift optimizer background.

Limits: the EqV3/SevenNet fixed confirmation cohort was previously inspected and is not a new blind test; the synthetic same-composition test has only 22 groups; force drift is evaluated on 12 fixed structures and is not a force error against DFT labels; and the conclusions concern these fixed models, datasets, and adaptation protocols.

All detailed predictions, pair lists, bootstrap tables, force vectors, finite-difference checks, histories, selected checkpoint hashes, and the frozen protocol remain under this experiment directory.
