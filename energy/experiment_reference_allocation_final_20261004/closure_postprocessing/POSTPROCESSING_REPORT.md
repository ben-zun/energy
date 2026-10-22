# Final no-training post-processing

All analyses use frozen predictions and forces. No model was trained, selected, or refit.

## Paired cluster bootstrap

EC drift direct − aligned: 15.226 meV/atom (95% whole-group bootstrap CI 6.304 to 28.336; 22 groups, 10,000 draws).
EC drift aligned − zero: -0.005 meV/atom (95% CI -0.088 to 0.069).
Force drift direct − aligned: 0.74496 eV/angstrom (95% whole-structure bootstrap CI 0.52610 to 0.84493; 12 structures, 10,000 draws).
Force drift aligned − zero: -0.00033 eV/angstrom (95% CI -0.00129 to 0.00093).

A confidence interval overlapping zero is reported as no detected excess over the zero-shift background, not as proof of equivalence.

## Composition-direction projection

Primary values use native predictions before post-hoc composition calibration.

| Condition | beta | ||u_perp|| / ||h|| | corr(u,h) | residual RMS (meV/atom) |
|---|---:|---:|---:|---:|
| reference_aligned | 1.000052 ± 0.000008 | 0.000481 ± 0.000045 | 0.999999 ± 0.000000 | 2.287 ± 0.215 |
| reference_direct | 0.998613 ± 0.001182 | 0.022250 ± 0.000140 | 0.997847 ± 0.000028 | 105.775 ± 0.666 |

The orthogonal component means not explained by the known reference vector; it may include other composition-dependent directions as well as structure-dependent change and is therefore not labelled a pure PES component.

## Raw and post-calibrated results

The complete per-seed and mean±SD tables are saved as CSV. EC-MAE is numerically unchanged by the pure composition post-calibration (maximum difference recorded in audit).

## Reproducibility reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
