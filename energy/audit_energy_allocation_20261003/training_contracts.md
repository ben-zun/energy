# Training contracts (code-traced)

## Experiment identity

The audited main results are the `experiment3_20260922` / `experiment6_20260922` / `experiment8_20260923` / `experiment9_20260924` chain on `medium_same_split_269` (8192/1024/2048), not later Experiment 14 or 17 controls.

## SevenNet main model

- Training source: `/home/yxzhang/energy/experiment9_20260924/code/train_actual_main_multiseed.py`.
- Prediction: `e_raw(X)=E_model_total(X)/N_X`, in eV/atom.
- Target and loss: `SmoothL1(e_raw, y_form; beta=0.05 eV/atom)`.
- Trainable parameters: 301,300 / 842,623 (prefixes `3_`, `4_`, `reduce_input_to_hidden`, `reduce_hidden_to_energy`, `rescale_atomic_energy`).
- AdamW: LR 3e-4, weight decay 1e-5; batch 16; at most 20 epochs; patience 8; best checkpoint selected by official validation MAE.
- Post-training calibration: RidgeCV on official 8192 training rows predicts `y-e_raw` from composition/count features, then `e_adapt=e_raw+h_train-only(x,N)`.

## EqV3 main model

- Training source: `/home/yxzhang/energy/rclfa_v3/scripts/run_eqv3_native_upper_bound_smoke_20260725.py`; checkpoint `/home/yxzhang/energy/runs/rclfa_v3/eqv3_lora_fegx_multiseed_20260904/seed20260528/native/residual_lora_seed20260528_best.pt`.
- Frozen reference is `e_ref=pred_element_reference` from the older EqV3 reference table.  For each structure, `E_ref=N*e_ref`.
- The EqV3 energy branch emits a scaled **residual total energy**, `r_scaled`.  Decoding is `E_pred=E_ref+s*r_scaled`, where `s=model.avg_num_nodes` (`scale_divisor=1.0`); `e_pred=E_pred/N`.
- At initialization every LoRA `B` matrix is zero, so the backbone contribution is unchanged, but the pretrained `energy_block` is merely unfrozen: it is **not reset or zeroed**. Therefore the residual branch initially emits the pretrained absolute EqV3 total-energy output. The decoded step-0 prediction is `E_ref + E_pretrained`, not `E_ref`. This treats an absolute pretrained output as a residual and is an implementation/output-definition mismatch for a clean residual route.
- Training target is `(N*y_form-E_ref)/s`.  Loss is SmoothL1 in that scaled residual-total-energy coordinate with beta 0.05 (not the same physical loss coordinate as SevenNet).
- Tuning: LoRA plus energy head, 860,417 / 31,127,265 parameters (2.764%); LoRA rank 8, alpha 16; AdamW head/LoRA LR 2e-4, other trainable backbone LR 5e-5, weight decay 1e-4; batch 8; 12 requested epochs; validation-MAE checkpoint, patience 6; bfloat16 AMP.
- A second train-only element-reference calibration is then fitted on official train rows; the paper's EqV3 adapted column is that calibrated output.

## Inference and correction formulas

For the baseline element-reference calibration, `e_base=e_model-sum_Z x_Z mu_Z-b`, so `E_base=N*e_base` and `F_base=F_model`.  For affine calibration, `e_aff=a*e_model+sum_Z x_Z beta_Z+b`, hence `F_aff=a F_model` at fixed composition.

For geometry-free downstream correction, `e_value=e_base+h_target(x,N)`; its force is exactly `F_base` at fixed composition.  For an interpolation `e_alpha=e_base+h_target+alpha*(e_adapt-e_base-h_target)`, total energy is `E_alpha=N*e_alpha` and `F_alpha=(1-alpha)F_base+alpha F_adapt`; `h_target` contributes zero force.

## Attribution verdict

The current SevenNet/EqV3 comparison mixes backbone, trainable parameterization, trainable fraction, learning rates, epochs, mixed precision, output coordinate, and loss coordinate. In addition, the audited EqV3 residual route starts from `E_ref + E_pretrained` because the absolute pretrained energy head is not converted to a zero residual. Matched same-model SevenNet adaptation controls do exist and are tabulated in `matched_controls.csv`; they support claims about protocol effects within SevenNet. No existing **same-protocol different-model** control matches split, evaluation set, physical output definition and loss coordinate. Therefore the cross-model evidence compares **model + adaptation implementation**, not architecture alone.
