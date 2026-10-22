# Final numerical audit

All checks below were executed; none are pre-marked.

- PASS — frozen protocol/config: `5637cdd2cae838c6f4f6adc7a18d6e00f99f1f98d5c9f5c2f9517b32aece5d32`
- PASS — FP32/no-autocast physical and encoded loss/gradient equivalence: `{'loss_abs_difference': 2.9802322387695312e-08, 'raw_gradient_max_abs_difference': 3.5529956221580505e-07, 'parameter_gradient_max_abs_difference': 7.152557373046875e-07}`
- PASS — component audit: eqv3: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/eqv3_final_eval/audit.json`
- PASS — component audit: sevennet: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/sevennet_final_eval_v2/audit.json`
- PASS — component audit: required_vs_induced: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/required_vs_induced/audit.json`
- PASS — component audit: synthetic_labels: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/synthetic_reference_control/labels_v1/audit.json`
- PASS — component audit: resume: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/synthetic_reference_control/resume_accept/RESUME_AUDIT.json`
- PASS — component audit: synthetic_energy: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/synthetic_reference_control/final_energy/audit.json`
- PASS — component audit: synthetic_force: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/synthetic_reference_control/final_force/audit.json`
- PASS — component audit: synthetic_analysis: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/synthetic_reference_control/final_analysis/audit.json`
- PASS — component audit: cross_protocol: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/cross_protocol_summary/audit.json`
- PASS — component audit: closure_postprocessing: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/closure_postprocessing/audit.json`
- PASS — component audit: publication_lock: `/home/yxzhang/energy/experiment_reference_allocation_final_20261004/publication_lock/audit.json`
- PASS — synthetic split identity/disjointness: `{'train': 8192, 'test': 2048, 'val': 1024}`
- PASS — synthetic epoch-0 aligned/direct centered prediction identity: `8.881784197001252e-16`
- PASS — nine fixed 120-epoch runs and full resume states: `c53990a996404d163f1831b6eeb207d6ab11b8d09ab75c7cb86032cab25f0afe`
- PASS — EqV3 paired initialization and production loss gates: `{'paired_initialization_hashes': {20260528: 'b3120e2e73952dc920490e3dfafcb11a5b4da38a388f7e2d03cfd0c1bc2d21da', 20260529: '2f1fd1d4dd23eb95770fe26dacacad77403ec63a817b20bdcfdd75312eb68d47', 20260530: 'd0f9a095b56529eb4608c974f147bc43eff86232121c63b5f4287ca478399778'}, 'maximum_within_group_spread_ev_atom': 3.0531133177191805e-16, 'maximum_loss_abs_difference': 3.552713678800501e-15, 'maximum_raw_gradient_abs_difference': 0.0, 'maximum_gradient_of_loss_difference_relative_norm': 0.0}`
- PASS — SevenNet paired initialization and epoch-0 group identity: `{'paired_initialization_hashes': {20260528: 'c53990a996404d163f1831b6eeb207d6ab11b8d09ab75c7cb86032cab25f0afe', 20260529: 'c53990a996404d163f1831b6eeb207d6ab11b8d09ab75c7cb86032cab25f0afe', 20260530: 'c53990a996404d163f1831b6eeb207d6ab11b8d09ab75c7cb86032cab25f0afe'}, 'maximum_within_group_reference_spread_ev_atom': 0.0}`
- PASS — EqV3/SevenNet prediction identity and finiteness: `14616 rows each`
- PASS — synthetic post-calibration same-composition cancellation: `1.3322676295501878e-15`
- PASS — force drift and finite difference tables: `128 finite differences`
- PASS — final reports, tables and figures: `17`
- PASS — final-report terminology and canonical-number lock: `fixed confirmation; 90/214/202; direct 83.667`

## Interpretation boundary

PASS certifies the frozen definitions, artifacts, IDs, checkpoint states and numerical invariants above. It does not by itself establish a scientific mechanism or guarantee that aligned must outperform direct.

AUDIT_OK
