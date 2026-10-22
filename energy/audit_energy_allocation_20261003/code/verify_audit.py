#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

out=Path(sys.argv[1]).resolve()
required=["AUDIT_REPORT.md","split_overlap.csv","training_contracts.md","energy_force_checks.csv","within_group_corrections.csv","within_group_summary.csv","matched_controls.csv","input_manifest.csv","environment.json","calibration_reproduction.csv","fully_independent_evaluation.csv","intersections/intersection_ids.csv"]
missing=[x for x in required if not (out/x).exists()]
assert not missing, missing
ov=pd.read_csv(out/"split_overlap.csv")
assert set(ov.key_type)=={"id","reduced_formula","chemical_system"}
wg=pd.read_csv(out/"within_group_summary.csv")
assert set(wg.model)=={"sevennet","eqv3"} and wg.n_groups.eq(45).all() and wg.n_structures.eq(193).all()
vals=pd.read_csv(out/"within_group_corrections.csv")
assert len(vals)==386 and vals.groupby("model").size().eq(193).all()
for _,g in vals.groupby(["model","exact_composition"]):
    assert abs(g.T_ev_atom.mean())<1e-12 and abs(g.C_ev_atom.mean())<1e-12
cal=pd.read_csv(out/"calibration_reproduction.csv")
assert cal.reproduced.all() and cal.max_abs_prediction_difference_ev_atom.max()<1e-8
force=pd.read_csv(out/"energy_force_checks.csv")
fs=force[force.check.eq("existing_raw_force_distribution")]
force_cols=["force_cosine_median","force_difference_rmse_median_ev_ang","base_force_rms_median_ev_ang","adapt_force_rms_median_ev_ang","force_norm_ratio_median","force_norm_ratio_p05","force_norm_ratio_p95"]
assert set(fs.model)=={"sevennet","eqv3"} and np.isfinite(fs[force_cols].to_numpy(float)).all()
man=pd.read_csv(out/"input_manifest.csv",keep_default_na=False)
assert (man.sha256.str.len()==64).all(), man[man.sha256.str.len()!=64]
assert np.isfinite(wg.select_dtypes(include="number")).all().all()
report=(out/"AUDIT_REPORT.md").read_text()
for phrase in ["manuscript wording error","implementation error","existing evidence insufficient","formula audit passed","No training was launched"]: assert phrase in report
summary={"status":"verified","required_files":len(required),"overlap_rows":len(ov),"within_group_rows":len(vals),"calibration_reproduced":True,"numeric_finite":True}
(out/"verification.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
