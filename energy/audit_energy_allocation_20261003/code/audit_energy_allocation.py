#!/usr/bin/env python3
"""Read-only audit for the 2026-09 energy-only adaptation experiment chain.

This script never writes outside --out.  It deliberately identifies the paper
experiment by fixed paths and numerical fingerprints instead of searching for a
newer/better run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/yxzhang/energy")
SEED = 20261003
NBOOT = 10000

DATASET = ROOT / "runs/rclfa_v3/task_specific_reference_299_308/medium_same_split_269"
MANIFEST = ROOT / "experiment0_20260920/freeze/split_manifest.csv"
MASTER = ROOT / "experiment0_20260920/freeze/master_structures.parquet"
SEVEN_BASE = ROOT / "runs/rclfa_v3/real_fegx_foundation_replacement_272/reference_conversion/sevennet_reference/calibrated_predictions.csv"
SEVEN_COEF = SEVEN_BASE.with_name("calibration_coefficients.csv")
EQ_BASE = ROOT / "runs/rclfa_v3/same_subset_real_fegx_sanity_271/eqv3_subset_reference/calibrated_predictions.csv"
EQ_COEF = EQ_BASE.with_name("calibration_coefficients.csv")
SEVEN_RAW_ADAPT = ROOT / "runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision/fair_finetune_293/sweep/last_two_blocks_lr0p0003/predictions.csv"
EQ_ADAPT = ROOT / "runs/rclfa_v3/eqv3_lora_fegx_multiseed_20260904/seed20260528/lora_primary_calibration/calibrated_predictions.csv"
EQ_ADAPT_COEF = EQ_ADAPT.with_name("calibration_coefficients.csv")
SEVEN_CKPT = ROOT / "runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision/fair_finetune_293/sweep/last_two_blocks_lr0p0003/best.pt"
EQ_CKPT = ROOT / "runs/rclfa_v3/eqv3_lora_fegx_multiseed_20260904/seed20260528/native/residual_lora_seed20260528_best.pt"
EQ_SUMMARY = EQ_CKPT.with_name("residual_lora_seed20260528_summary.json")
SEVEN_TRAINER = ROOT / "experiment9_20260924/code/train_actual_main_multiseed.py"
EQ_TRAINER = ROOT / "rclfa_v3/scripts/run_eqv3_native_upper_bound_smoke_20260725.py"
MECH_SCRIPT = ROOT / "experiment3_20260922/code/experiment3_mechanism.py"
REF_SCRIPT = ROOT / "rclfa_v3/reference.py"
FD_EXISTING = ROOT / "experiment1_20260922/analysis/E3_gradient_consistency.csv"
FORCE_EQ = ROOT / "experiment1_20260922/analysis/E3_force_preservation.csv"
FORCE_SEVEN = ROOT / "experiment1_20260922/e6/forces_sevennet_base.csv"
FORCE_SEVEN_ADAPT = ROOT / "experiment1_20260922/e6/forces_sevennet_adapted.csv"
PROBE500 = ROOT / "experiment1_20260922/confirmation_probe500.csv"
EXISTING_CORRECTIONS = ROOT / "experiment3_20260922/analysis/adaptation_correction_values.csv"
SEVEN_CONTROLS = ROOT / "experiment6_20260922/analysis/exact_composition_relative_energy.csv"

INPUTS = [
    MANIFEST, MASTER, SEVEN_BASE, SEVEN_COEF, EQ_BASE, EQ_COEF,
    SEVEN_RAW_ADAPT, EQ_ADAPT, EQ_ADAPT_COEF, EQ_SUMMARY,
    SEVEN_TRAINER, EQ_TRAINER, MECH_SCRIPT, REF_SCRIPT,
    FD_EXISTING, FORCE_EQ, FORCE_SEVEN, FORCE_SEVEN_ADAPT, PROBE500, EXISTING_CORRECTIONS, SEVEN_CONTROLS,
    DATASET / "train/train.csv", DATASET / "val/val.csv", DATASET / "test/test.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN", "None"])


def chem_features(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for formula, natoms in zip(d.formula, d.natoms):
        c = Composition(str(formula)); total = float(c.num_atoms)
        row = {f"frac_Z{z}": c.get_atomic_fraction(Element.from_Z(z)) for z in range(1, 119)}
        row.update({f"count_Z{z}": c[Element.from_Z(z)] / total * float(natoms) for z in range(1, 119)})
        row["n_atoms"] = float(natoms); row["n_elements"] = float(len(c.elements))
        rows.append(row)
    return pd.DataFrame(rows, index=d.index).fillna(0.0)


def add_chem(d: pd.DataFrame) -> pd.DataFrame:
    q = d.copy()
    q["id"] = q["id"].astype(str)
    q["reduced_formula"] = q["formula"].map(lambda x: Composition(str(x)).reduced_formula)
    q["chemical_system"] = q["formula"].map(lambda x: "-".join(sorted(str(e) for e in Composition(str(x)).elements)))
    return q


def load_dataset() -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        q = read_csv(DATASET / split / f"{split}.csv")
        assert (q.split == split).all()
        parts.append(q.rename(columns={"formation_energy_per_atom_ev": "y_true"}))
    d = add_chem(pd.concat(parts, ignore_index=True))
    assert d.id.is_unique and d.split.value_counts().to_dict() == {"train": 8192, "test": 2048, "val": 1024}
    return d


def load_predictions(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sb = add_chem(read_csv(SEVEN_BASE))
    eb = add_chem(read_csv(EQ_BASE))
    sr = add_chem(read_csv(SEVEN_RAW_ADAPT))
    ea = add_chem(read_csv(EQ_ADAPT))
    # Reuse the already-frozen Experiment 3 confirmation predictions.  Do not
    # rerun RidgeCV or select a new parameter in this audit.
    frozen = read_csv(EXISTING_CORRECTIONS)
    frozen = frozen[(frozen.model == "sevennet") & (frozen.recipe == "last_two_blocks")][["id", "adapt"]]
    assert len(frozen) == 2020 and frozen.id.is_unique
    sr = sr.merge(frozen, on="id", how="left", validate="one_to_one")
    ea["adapt"] = ea.pred_element_reference
    out = {"seven_base": sb, "eq_base": eb, "seven_adapt": sr, "eq_adapt": ea}
    assert sb.id.is_unique and eb.id.is_unique and sr.id.is_unique and ea.id.is_unique
    assert len(sr) == len(dataset) == 11264 and len(ea) == 11264 and sr.adapt.notna().sum() == 2020
    return out


def exact_key(d: pd.DataFrame) -> pd.Series:
    rows=[]
    for formula,natoms in zip(d.formula,d.natoms):
        c=Composition(str(formula)); factor=float(natoms)/float(c.num_atoms); row=np.zeros(118,dtype=int)
        for element,amount in c.items(): row[element.Z-1]=int(round(float(amount)*factor))
        rows.append(row)
    counts=np.asarray(rows,dtype=int)
    assert np.array_equal(counts.sum(axis=1), d.natoms.astype(int).to_numpy())
    return pd.Series(["|".join(map(str, row)) for row in counts], index=d.index)


def cohorts(dataset: pd.DataFrame, pred: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    manifest = read_csv(MANIFEST).rename(columns={"structure_id": "id"})
    manifest.id = manifest.id.astype(str)
    meta = dataset[["id", "formula", "reduced_formula", "chemical_system", "split"]].copy()
    # The downstream manifest is authoritative for its precomputed reduced
    # formula/system; only add the full formula needed for exact count vectors.
    m = manifest.merge(meta[["id", "formula"]], on="id", how="left", validate="one_to_one")
    assert m.formula.notna().all() and manifest.id.is_unique
    sets: dict[str, pd.DataFrame] = {}
    for model in ("sevennet", "eqv3"):
        for split in ("train", "val", "test"):
            sets[f"{model}_model_{split}"] = meta[meta.split.eq(split)].copy()
    # Actual fit rows: official train rows with finite source prediction.
    for model, b in (("sevennet", pred["seven_base"]), ("eqv3", pred["eq_base"])):
        ok = b.split.eq("train") & np.isfinite(pd.to_numeric(b.model_energy_per_atom, errors="coerce")) & np.isfinite(pd.to_numeric(b.y_true, errors="coerce"))
        sets[f"{model}_baseline_calibration_fit"] = b.loc[ok, ["id", "formula", "reduced_formula", "chemical_system"]].copy()
    for model, a in (("sevennet", pred["seven_adapt"]), ("eqv3", pred["eq_adapt"])):
        # These are actual fitting-input IDs. SevenNet's saved audit prediction
        # table contains confirmation rows only, so availability is traced from
        # the raw checkpoint prediction rather than the confirmation-only merge.
        availability = a.pred_sevennet_finetuned if model == "sevennet" else a.adapt
        ok = a.split.eq("train") & np.isfinite(pd.to_numeric(availability, errors="coerce"))
        sets[f"{model}_postadapt_calibration_fit"] = a.loc[ok, ["id", "formula", "reduced_formula", "chemical_system"]].copy()
    for purpose, label in (("transfer_train", "composition_fit"), ("relation", "alpha_selection"), ("confirmation", "confirmation")):
        sets[label] = m[m.purpose.eq(purpose)][["id", "formula", "reduced_formula", "chemical_system"]].copy()
    # Exact-composition evaluation comes from confirmation only, count vector repeated >=2.
    c = sets["confirmation"].copy(); c["exact_composition"] = exact_key(c.assign(natoms=c.id.map(dataset.set_index("id").natoms)))
    sizes = c.groupby("exact_composition").size()
    ex = c[c.exact_composition.isin(sizes[sizes >= 2].index)].copy()
    assert ex.exact_composition.nunique() == 45 and len(ex) == 193
    sets["exact193"] = ex
    return sets, manifest


def overlap_audit(out: Path, sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows, detail = [], []
    names = sorted(sets)
    lookup = {}
    for name in names:
        for key in ("id", "reduced_formula", "chemical_system"):
            lookup[(name,key)] = {str(value): sorted(g.id.astype(str).unique()) for value,g in sets[name].groupby(key,sort=False)}
    for a, b in combinations(names, 2):
        for key in ("id", "reduced_formula", "chemical_system"):
            ma=lookup[(a,key)]; mb=lookup[(b,key)]; va=set(ma); vb=set(mb); common = sorted(va & vb)
            rows.append({"set_a": a, "set_b": b, "key_type": key, "n_a": len(sets[a]), "n_b": len(sets[b]), "n_unique_a": len(va), "n_unique_b": len(vb), "n_intersection": len(common)})
            for value in common:
                detail.append({"set_a": a, "set_b": b, "key_type": key, "key_value": value, "structure_ids_a": ";".join(ma[value]), "structure_ids_b": ";".join(mb[value])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "split_overlap.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "intersections/intersection_ids.csv", index=False)
    return summary


def calibration_reproduction(out: Path, pred: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from rclfa_v3.reference import fit_calibrators
    rows = []
    for model, d, coef_path in (("sevennet", pred["seven_base"], SEVEN_COEF), ("eqv3", pred["eq_base"], EQ_COEF)):
        q = d.copy()
        q["model_energy_per_atom"] = pd.to_numeric(q.model_energy_per_atom, errors="coerce")
        q["y_true"] = pd.to_numeric(q.y_true, errors="coerce")
        q = q[np.isfinite(q.model_energy_per_atom) & np.isfinite(q.y_true)].reset_index(drop=True)
        fractions=np.zeros((len(q),118),dtype=float)
        for i,formula in enumerate(q.formula):
            c=Composition(str(formula))
            for element,amount in c.items(): fractions[i,element.Z-1]=float(amount)/float(c.num_atoms)
        assert np.allclose(fractions.sum(axis=1),1.0,atol=1e-12)
        q=pd.concat([q,pd.DataFrame(fractions,columns=[f"x_Z{z}" for z in range(1,119)])],axis=1)
        fits = {x.name: x for x in fit_calibrators(q, energy_column="model_energy_per_atom", target_column="y_true", ridge=1e-6)}
        stored = read_csv(coef_path)
        for name, col in (("element_reference", "pred_element_reference"), ("affine_energy_composition", "pred_affine_energy_composition")):
            fit = fits[name]
            got = fit.all_predictions
            expected = pd.to_numeric(q[col], errors="coerce").to_numpy(float)
            c = stored[stored.calibration.eq(name)].set_index("coefficient").value.astype(float).to_dict()
            keys = sorted(set(c) | set(fit.coefficients))
            max_coeff = max(abs(c.get(k, np.nan) - fit.coefficients.get(k, np.nan)) for k in keys)
            rows.append({"model": model, "calibration": name, "actual_fit_rows": int((q.split == "train").sum()), "max_abs_prediction_difference_ev_atom": float(np.max(np.abs(got - expected))), "max_abs_coefficient_difference": float(max_coeff), "energy_scale_a": 1.0 if name == "element_reference" else float(fit.coefficients["alpha_energy"]), "reproduced": bool(np.allclose(got, expected, atol=1e-9, rtol=1e-9))})
    r = pd.DataFrame(rows); r.to_csv(out / "calibration_reproduction.csv", index=False); return r


def ec_values(frame: pd.DataFrame, base_col: str, adapt_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = frame.copy()
    gy = q.groupby("exact_composition").y_true.transform("mean")
    gb = q.groupby("exact_composition")[base_col].transform("mean")
    ga = q.groupby("exact_composition")[adapt_col].transform("mean")
    q["T_ev_atom"] = (q.y_true - q[base_col]) - (gy - gb)
    q["C_ev_atom"] = (q[adapt_col] - q[base_col]) - (ga - gb)
    q["base_centered_error_ev_atom"] = -q.T_ev_atom
    q["adapt_centered_error_ev_atom"] = q.C_ev_atom - q.T_ev_atom
    return q, group_metrics(q)


def ordering_stats(g: pd.DataFrame, col: str) -> tuple[int, int]:
    yt = g.y_true.to_numpy(float); yp = g[col].to_numpy(float)
    ok = total = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            st = np.sign(yt[i] - yt[j]); sp = np.sign(yp[i] - yp[j])
            if st == 0: continue
            total += 1; ok += int(st == sp)
    return ok, total


def group_metrics(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in q.groupby("exact_composition"):
        bo, bt = ordering_stats(g, "base")
        ao, at = ordering_stats(g, "adapt")
        rows.append({"exact_composition": key, "n_structures": len(g), "T_rms_mev_atom": float(np.sqrt(np.mean(g.T_ev_atom**2))*1000), "C_rms_mev_atom": float(np.sqrt(np.mean(g.C_ev_atom**2))*1000), "C_over_T_norm": float(np.linalg.norm(g.C_ev_atom)/np.linalg.norm(g.T_ev_atom)) if np.linalg.norm(g.T_ev_atom) else np.nan, "base_ec_mae_mev_atom": float(np.mean(np.abs(g.base_centered_error_ev_atom))*1000), "adapt_ec_mae_mev_atom": float(np.mean(np.abs(g.adapt_centered_error_ev_atom))*1000), "base_pairwise_order_correct": bo, "base_pairwise_order_total": bt, "adapt_pairwise_order_correct": ao, "adapt_pairwise_order_total": at, "base_lowest_correct": int(g.loc[g.base.idxmin(), "id"] == g.loc[g.y_true.idxmin(), "id"]), "adapt_lowest_correct": int(g.loc[g.adapt.idxmin(), "id"] == g.loc[g.y_true.idxmin(), "id"])})
    return pd.DataFrame(rows)


def overall_stats(q: pd.DataFrame) -> dict[str, float]:
    T = q.T_ev_atom.to_numpy(float); C = q.C_ev_atom.to_numpy(float)
    denom = float(C @ C); alpha = float(T @ C / denom) if denom else np.nan
    gm = group_metrics(q)
    return {
        "n_groups": int(q.exact_composition.nunique()), "n_structures": len(q),
        "T_rms_mev_atom": float(np.sqrt(np.mean(T*T))*1000), "C_rms_mev_atom": float(np.sqrt(np.mean(C*C))*1000),
        "T_norm_ev_atom": float(np.linalg.norm(T)), "C_norm_ev_atom": float(np.linalg.norm(C)), "C_over_T_norm": float(np.linalg.norm(C)/np.linalg.norm(T)),
        "pearson_T_C": float(pearsonr(T,C).statistic), "spearman_T_C": float(spearmanr(T,C).statistic),
        "alpha_EC_star": alpha, "scaled_R2_zero_centered": float(1-np.sum((T-alpha*C)**2)/np.sum(T*T)),
        "base_ec_mae_mev_atom": float(np.mean(np.abs(T))*1000), "adapt_ec_mae_mev_atom": float(np.mean(np.abs(C-T))*1000),
        "ec_mae_delta_mev_atom": float((np.mean(np.abs(C-T))-np.mean(np.abs(T)))*1000),
        "base_pairwise_order_accuracy": float(gm.base_pairwise_order_correct.sum()/gm.base_pairwise_order_total.sum()),
        "adapt_pairwise_order_accuracy": float(gm.adapt_pairwise_order_correct.sum()/gm.adapt_pairwise_order_total.sum()),
        "base_lowest_structure_accuracy": float(gm.base_lowest_correct.mean()), "adapt_lowest_structure_accuracy": float(gm.adapt_lowest_correct.mean()),
    }


def bootstrap_groups(q: pd.DataFrame, seed: int) -> dict[str, tuple[float,float]]:
    groups = []
    for _, g in q.groupby("exact_composition"):
        bo, bt = ordering_stats(g, "base"); ao, at = ordering_stats(g, "adapt")
        groups.append({"T": g.T_ev_atom.to_numpy(float), "C": g.C_ev_atom.to_numpy(float),
                       "bo": bo, "bt": bt, "ao": ao, "at": at,
                       "bl": int(g.loc[g.base.idxmin(), "id"] == g.loc[g.y_true.idxmin(), "id"]),
                       "al": int(g.loc[g.adapt.idxmin(), "id"] == g.loc[g.y_true.idxmin(), "id"])})
    rng = np.random.default_rng(seed); keys = ["C_over_T_norm", "pearson_T_C", "spearman_T_C", "alpha_EC_star", "scaled_R2_zero_centered", "base_ec_mae_mev_atom", "adapt_ec_mae_mev_atom", "ec_mae_delta_mev_atom", "base_pairwise_order_accuracy", "adapt_pairwise_order_accuracy", "base_lowest_structure_accuracy", "adapt_lowest_structure_accuracy"]
    vals = {k: [] for k in keys}
    for _ in range(NBOOT):
        sampled = rng.integers(0, len(groups), len(groups))
        T=np.concatenate([groups[i]["T"] for i in sampled]); C=np.concatenate([groups[i]["C"] for i in sampled])
        nt=float(np.linalg.norm(T)); nc=float(np.linalg.norm(C)); denom=float(C@C)
        alpha=float(T@C/denom) if denom else np.nan
        bo=sum(groups[i]["bo"] for i in sampled); bt=sum(groups[i]["bt"] for i in sampled)
        ao=sum(groups[i]["ao"] for i in sampled); at=sum(groups[i]["at"] for i in sampled)
        s={"C_over_T_norm":nc/nt,"pearson_T_C":float(pearsonr(T,C).statistic),"spearman_T_C":float(spearmanr(T,C).statistic),"alpha_EC_star":alpha,"scaled_R2_zero_centered":float(1-np.sum((T-alpha*C)**2)/np.sum(T*T)),"base_ec_mae_mev_atom":float(np.mean(np.abs(T))*1000),"adapt_ec_mae_mev_atom":float(np.mean(np.abs(C-T))*1000),"ec_mae_delta_mev_atom":float((np.mean(np.abs(C-T))-np.mean(np.abs(T)))*1000),"base_pairwise_order_accuracy":bo/bt,"adapt_pairwise_order_accuracy":ao/at,"base_lowest_structure_accuracy":float(np.mean([groups[i]["bl"] for i in sampled])),"adapt_lowest_structure_accuracy":float(np.mean([groups[i]["al"] for i in sampled]))}
        for k in keys: vals[k].append(s[k])
    return {k:(float(np.nanquantile(v,.025)),float(np.nanquantile(v,.975))) for k,v in vals.items()}


def within_group(out: Path, sets: dict[str, pd.DataFrame], pred: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries=[]; all_values=[]; all_groups=[]
    ids=set(sets["exact193"].id)
    for j, model in enumerate(("sevennet","eqv3")):
        b=pred["seven_base" if model=="sevennet" else "eq_base"]
        a=pred["seven_adapt" if model=="sevennet" else "eq_adapt"]
        q=b[b.id.isin(ids)][["id","formula","natoms","y_true","pred_element_reference"]].rename(columns={"pred_element_reference":"base"})
        q=q.merge(a[["id","adapt"]],on="id",validate="one_to_one")
        q["exact_composition"]=exact_key(q)
        q,g=ec_values(q,"base","adapt"); q.insert(0,"model",model); g.insert(0,"model",model)
        s=overall_stats(q); ci=bootstrap_groups(q,SEED+j)
        row={"model":model,**s}
        for k,(lo,hi) in ci.items(): row[f"{k}_ci95_low"]=lo; row[f"{k}_ci95_high"]=hi
        summaries.append(row); all_values.append(q); all_groups.append(g)
    values=pd.concat(all_values,ignore_index=True); values.to_csv(out/"within_group_corrections.csv",index=False)
    pd.concat(all_groups,ignore_index=True).to_csv(out/"within_group_group_metrics.csv",index=False)
    summary=pd.DataFrame(summaries); summary.to_csv(out/"within_group_summary.csv",index=False)
    assert summary.n_groups.eq(45).all() and summary.n_structures.eq(193).all()
    assert abs(summary.loc[summary.model.eq("sevennet"),"base_ec_mae_mev_atom"].iloc[0]-4.190)<0.01
    assert abs(summary.loc[summary.model.eq("sevennet"),"adapt_ec_mae_mev_atom"].iloc[0]-17.034)<0.02
    assert abs(summary.loc[summary.model.eq("eqv3"),"base_ec_mae_mev_atom"].iloc[0]-2.910)<0.01
    assert abs(summary.loc[summary.model.eq("eqv3"),"adapt_ec_mae_mev_atom"].iloc[0]-3.402)<0.02
    return summary


def independent_eval(out: Path, sets: dict[str,pd.DataFrame], pred: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]; conf=set(sets["confirmation"].id)
    for model in ("sevennet","eqv3"):
        exposed=set()
        for name in (f"{model}_model_train",f"{model}_model_val",f"{model}_baseline_calibration_fit",f"{model}_postadapt_calibration_fit","composition_fit","alpha_selection"):
            exposed |= set(sets[name].id)
        keep=sorted(conf-exposed)
        b=pred["seven_base" if model=="sevennet" else "eq_base"]
        a=pred["seven_adapt" if model=="sevennet" else "eq_adapt"]
        q=b[b.id.isin(keep)][["id","formula","natoms","y_true","pred_element_reference"]].rename(columns={"pred_element_reference":"base"}).merge(a[["id","adapt"]],on="id",validate="one_to_one")
        q["exact_composition"]=exact_key(q); sz=q.groupby("exact_composition").size(); ec=q[q.exact_composition.isin(sz[sz>=2].index)].copy()
        r={"model":model,"n_fully_unexposed_structures":len(q),"n_repeated_exact_composition_groups":int(ec.exact_composition.nunique()),"n_ec_structures":len(ec),"base_mae_mev_atom":float(np.mean(np.abs(q.base-q.y_true))*1000),"adapt_mae_mev_atom":float(np.mean(np.abs(q.adapt-q.y_true))*1000)}
        if len(ec):
            ec,_=ec_values(ec,"base","adapt"); r.update({"base_ec_mae_mev_atom":float(np.mean(np.abs(ec.T_ev_atom))*1000),"adapt_ec_mae_mev_atom":float(np.mean(np.abs(ec.C_ev_atom-ec.T_ev_atom))*1000)})
        rows.append(r); q.insert(0,"model",model); q.to_csv(out/f"intersections/{model}_fully_unexposed_confirmation_ids.csv",index=False)
    result=pd.DataFrame(rows); result.to_csv(out/"fully_independent_evaluation.csv",index=False); return result


def force_audit(out: Path) -> pd.DataFrame:
    rows=[]
    for model,path in (("sevennet",SEVEN_COEF),("eqv3",EQ_COEF),("eqv3_adapt",EQ_ADAPT_COEF)):
        c=read_csv(path); aff=c[(c.calibration=="affine_energy_composition")&(c.coefficient=="alpha_energy")]
        rows.append({"model":model,"check":"element_reference_force_scale","energy_definition":"e_cal=e_raw-sum(x*mu)-b per atom; E=N*e","expected_force_relation":"F_cal=F_raw","scale":1.0,"status":"passed_by_formula","n":np.nan,"absolute_error":0.0,"relative_error":0.0})
        if len(aff): rows.append({"model":model,"check":"affine_force_scale","energy_definition":"e_cal=a*e_raw+sum(x*beta)+b; E=N*e","expected_force_relation":"F_cal=a*F_raw","scale":float(aff.value.iloc[0]),"status":"passed_by_formula","n":np.nan,"absolute_error":0.0,"relative_error":0.0})
    fd=read_csv(FD_EXISTING)
    for r in fd.itertuples(index=False):
        rows.append({"model":"eqv3_adapt_raw","check":"existing_central_finite_difference","structure_id":r.structure_id,"epsilon_ang":r.epsilon_ang_rms,"energy_definition":"raw EqV3 reconstructed total energy; not post-calibrated energy","expected_force_relation":"dE/du=-F.u","scale":1.0,"status":"existing_evidence_single_step_only","n":1,"absolute_error":r.absolute_error,"relative_error":r.relative_error})
    # Preserve magnitude information; high cosine alone is not sufficient.
    eq=read_csv(FORCE_EQ)
    rows.append({"model":"eqv3","check":"existing_raw_force_distribution","energy_definition":"raw base calculator versus raw reconstructed residual-LoRA calculator","status":"existing_evidence","n":len(eq),"force_cosine_median":float(eq.force_cosine.median()),"force_cosine_p05":float(eq.force_cosine.quantile(.05)),"force_cosine_p95":float(eq.force_cosine.quantile(.95)),"force_difference_rmse_median_ev_ang":float(eq.force_difference_rmse.median()),"force_difference_rmse_p95_ev_ang":float(eq.force_difference_rmse.quantile(.95)),"base_force_rms_median_ev_ang":float(eq.base_force_rms.median()),"adapt_force_rms_median_ev_ang":float(eq.lora_force_rms.median()),"force_norm_ratio_median":float(eq.force_norm_ratio.median()),"force_norm_ratio_p05":float(eq.force_norm_ratio.quantile(.05)),"force_norm_ratio_p95":float(eq.force_norm_ratio.quantile(.95))})
    keys=["structure_id","family","axis","magnitude","replicate","atom_index"]
    fb=read_csv(FORCE_SEVEN); fa=read_csv(FORCE_SEVEN_ADAPT)
    z=fb.merge(fa,on=keys,suffixes=("_b","_a"),validate="one_to_one")
    z["dot"]=z.fx_b*z.fx_a+z.fy_b*z.fy_a+z.fz_b*z.fz_a
    z["sq_b"]=z.fx_b**2+z.fy_b**2+z.fz_b**2; z["sq_a"]=z.fx_a**2+z.fy_a**2+z.fz_a**2
    z["sq_d"]=(z.fx_a-z.fx_b)**2+(z.fy_a-z.fy_b)**2+(z.fz_a-z.fz_b)**2
    config=keys[:-1]
    s=z.groupby(config,as_index=False).agg(dot=("dot","sum"),sq_b=("sq_b","sum"),sq_a=("sq_a","sum"),sq_d=("sq_d","sum"),natoms=("atom_index","size"))
    s["cosine"]=s["dot"]/np.sqrt(s.sq_b*s.sq_a); s["norm_ratio"]=np.sqrt(s.sq_a/s.sq_b)
    s["base_rms"]=np.sqrt(s.sq_b/(3*s.natoms)); s["adapt_rms"]=np.sqrt(s.sq_a/(3*s.natoms)); s["diff_rms"]=np.sqrt(s.sq_d/(3*s.natoms))
    rows.append({"model":"sevennet","check":"existing_raw_force_distribution","energy_definition":"raw base calculator versus raw fine-tuned calculator","status":"existing_evidence","n":len(s),"force_cosine_median":float(s.cosine.median()),"force_cosine_p05":float(s.cosine.quantile(.05)),"force_cosine_p95":float(s.cosine.quantile(.95)),"force_difference_rmse_median_ev_ang":float(s.diff_rms.median()),"force_difference_rmse_p95_ev_ang":float(s.diff_rms.quantile(.95)),"base_force_rms_median_ev_ang":float(s.base_rms.median()),"adapt_force_rms_median_ev_ang":float(s.adapt_rms.median()),"force_norm_ratio_median":float(s.norm_ratio.median()),"force_norm_ratio_p05":float(s.norm_ratio.quantile(.05)),"force_norm_ratio_p95":float(s.norm_ratio.quantile(.95))})
    f=pd.DataFrame(rows); f.to_csv(out/"energy_force_checks.csv",index=False)
    return f


def matched_controls(out: Path) -> pd.DataFrame:
    d=read_csv(SEVEN_CONTROLS)
    base=d[d.checkpoint.eq("sevennet_base")].iloc[0]
    d=d.copy()
    d.insert(0,"comparison_class","same_model_different_adaptation_protocol")
    d["absolute_mae_delta_vs_sevennet_base_mev_atom"]=d.energy_mae_mev_atom.astype(float)-float(base.energy_mae_mev_atom)
    d["ec_mae_delta_vs_sevennet_base_mev_atom"]=d.exact_composition_mae_mev_atom.astype(float)-float(base.exact_composition_mae_mev_atom)
    d["paired_scope"]="same frozen 2020-row confirmation set; EC uses same 45 groups / 193 structures"
    d["usable_for_architecture_attribution"]=False
    d["reason"]="usable within SevenNet; no EqV3 run with the same output/loss/training contract"
    d.to_csv(out/"matched_controls.csv",index=False)
    return d


def write_contracts(out: Path, cal: pd.DataFrame) -> None:
    eq=json.loads(EQ_SUMMARY.read_text())
    text=f"""# Training contracts (code-traced)

## Experiment identity

The audited main results are the `experiment3_20260922` / `experiment6_20260922` / `experiment8_20260923` / `experiment9_20260924` chain on `medium_same_split_269` (8192/1024/2048), not later Experiment 14 or 17 controls.

## SevenNet main model

- Training source: `{SEVEN_TRAINER}`.
- Prediction: `e_raw(X)=E_model_total(X)/N_X`, in eV/atom.
- Target and loss: `SmoothL1(e_raw, y_form; beta=0.05 eV/atom)`.
- Trainable parameters: 301,300 / 842,623 (prefixes `3_`, `4_`, `reduce_input_to_hidden`, `reduce_hidden_to_energy`, `rescale_atomic_energy`).
- AdamW: LR 3e-4, weight decay 1e-5; batch 16; at most 20 epochs; patience 8; best checkpoint selected by official validation MAE.
- Post-training calibration: RidgeCV on official 8192 training rows predicts `y-e_raw` from composition/count features, then `e_adapt=e_raw+h_train-only(x,N)`.

## EqV3 main model

- Training source: `{EQ_TRAINER}`; checkpoint `{EQ_CKPT}`.
- Frozen reference is `e_ref=pred_element_reference` from the older EqV3 reference table.  For each structure, `E_ref=N*e_ref`.
- The EqV3 energy branch emits a scaled **residual total energy**, `r_scaled`.  Decoding is `E_pred=E_ref+s*r_scaled`, where `s=model.avg_num_nodes` (`scale_divisor={eq['scale_divisor']}`); `e_pred=E_pred/N`.
- At initialization every LoRA `B` matrix is zero, so the backbone contribution is unchanged, but the pretrained `energy_block` is merely unfrozen: it is **not reset or zeroed**. Therefore the residual branch initially emits the pretrained absolute EqV3 total-energy output. The decoded step-0 prediction is `E_ref + E_pretrained`, not `E_ref`. This treats an absolute pretrained output as a residual and is an implementation/output-definition mismatch for a clean residual route.
- Training target is `(N*y_form-E_ref)/s`.  Loss is SmoothL1 in that scaled residual-total-energy coordinate with beta 0.05 (not the same physical loss coordinate as SevenNet).
- Tuning: LoRA plus energy head, {int(eq['parameter_stats']['trainable_parameters']):,} / {int(eq['parameter_stats']['total_parameters']):,} parameters ({eq['parameter_stats']['trainable_fraction']*100:.3f}%); LoRA rank 8, alpha 16; AdamW head/LoRA LR 2e-4, other trainable backbone LR 5e-5, weight decay 1e-4; batch 8; 12 requested epochs; validation-MAE checkpoint, patience 6; bfloat16 AMP.
- A second train-only element-reference calibration is then fitted on official train rows; the paper's EqV3 adapted column is that calibrated output.

## Inference and correction formulas

For the baseline element-reference calibration, `e_base=e_model-sum_Z x_Z mu_Z-b`, so `E_base=N*e_base` and `F_base=F_model`.  For affine calibration, `e_aff=a*e_model+sum_Z x_Z beta_Z+b`, hence `F_aff=a F_model` at fixed composition.

For geometry-free downstream correction, `e_value=e_base+h_target(x,N)`; its force is exactly `F_base` at fixed composition.  For an interpolation `e_alpha=e_base+h_target+alpha*(e_adapt-e_base-h_target)`, total energy is `E_alpha=N*e_alpha` and `F_alpha=(1-alpha)F_base+alpha F_adapt`; `h_target` contributes zero force.

## Attribution verdict

The current SevenNet/EqV3 comparison mixes backbone, trainable parameterization, trainable fraction, learning rates, epochs, mixed precision, output coordinate, and loss coordinate. In addition, the audited EqV3 residual route starts from `E_ref + E_pretrained` because the absolute pretrained energy head is not converted to a zero residual. Matched same-model SevenNet adaptation controls do exist and are tabulated in `matched_controls.csv`; they support claims about protocol effects within SevenNet. No existing **same-protocol different-model** control matches split, evaluation set, physical output definition and loss coordinate. Therefore the cross-model evidence compares **model + adaptation implementation**, not architecture alone.
"""
    (out/"training_contracts.md").write_text(text)


def write_report(out: Path, overlap: pd.DataFrame, cal: pd.DataFrame, indep: pd.DataFrame, wg: pd.DataFrame, force: pd.DataFrame, controls: pd.DataFrame, sets: dict[str,pd.DataFrame]) -> None:
    def ov(a,b,key="id"):
        q=overlap[((overlap.set_a==min(a,b))&(overlap.set_b==max(a,b))|(overlap.set_a==a)&(overlap.set_b==b))&overlap.key_type.eq(key)]
        if q.empty:
            q=overlap[((overlap.set_a==b)&(overlap.set_b==a))&overlap.key_type.eq(key)]
        return int(q.n_intersection.iloc[0])
    # Direct set arithmetic avoids dependence on lexicographic orientation in display table.
    conf=set(sets["confirmation"].id)
    seven_train=len(conf & set(sets["sevennet_model_train"].id)); eq_train=len(conf & set(sets["eqv3_model_train"].id))
    selected=controls[controls.checkpoint.isin(["sevennet_base","last_two_blocks_lr0p0003","l2sp","replay_1","recal_alpha_0","recal_alpha_0.05"])][["checkpoint","method","energy_mae_mev_atom","exact_composition_mae_mev_atom","absolute_mae_delta_vs_sevennet_base_mev_atom","ec_mae_delta_vs_sevennet_base_mev_atom"]]
    force_stats=force[force.check.eq("existing_raw_force_distribution")][["model","n","force_cosine_median","force_difference_rmse_median_ev_ang","base_force_rms_median_ev_ang","adapt_force_rms_median_ev_ang","force_norm_ratio_median","force_norm_ratio_p05","force_norm_ratio_p95"]]
    lines=["# Audit report: energy-only fine-tuning and correction allocation","","## Scope and experiment lock","","Audited only the fixed Experiment 3/6/8/9 chain. No training was launched, no original file was modified, and no paper result was rewritten. Input hashes and code versions are in `input_manifest.csv` and `environment.json`.","","## 1. Confirmation-set independence — **manuscript wording error**","",f"The 2,020-row downstream confirmation set is disjoint from downstream transfer fitting and alpha selection by structure ID, but it is **not independent of the earlier official model split**. Confirmation contains **{seven_train} SevenNet official-train IDs** and **{eq_train} EqV3 official-train IDs**. Those same rows entered the actual baseline element-reference fits and the post-adaptation train-only calibration whenever predictions were available.","",f"Thus the statement that 'confirmation labels do not participate in calibration' is false if 'calibration' includes the baseline/post-adaptation official-train fits. It is true only for the later downstream composition fit. The set must not be described as a completely independent test set.","",f"The actually unexposed remainder (excluding official train, official validation/model selection, baseline fit, post-adaptation fit, transfer fitting and alpha selection) is summarized below:","",indep.to_markdown(index=False),"",f"Baseline refits reproduce stored predictions: {bool(cal.reproduced.all())}. This traces the fit to actual official-train rows, not names.","","Detailed intersections by ID, reduced formula and chemical system are in `split_overlap.csv` and `intersections/intersection_ids.csv`.","","## 2. Model-vs-protocol attribution — **implementation error plus insufficient architecture-only evidence**","","The exact executable contracts are in `training_contracts.md`. SevenNet predicts direct per-atom formation energy with a per-atom SmoothL1 loss. EqV3 labels its route residual, but LoRA starts at zero while the pretrained absolute `energy_block` is left intact; the step-0 decoded prediction is therefore `E_ref + E_pretrained`, not `E_ref`. Treating the absolute pretrained branch as a residual is an **implementation/output-definition error** for a clean residual route. The saved 9.63 meV/atom result remains a valid measurement of the implemented route, but not of a correctly initialized residual route.","","They also differ in trainable fraction, optimizer groups, epoch budget and precision. Matched within-SevenNet protocol contrasts are valid and summarized below; the full 21-row table is `matched_controls.csv`.","",selected.to_markdown(index=False),"","There is no existing same-protocol different-model control matching physical output and loss definitions. The supported cross-model comparison is 'model + adaptation implementation', not architecture alone.","","## 3. Energy/calibration/force consistency — **formula audit passed; requested new numerical cross-check incomplete**","","All reported energies are per-atom eV in prediction tables; model forces must come from total energy `E=N e`. Element-reference and composition-only terms are position-independent at fixed composition. Therefore element-reference calibration leaves force unchanged, affine calibration scales force by its energy coefficient, and alpha interpolation gives `(1-alpha)F_base+alpha F_adapt`.","","Existing raw-force magnitude statistics (not calibrated-gradient claims):","",force_stats.to_markdown(index=False),"",f"The old force probes used raw model calculator forces. They were not recomputed as gradients of every published calibrated/interpolated energy. The existing EqV3 finite-difference table has {len(force[force.check.eq('existing_central_finite_difference')])} structures at only one step (`1e-3 Å`); median relative error is {force.loc[force.check.eq('existing_central_finite_difference'),'relative_error'].median():.3g}. This does not meet the requested multi-step convergence check.","","The NVIDIA driver is unavailable under the current kernel, so new real forward/autograd/multi-step finite differences were not fabricated. The missing checks remain explicitly unresolved. Local-perturbation provenance is: 500 stratified confirmation parents; EqV3 energies on 500 parents × 33 configurations = 16,500; forces on the first 75 parents × 7 retained configurations = 525 in the analyzed force table; random-displacement RMS values and exact generator are preserved in the cited probe script. The old gradient check used the first 100 parents, one deterministic zero-COM random direction per structure, and `eps=1e-3 Å`.","","## 4. EqV3 'small error increase' versus actual change — **claim not supported by EC-MAE delta alone**","",wg.to_markdown(index=False),"","`C_rms`, `C/T`, correlations and `alpha_EC*` directly quantify prediction change and alignment with the needed correction. EC-MAE delta instead quantifies only the change in error. They are not interchangeable. Ranking, lowest-energy selection, per-group values and complete-group bootstrap intervals are included in `within_group_summary.csv`, `within_group_group_metrics.csv`, and `within_group_corrections.csv`. Force cosine likewise does not replace force RMS/norm-ratio statistics; the existing force evidence is preserved without upgrading it to a calibrated-gradient claim.","","## Classification summary","","- Confirmation described as fully label-independent: **manuscript wording error**.","- EqV3 absolute pretrained output used as a residual at initialization: **implementation error**.","- Architecture-only attribution from SevenNet versus EqV3: **existing evidence insufficient**.","- Algebraic energy/calibration/force definitions: **passed**.","- Published raw-force metrics treated as gradients of every calibrated/interpolated energy: **existing evidence insufficient / wording must be qualified**.","- New multi-step finite-difference/autograd check: **unresolved due NVIDIA driver**, not passed.","- EqV3 actual-change inference from a small EC-MAE increase: **existing evidence insufficient**; use the new C/T/alignment statistics.","","## Reproduction","","```bash",f"/home/yxzhang/envs/fegx_benchmark214/bin/python {out/'code/audit_energy_allocation.py'} --out {out}",f"/home/yxzhang/envs/fegx_benchmark214/bin/python {out/'code/verify_audit.py'} {out}","```",""]
    (out/"AUDIT_REPORT.md").write_text("\n".join(lines))


def manifest(out: Path) -> None:
    rows=[]
    for p in INPUTS+[SEVEN_CKPT,EQ_CKPT,out/"code/audit_energy_allocation.py",out/"code/verify_audit.py"]:
        rows.append({"path":str(p),"exists":p.exists(),"size_bytes":p.stat().st_size if p.exists() else np.nan,"sha256":sha256(p) if p.exists() else "MISSING"})
    pd.DataFrame(rows).to_csv(out/"input_manifest.csv",index=False)
    def run(cmd):
        try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
        except Exception as e:return f"unavailable: {e}"
    env={"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"git_energy":run(["git","-C",str(ROOT),"rev-parse","HEAD"]),"git_eqv3":run(["git","-C",str(ROOT/'external_repos/equiformer_v3-main'),"rev-parse","HEAD"]),"nvidia_smi":run(["nvidia-smi"])}
    (out/"environment.json").write_text(json.dumps(env,indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args(); out=args.out.resolve()
    if out == ROOT or ROOT not in out.parents: raise ValueError("--out must be a new child directory of /home/yxzhang/energy")
    out.mkdir(parents=True,exist_ok=True); (out/"intersections").mkdir(exist_ok=True); (out/"logs").mkdir(exist_ok=True); (out/"code").mkdir(exist_ok=True)
    dataset=load_dataset(); pred=load_predictions(dataset); sets,_=cohorts(dataset,pred)
    overlap=overlap_audit(out,sets); cal=calibration_reproduction(out,pred); indep=independent_eval(out,sets,pred); wg=within_group(out,sets,pred); force=force_audit(out); controls=matched_controls(out)
    write_contracts(out,cal); manifest(out); write_report(out,overlap,cal,indep,wg,force,controls,sets)
    print(json.dumps({"status":"complete","output":str(out),"confirmation":len(sets['confirmation']),"exact_groups":int(sets['exact193'].exact_composition.nunique()),"exact_structures":len(sets['exact193'])},indent=2))


if __name__ == "__main__": main()
