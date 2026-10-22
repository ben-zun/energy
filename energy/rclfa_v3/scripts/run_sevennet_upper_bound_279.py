#!/usr/bin/env python3
"""Run the 279B SevenNet capability upper-bound ladder.

This is the fallback after the formal MACE M5/M6 path was blocked.  The goal is
method transferability and the upper bound of SevenNet on the 269 medium subset,
not replacement of EqV3/MatRIS.

Ladder used here:
M1 scalar train-only calibrated reference
M2 scalar reference + real FE-GX residual
M3 native SevenNet embedding frozen adapter
M4 native embedding adapter reference + real FE-GX residual
M5 SevenNet head-only formation-energy fine-tune
M6 SevenNet partial formation-energy fine-tune
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/home/yxzhang/energy")
OUT = ROOT / "runs/rclfa_v3/sevennet_upper_bound_279"
REPORT = ROOT / "codex_reports/279B_sevennet_upper_bound.md"
PET_PYTHON = Path("/home/yxzhang/envs/pet_forme/bin/python3.11")
FOUNDATION_SITE = Path("/home/yxzhang/envs/foundation_teacher/lib/python3.11/site-packages")
SUBSET_DIR = ROOT / "runs/rclfa_v3/full_fegx_like_foundation_probe_269/medium_subset_probe"
SUBSET_EXTXYZ = SUBSET_DIR / "medium_subset_probe.extxyz"
SUBSET_MANIFEST = SUBSET_DIR / "medium_subset_probe_manifest.csv"
FOLD0_CACHE = ROOT / "runs/rclfa_v3/official_benchmark_214/matbench_e_form_5fold/fold0/cache"
REF272 = ROOT / "runs/rclfa_v3/real_fegx_foundation_replacement_272"
SEVEN_REF272 = REF272 / "reference_conversion/sevennet_reference"
SEVEN_REF272_SUMMARY = REF272 / "reference_conversion/sevennet_reference_summary.json"
SEVEN_SINGLE272 = REF272 / "single_foundation/real_fegx_single_foundation_metrics.csv"
TRAIN_FEGX = ROOT / "rclfa_v3/scripts/train_fegx1.py"
SEED = 20260528

sys.path.insert(0, str(ROOT))
if FOUNDATION_SITE.exists():
    site.addsitedir(str(FOUNDATION_SITE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 279B SevenNet upper-bound ladder.")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--m5-lr", type=float, default=1e-3)
    parser.add_argument("--m6-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--run-m3", action="store_true")
    parser.add_argument("--run-m4", action="store_true")
    parser.add_argument("--run-m5", action="store_true")
    parser.add_argument("--run-m6", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0}
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae_mev": float(np.mean(np.abs(err)) * 1000.0),
        "rmse_mev": float(np.sqrt(np.mean(err**2)) * 1000.0),
        "p95_mev": float(np.percentile(np.abs(err), 95) * 1000.0),
        "bias_mev": float(np.mean(err) * 1000.0),
    }


def runtime_status(args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] = {
        "goal": "SevenNet capability upper bound after MACE formal medium fine-tune was blocked",
        "test_label_policy": "test labels are evaluation-only; no training, calibration, early stopping, or model selection",
        "foundation_site": str(FOUNDATION_SITE),
        "foundation_site_exists": FOUNDATION_SITE.exists(),
        "requested_device": args.device,
        "sevennet_model": "7net-0",
    }
    try:
        import torch
        import torch_geometric
        import sevenn
        import sevenn.util as util

        checkpoint = util.pretrained_name_to_path("7net-0")
        status.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_geometric": torch_geometric.__version__,
                "sevenn": getattr(sevenn, "__version__", ""),
                "sevenn_file": getattr(sevenn, "__file__", ""),
                "sevennet_checkpoint": checkpoint,
                "sevennet_checkpoint_exists": Path(checkpoint).exists(),
            }
        )
    except Exception as exc:
        status["probe_error"] = type(exc).__name__ + ": " + str(exc)
    return status


def split_limits(args: argparse.Namespace) -> dict[str, int | None]:
    return {
        "train": args.max_train_rows,
        "val": args.max_val_rows,
        "test": args.max_test_rows,
    }


def graph_cache_dir(args: argparse.Namespace) -> Path:
    suffix = "medium"
    limits = split_limits(args)
    if any(v is not None for v in limits.values()):
        suffix = f"limited_train{limits['train']}_val{limits['val']}_test{limits['test']}"
    return args.out_dir / "graph_cache" / suffix


def load_sevennet_model(*, remove_force_output: bool = True):
    import sevenn.util as util

    model, _ = util.model_from_checkpoint("7net-0")
    if remove_force_output:
        model.delete_module_by_key("force_output")
        model.key_grad = None
    model.set_is_batch_data(True)
    return model


def prepare_graph_cache(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from ase.io import iread
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.dataload import unlabeled_atoms_to_graph

    cache = graph_cache_dir(args)
    summary_path = cache / "graph_cache_summary.json"
    graph_paths = {split: cache / f"{split}_graphs.pt" for split in ("train", "val", "test")}
    meta_path = cache / "graph_metadata.csv"
    if summary_path.exists() and all(path.exists() for path in graph_paths.values()) and meta_path.exists() and not args.force:
        return read_json(summary_path)

    set_seed(args.seed)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SUBSET_MANIFEST)
    model = load_sevennet_model(remove_force_output=True)
    cutoff = float(model.cutoff)
    limits = split_limits(args)
    counts = {"train": 0, "val": 0, "test": 0}
    graphs: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    meta_rows: list[dict[str, Any]] = []
    start = time.time()
    for idx, atoms in enumerate(iread(SUBSET_EXTXYZ, index=":")):
        row = manifest.iloc[idx]
        split = str(row["split"])
        if split not in graphs:
            continue
        limit = limits[split]
        if limit is not None and counts[split] >= int(limit):
            continue
        natoms = int(row.get("natoms", row.get("n_atoms", len(atoms))))
        y_true = float(row["y_true"])
        graph_dict = unlabeled_atoms_to_graph(atoms, cutoff=cutoff, with_shift=False)
        graph_dict["target_per_atom"] = np.array(y_true, dtype=np.float32)
        graph_dict["target_total_energy"] = np.array(y_true * natoms, dtype=np.float32)
        # Avoid names containing "index": PyG treats those as graph indices and
        # offsets them during batching.
        graph_dict["manifest_row"] = np.array(idx, dtype=np.int64)
        graph = AtomGraphData.from_numpy_dict(graph_dict)
        graphs[split].append(graph)
        counts[split] += 1
        meta_rows.append(
            {
                "row_index": idx,
                "id": str(row["id"]),
                "split": split,
                "formula": str(row["formula"]),
                "cif_path": str(row.get("cif_path", "")),
                "natoms": natoms,
                "nelements": int(row.get("nelements", 0)) if "nelements" in row else "",
                "y_true": y_true,
            }
        )
    for split, path in graph_paths.items():
        torch.save(graphs[split], path)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    summary = {
        "status": "complete",
        "cache_dir": str(cache),
        "cutoff": cutoff,
        "counts": counts,
        "limits": limits,
        "metadata_csv": str(meta_path),
        "graph_paths": {split: str(path) for split, path in graph_paths.items()},
        "elapsed_seconds": time.time() - start,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def load_graphs(args: argparse.Namespace) -> tuple[dict[str, list[Any]], pd.DataFrame]:
    import torch

    summary = prepare_graph_cache(args)
    cache = Path(str(summary["cache_dir"]))
    graphs = {
        split: torch.load(cache / f"{split}_graphs.pt", map_location="cpu", weights_only=False)
        for split in ("train", "val", "test")
    }
    meta = pd.read_csv(cache / "graph_metadata.csv")
    return graphs, meta


def forward_until(model: Any, batch: Any, *, stop_after: str):
    import sevenn._keys as KEY

    data = model._preprocess(batch)
    for name, module in model._modules.items():
        if name == "force_output":
            continue
        data = module(data)
        if name == stop_after:
            break
    return data[KEY.NODE_FEATURE]


def pool_node_features(x: Any, batch_index: Any, size: int):
    import torch
    from torch_geometric.nn import global_max_pool, global_mean_pool

    mean = global_mean_pool(x, batch_index, size=size)
    sq_mean = global_mean_pool(x * x, batch_index, size=size)
    std = torch.clamp(sq_mean - mean * mean, min=0.0).sqrt()
    maxv = global_max_pool(x, batch_index, size=size)
    minv = -global_max_pool(-x, batch_index, size=size)
    return torch.cat([mean, std, maxv, minv], dim=1)


def extract_embedding_features(args: argparse.Namespace, graphs: dict[str, list[Any]], meta: pd.DataFrame) -> pd.DataFrame:
    import torch
    from torch_geometric.loader import DataLoader

    out_dir = args.out_dir / "native_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / "sevennet_native_embedding_features.csv"
    if features_csv.exists() and not args.force:
        return pd.read_csv(features_csv)

    model = load_sevennet_model(remove_force_output=True).to(args.device)
    model.eval()
    rows: list[dict[str, Any]] = []
    meta_by_row = meta.set_index("row_index").to_dict("index")
    start = time.time()
    for split in ("train", "val", "test"):
        loader = DataLoader(graphs[split], batch_size=args.eval_batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)
                node_x = forward_until(model, batch, stop_after="reduce_input_to_hidden")
                pooled = pool_node_features(node_x, batch.batch, size=int(batch.num_graphs)).detach().cpu().numpy()
                row_indices = batch.manifest_row.detach().cpu().numpy().astype(int)
                for i, row_index in enumerate(row_indices):
                    meta_row = meta_by_row[int(row_index)]
                    record = {
                        "row_index": int(row_index),
                        "id": meta_row["id"],
                        "split": split,
                        "formula": meta_row["formula"],
                        "cif_path": meta_row.get("cif_path", ""),
                        "natoms": int(meta_row["natoms"]),
                        "nelements": meta_row.get("nelements", ""),
                        "y_true": float(meta_row["y_true"]),
                    }
                    for j, value in enumerate(pooled[i]):
                        record[f"sevennet_emb_pool_{j:04d}"] = float(value)
                    rows.append(record)
    frame = pd.DataFrame(rows).sort_values("row_index")
    frame.to_csv(features_csv, index=False)
    summary = {
        "status": "complete",
        "features_csv": str(features_csv),
        "n_rows": int(len(frame)),
        "n_features": len([c for c in frame.columns if c.startswith("sevennet_emb_pool_")]),
        "stop_after": "reduce_input_to_hidden",
        "elapsed_seconds": time.time() - start,
    }
    (out_dir / "sevennet_native_embedding_features_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return frame


def run_m3_embedding_adapter(args: argparse.Namespace, graphs: dict[str, list[Any]], meta: pd.DataFrame) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out_dir = args.out_dir / "native_embedding"
    summary_path = out_dir / "sevennet_native_embedding_frozen_adapter_summary.json"
    metrics_path = out_dir / "sevennet_native_embedding_frozen_adapter_metrics.csv"
    predictions_path = out_dir / "sevennet_native_embedding_frozen_adapter_predictions.csv"
    if summary_path.exists() and metrics_path.exists() and predictions_path.exists() and not args.force:
        return read_json(summary_path)

    features = extract_embedding_features(args, graphs, meta)
    composition = pd.read_csv(FOLD0_CACHE / "composition_features.csv")
    comp_cols = [c for c in composition.columns if c.startswith("x_Z")]
    features = features.merge(composition[["id", *comp_cols]], on="id", how="left")
    features[comp_cols] = features[comp_cols].fillna(0.0)
    emb_cols = [c for c in features.columns if c.startswith("sevennet_emb_pool_")]
    split = features["split"].astype(str)
    masks = {
        "train": split.eq("train").to_numpy(),
        "val": split.eq("val").to_numpy(),
        "test": split.eq("test").to_numpy(),
    }
    y = features["y_true"].to_numpy(float)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    feature_sets = {
        "sevennet_embedding_pooled": emb_cols,
        "sevennet_embedding_pooled_plus_composition": [*emb_cols, *comp_cols],
    }
    metrics_rows: list[dict[str, Any]] = []
    predictions_by_variant: dict[str, np.ndarray] = {}
    for family, cols in feature_sets.items():
        x = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        for alpha in alphas:
            variant = f"{family}_ridge_a{alpha:g}"
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[masks["train"]], y[masks["train"]])
            pred = model.predict(x)
            predictions_by_variant[variant] = pred
            for split_name, mask in masks.items():
                row = regression_metrics(y[mask], pred[mask])
                row.update(
                    {
                        "level": "M3",
                        "variant": variant,
                        "feature_family": family,
                        "alpha": alpha,
                        "split": split_name,
                        "selection_policy": "fit train only; choose alpha/family on val; evaluate test once",
                    }
                )
                metrics_rows.append(row)
    val_rows = [row for row in metrics_rows if row["split"] == "val"]
    best_val = min(val_rows, key=lambda row: float(row["mae_mev"]))
    best_variant = str(best_val["variant"])
    test_row = next(row for row in metrics_rows if row["split"] == "test" and row["variant"] == best_variant)
    write_csv(
        metrics_path,
        metrics_rows,
        ["level", "variant", "feature_family", "alpha", "split", "n", "mae_mev", "rmse_mev", "p95_mev", "bias_mev", "selection_policy"],
    )
    predictions = features[["row_index", "id", "split", "formula", "cif_path", "natoms", "nelements", "y_true"]].copy()
    predictions["pred_sevennet_native_embedding_adapter"] = predictions_by_variant[best_variant]
    predictions.to_csv(predictions_path, index=False)
    summary = {
        "status": "complete",
        "level": "M3",
        "features_csv": str(out_dir / "sevennet_native_embedding_features.csv"),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "selection_split": "val",
        "best_val_variant": best_variant,
        "best_val_mae_mev": best_val["mae_mev"],
        "best_test_n": test_row["n"],
        "best_test_mae_mev": test_row["mae_mev"],
        "best_test_rmse_mev": test_row["rmse_mev"],
        "best_test_p95_mev": test_row["p95_mev"],
        "no_test_label_for_selection": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def convert_predictions_to_reference(
    *,
    pred_csv: Path,
    pred_col: str,
    ref_dir: Path,
    tag: str,
) -> dict[str, Any]:
    from rclfa_v3.data import build_feature_frame
    from rclfa_v3.reference import fit_calibrators, regression_metrics as ref_regression_metrics

    ref_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(pred_csv)
    frame = frame.rename(columns={pred_col: "model_energy_per_atom"}).copy()
    manifest_cols = ["id", "cif_path"]
    manifest = pd.read_csv(SUBSET_MANIFEST, usecols=manifest_cols)
    if "cif_path" in frame.columns:
        frame = frame.drop(columns=["cif_path"])
    frame = frame.merge(manifest, on="id", how="left")
    frame = build_feature_frame(frame, cache_path=FOLD0_CACHE / "composition_features.csv")
    frame["pred_raw_model_energy"] = frame["model_energy_per_atom"].astype(float)
    calibrators = fit_calibrators(frame, energy_column="model_energy_per_atom", target_column="y_true", ridge=1e-6)
    coeff_rows: list[dict[str, Any]] = []
    for result in calibrators:
        frame[f"pred_{result.name}"] = result.all_predictions
        for key, value in result.coefficients.items():
            coeff_rows.append({"calibration": result.name, "coefficient": key, "value": float(value)})
    output_cols = [
        "split",
        "id",
        "formula",
        "cif_path",
        "natoms",
        "nelements",
        "y_true",
        "model_energy_per_atom",
        "pred_raw_model_energy",
        "pred_element_reference",
        "pred_affine_energy_composition",
        "pred_composition_only_linear",
    ]
    frame[output_cols].to_csv(ref_dir / "calibrated_predictions.csv", index=False)
    pd.DataFrame(coeff_rows).to_csv(ref_dir / "calibration_coefficients.csv", index=False)
    metrics: dict[str, Any] = {
        "tag": tag,
        "reference_fit_scope": "train split only",
        "test_label_policy": "test labels are evaluation-only",
        "source_predictions": str(pred_csv),
        "calibrated_predictions": str(ref_dir / "calibrated_predictions.csv"),
        "calibration_coefficients": str(ref_dir / "calibration_coefficients.csv"),
        "metrics": {},
    }
    for calibration, column in (
        ("raw_model_energy", "pred_raw_model_energy"),
        ("element_reference", "pred_element_reference"),
        ("affine_energy_composition", "pred_affine_energy_composition"),
        ("composition_only_linear", "pred_composition_only_linear"),
    ):
        metrics["metrics"][calibration] = {}
        for split in ("train", "val", "test"):
            group = frame[frame["split"].astype(str) == split]
            metrics["metrics"][calibration][split] = ref_regression_metrics(
                group["y_true"].to_numpy(float),
                group[column].to_numpy(float),
            )
    (ref_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics


def train_fegx_from_reference(args: argparse.Namespace, *, ref_dir: Path, out_dir: Path, exp_name: str) -> dict[str, Any]:
    if (out_dir / "metrics.json").exists() and (out_dir / f"{exp_name}_predictions.csv").exists() and not args.force:
        return {"status": "complete_existing", "metrics": str(out_dir / "metrics.json"), "predictions": str(out_dir / f"{exp_name}_predictions.csv")}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "logs" / f"{exp_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PET_PYTHON),
        str(TRAIN_FEGX),
        "--calibrated-predictions",
        str(ref_dir / "calibrated_predictions.csv"),
        "--feature-cache",
        str(FOLD0_CACHE / "composition_features.csv"),
        "--structure-cache",
        str(FOLD0_CACHE / "structure_metadata.csv"),
        "--coefficient-csv",
        str(ref_dir / "calibration_coefficients.csv"),
        "--use-element-env",
        "--element-env-cache",
        str(FOLD0_CACHE / "element_env.csv"),
        "--use-pair-env",
        "--pair-env-cache",
        str(FOLD0_CACHE / "pair_env.csv"),
        "--residual-head-type",
        "moe",
        "--num-experts",
        "3",
        "--moe-load-balance-weight",
        "0.001",
        "--moe-entropy-weight",
        "0.0005",
        "--moe-router-init",
        "expert0",
        "--epochs",
        "200",
        "--patience",
        "30",
        "--batch-size",
        "2048",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
        "--exp-name",
        exp_name,
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return {"status": "complete", "metrics": str(out_dir / "metrics.json"), "predictions": str(out_dir / f"{exp_name}_predictions.csv"), "log": str(log_path)}


def run_m4_embedding_plus_fegx(args: argparse.Namespace, m3_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if m3_summary is None:
        m3_summary = read_json(args.out_dir / "native_embedding/sevennet_native_embedding_frozen_adapter_summary.json")
    pred_csv = Path(str(m3_summary.get("predictions_csv", "")))
    if not pred_csv.exists():
        return {"status": "blocked_missing_m3_predictions", "predictions_csv": str(pred_csv)}
    ref_dir = args.out_dir / "m4_native_embedding_reference"
    convert_predictions_to_reference(
        pred_csv=pred_csv,
        pred_col="pred_sevennet_native_embedding_adapter",
        ref_dir=ref_dir,
        tag="M4 SevenNet native embedding frozen adapter reference",
    )
    exp = f"fegx_279B_sevennet_native_embedding_reference_seed{args.seed}"
    result = train_fegx_from_reference(args, ref_dir=ref_dir, out_dir=args.out_dir / "m4_native_embedding_plus_fegx", exp_name=exp)
    return {"status": result["status"], "reference_dir": str(ref_dir), **result}


def set_trainable_parameters(model: Any, level: str) -> dict[str, Any]:
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    if level == "M5":
        prefixes = ("reduce_input_to_hidden", "reduce_hidden_to_energy", "rescale_atomic_energy")
    elif level == "M6":
        prefixes = ("4_", "reduce_input_to_hidden", "reduce_hidden_to_energy", "rescale_atomic_energy")
    else:
        raise ValueError(level)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad_(True)
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return {
        "level": level,
        "trainable_prefixes": list(prefixes),
        "n_trainable_tensors": len(trainable_names),
        "n_trainable_parameters": int(sum(model.state_dict()[name].numel() for name in trainable_names if name in model.state_dict())),
        "trainable_names": trainable_names,
        "n_frozen_tensors": len(frozen_names),
    }


def predict_batch(model: Any, batch: Any) -> Any:
    import sevenn._keys as KEY

    output = model(batch)
    pred_total = output[KEY.PRED_TOTAL_ENERGY].reshape(-1)
    natoms = output[KEY.NUM_ATOMS].reshape(-1).to(pred_total.dtype)
    return pred_total / natoms


def eval_model(args: argparse.Namespace, model: Any, graphs: dict[str, list[Any]], meta: pd.DataFrame, *, level: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    import torch
    from torch_geometric.loader import DataLoader

    model.eval()
    meta_by_row = meta.set_index("row_index").to_dict("index")
    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for split in ("train", "val", "test"):
            loader = DataLoader(graphs[split], batch_size=args.eval_batch_size, shuffle=False)
            y_values: list[float] = []
            pred_values: list[float] = []
            for batch in loader:
                batch = batch.to(args.device)
                pred = predict_batch(model, batch).detach().cpu().numpy().astype(float)
                y = batch.target_per_atom.reshape(-1).detach().cpu().numpy().astype(float)
                row_indices = batch.manifest_row.detach().cpu().numpy().astype(int)
                y_values.extend(y.tolist())
                pred_values.extend(pred.tolist())
                for i, row_index in enumerate(row_indices):
                    meta_row = meta_by_row[int(row_index)]
                    pred_rows.append(
                        {
                            "level": level,
                            "row_index": int(row_index),
                            "id": meta_row["id"],
                            "split": split,
                            "formula": meta_row["formula"],
                            "cif_path": meta_row.get("cif_path", ""),
                            "natoms": int(meta_row["natoms"]),
                            "nelements": meta_row.get("nelements", ""),
                            "y_true": float(y[i]),
                            "pred_sevennet_finetuned": float(pred[i]),
                        }
                    )
            metrics = regression_metrics(np.asarray(y_values), np.asarray(pred_values))
            metrics.update({"level": level, "split": split})
            metric_rows.append(metrics)
    return metric_rows, pd.DataFrame(pred_rows).sort_values("row_index")


def train_finetune_level(args: argparse.Namespace, level: str, graphs: dict[str, list[Any]], meta: pd.DataFrame) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    out_dir = args.out_dir / level.lower()
    summary_path = out_dir / f"sevennet_{level.lower()}_summary.json"
    metrics_path = out_dir / f"sevennet_{level.lower()}_metrics.csv"
    predictions_path = out_dir / f"sevennet_{level.lower()}_predictions.csv"
    if summary_path.exists() and metrics_path.exists() and predictions_path.exists() and not args.force:
        return read_json(summary_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = load_sevennet_model(remove_force_output=True).to(args.device)
    trainable = set_trainable_parameters(model, level)
    if trainable["n_trainable_tensors"] == 0:
        raise RuntimeError(f"No trainable tensors for {level}")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.m5_lr if level == "M5" else args.m6_lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=args.huber_beta)
    train_loader = DataLoader(graphs["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(graphs["val"], batch_size=args.eval_batch_size, shuffle=False)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = predict_batch(model, batch)
            y = batch.target_per_atom.reshape(-1).to(pred.dtype)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        val_y: list[float] = []
        val_pred: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                pred = predict_batch(model, batch).detach().cpu().numpy().astype(float)
                y = batch.target_per_atom.reshape(-1).detach().cpu().numpy().astype(float)
                val_pred.extend(pred.tolist())
                val_y.extend(y.tolist())
        val_metric = regression_metrics(np.asarray(val_y), np.asarray(val_pred))
        val_mae = float(val_metric.get("mae_mev", math.inf))
        history_rows.append(
            {
                "epoch": epoch,
                "train_smooth_l1_ev": float(np.mean(losses)) if losses else "",
                "val_mae_mev": val_mae,
                "val_rmse_mev": val_metric.get("rmse_mev", ""),
                "val_p95_mev": val_metric.get("p95_mev", ""),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    checkpoint_path = out_dir / f"sevennet_{level.lower()}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "level": level,
            "best_epoch": best_epoch,
            "best_val_mae_mev": best_val,
            "trainable": trainable,
            "args": {
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "huber_beta": args.huber_beta,
                "lr": args.m5_lr if level == "M5" else args.m6_lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        },
        checkpoint_path,
    )
    write_csv(
        out_dir / f"sevennet_{level.lower()}_history.csv",
        history_rows,
        ["epoch", "train_smooth_l1_ev", "val_mae_mev", "val_rmse_mev", "val_p95_mev", "lr"],
    )
    metric_rows, predictions = eval_model(args, model, graphs, meta, level=level)
    write_csv(
        metrics_path,
        metric_rows,
        ["level", "split", "n", "mae_mev", "rmse_mev", "p95_mev", "bias_mev"],
    )
    predictions.to_csv(predictions_path, index=False)
    test_row = next(row for row in metric_rows if row["split"] == "test")
    val_row = next(row for row in metric_rows if row["split"] == "val")
    summary = {
        "status": "complete",
        "level": level,
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "history_csv": str(out_dir / f"sevennet_{level.lower()}_history.csv"),
        "selection_split": "val",
        "best_epoch": best_epoch,
        "best_val_mae_mev": best_val,
        "final_val_mae_mev": val_row.get("mae_mev", ""),
        "test_mae_mev": test_row.get("mae_mev", ""),
        "test_rmse_mev": test_row.get("rmse_mev", ""),
        "test_p95_mev": test_row.get("p95_mev", ""),
        "trainable": trainable,
        "elapsed_seconds": time.time() - start,
        "no_test_label_for_selection": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def metric_from_json(metrics_json: Path, exp_name: str) -> dict[str, Any]:
    if not metrics_json.exists():
        return {}
    metrics = read_json(metrics_json)
    test = metrics.get(exp_name, {}).get("test", {}).get("overall", {})
    val = metrics.get(exp_name, {}).get("val", {}).get("overall", {})
    return {
        "val_mae_mev": val.get("mae_mev_atom", ""),
        "n": test.get("n", ""),
        "mae_mev": test.get("mae_mev_atom", ""),
        "rmse_mev": test.get("rmse_mev_atom", ""),
        "p95_mev": test.get("p95_abs_mev_atom", ""),
    }


def load_m2_row() -> dict[str, Any]:
    if not SEVEN_SINGLE272.exists():
        return {}
    frame = pd.read_csv(SEVEN_SINGLE272)
    row = frame[frame["foundation_key"].astype(str).eq("sevennet")]
    if row.empty:
        return {}
    item = row.iloc[0].to_dict()
    return {
        "n": item.get("coverage", ""),
        "mae_mev": item.get("mae_mev", ""),
        "rmse_mev": item.get("rmse_mev", ""),
        "p95_mev": item.get("p95_mev", ""),
        "source": str(SEVEN_SINGLE272),
    }


def append_ladder_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "level": "",
        "experiment": "",
        "status": "",
        "split": "test",
        "n": "",
        "mae_mev": "",
        "rmse_mev": "",
        "p95_mev": "",
        "source": "",
        "selection_policy": "",
        "notes": "",
        "claim_strength": "",
    }
    base.update(kwargs)
    rows.append(base)


def build_ladder(args: argparse.Namespace, runtime: dict[str, Any], summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref_metrics = read_json(SEVEN_REF272 / "metrics.json")
    ref_summary = read_json(SEVEN_REF272_SUMMARY)
    raw_test = ref_metrics.get("raw_model_energy_as_formation", {}).get("test", {})
    append_ladder_row(
        rows,
        level="M0",
        experiment="SevenNet raw potential energy as formation energy",
        status="complete",
        n=raw_test.get("n", ""),
        mae_mev=raw_test.get("mae_mev_atom", ""),
        rmse_mev=raw_test.get("rmse_mev_atom", ""),
        p95_mev=raw_test.get("p95_abs_mev_atom", ""),
        source=str(SEVEN_REF272 / "metrics.json"),
        selection_policy="none; diagnostic only",
        notes="Raw potential energy is not MP formation energy.",
        claim_strength="strong",
    )
    append_ladder_row(
        rows,
        level="M1",
        experiment="SevenNet scalar train-only calibrated reference",
        status="complete",
        n=2048,
        mae_mev=ref_summary.get("element_reference_test_mae", ""),
        source=str(SEVEN_REF272_SUMMARY),
        selection_policy="element reference fit on train split only",
        notes=f"Affine/composition reference MAE: {ref_summary.get('affine_reference_test_mae', '')} meV/atom.",
        claim_strength="strong",
    )
    m2 = load_m2_row()
    append_ladder_row(
        rows,
        level="M2",
        experiment="SevenNet scalar reference + real FE-GX residual",
        status="complete_single_seed" if m2 else "missing",
        n=m2.get("n", ""),
        mae_mev=m2.get("mae_mev", ""),
        rmse_mev=m2.get("rmse_mev", ""),
        p95_mev=m2.get("p95_mev", ""),
        source=m2.get("source", ""),
        selection_policy="train/val only inside train_fegx1.py; seed20260528",
        notes="This is the 272 real FE-GX single SevenNet primary result.",
        claim_strength="strong" if m2 else "blocked",
    )
    m3 = summaries.get("M3") or read_json(args.out_dir / "native_embedding/sevennet_native_embedding_frozen_adapter_summary.json")
    append_ladder_row(
        rows,
        level="M3",
        experiment="SevenNet native embedding frozen ridge adapter",
        status=m3.get("status", "not_run"),
        n=m3.get("best_test_n", ""),
        mae_mev=m3.get("best_test_mae_mev", ""),
        rmse_mev=m3.get("best_test_rmse_mev", ""),
        p95_mev=m3.get("best_test_p95_mev", ""),
        source=m3.get("metrics_csv", ""),
        selection_policy="fit train only; choose alpha/family on val; evaluate test once",
        notes=f"Best variant: {m3.get('best_val_variant', '')}.",
        claim_strength="strong" if m3.get("status") == "complete" else "blocked",
    )
    m4 = summaries.get("M4") or read_json(args.out_dir / "m4_summary.json")
    m4_metrics: dict[str, Any] = {}
    if m4 and Path(str(m4.get("metrics", ""))).exists():
        exp = f"fegx_279B_sevennet_native_embedding_reference_seed{args.seed}"
        m4_metrics = metric_from_json(Path(str(m4["metrics"])), exp)
    append_ladder_row(
        rows,
        level="M4",
        experiment="SevenNet native embedding adapter reference + real FE-GX residual",
        status=m4.get("status", "not_run") if m4 else "not_run",
        n=m4_metrics.get("n", ""),
        mae_mev=m4_metrics.get("mae_mev", ""),
        rmse_mev=m4_metrics.get("rmse_mev", ""),
        p95_mev=m4_metrics.get("p95_mev", ""),
        source=m4.get("metrics", "") if m4 else "",
        selection_policy="M3 selected on val; FE-GX early stopping on val; test final",
        notes="Real train_fegx1.py residual on the M3 reference.",
        claim_strength="strong" if m4_metrics else ("blocked" if m4.get("status", "").startswith("blocked") else "inconclusive"),
    )
    for level, desc in (
        ("M5", "SevenNet head-only formation-energy fine-tune"),
        ("M6", "SevenNet partial formation-energy fine-tune"),
    ):
        summary = summaries.get(level) or read_json(args.out_dir / level.lower() / f"sevennet_{level.lower()}_summary.json")
        append_ladder_row(
            rows,
            level=level,
            experiment=desc,
            status=summary.get("status", "not_run"),
            n=2048 if summary.get("status") == "complete" and not any(split_limits(args).values()) else "",
            mae_mev=summary.get("test_mae_mev", ""),
            rmse_mev=summary.get("test_rmse_mev", ""),
            p95_mev=summary.get("test_p95_mev", ""),
            source=summary.get("metrics_csv", ""),
            selection_policy="train formation-energy labels; early stop/select on val; test final once",
            notes=(
                "Head modules only."
                if level == "M5"
                else "Head plus final interaction block."
            ),
            claim_strength="strong" if summary.get("status") == "complete" else "blocked",
        )
    return rows


def write_comparisons(args: argparse.Namespace, ladder_rows: list[dict[str, Any]]) -> None:
    by_level = {str(row["level"]): row for row in ladder_rows}
    fields = ["level", "experiment", "status", "n", "mae_mev", "rmse_mev", "p95_mev", "source", "selection_policy", "notes", "claim_strength"]
    write_csv(args.out_dir / "sevennet_embedding_vs_scalar_comparison.csv", [by_level.get(k, {}) for k in ("M1", "M2", "M3", "M4")], fields)
    write_csv(args.out_dir / "sevennet_finetune_vs_frozen_comparison.csv", [by_level.get(k, {}) for k in ("M3", "M5", "M6")], fields)
    write_csv(args.out_dir / "sevennet_residual_transfer_metrics.csv", [by_level.get(k, {}) for k in ("M2", "M4")], fields)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def render_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_report(args: argparse.Namespace, runtime: dict[str, Any], graph_summary: dict[str, Any], ladder_rows: list[dict[str, Any]], summaries: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = [row for row in ladder_rows if str(row.get("level", "")).startswith("M")]
    by_level = {str(row["level"]): row for row in ladder_rows}
    completed = [
        row
        for row in report_rows
        if row.get("mae_mev") not in {"", None}
        and str(row.get("status", "")).startswith("complete")
    ]
    best_overall = min(completed, key=lambda row: float(row["mae_mev"])) if completed else None
    ft_complete = [
        row
        for row in (by_level.get("M5", {}), by_level.get("M6", {}))
        if row.get("mae_mev") not in {"", None}
    ]
    best_ft = min(ft_complete, key=lambda row: float(row["mae_mev"])) if ft_complete else None
    if best_overall:
        conclusion = (
            f"SevenNet current best on this medium ladder is {best_overall['level']} "
            f"at {fmt(best_overall['mae_mev'])} meV/atom test MAE."
        )
    else:
        conclusion = "SevenNet upper-bound ladder is not complete yet."
    ft_conclusion = (
        f"Best explicit fine-tune step is {best_ft['level']} at {fmt(best_ft['mae_mev'])} meV/atom test MAE."
        if best_ft
        else "Explicit fine-tune steps M5/M6 are not complete yet."
    )
    scope = "full 269 medium subset" if not any(split_limits(args).values()) else "limited engineering smoke"
    lines = [
        "# 279B SevenNet Capability Upper Bound",
        "",
        "MACE formal medium fine-tune was blocked, so 279B switches the same capability ladder to SevenNet-7net0. This remains a method-transfer / upper-bound question, not an EqV3/MatRIS replacement claim.",
        "",
        "## Scope",
        "",
        f"- data scope: {scope}",
        f"- graph counts: {graph_summary.get('counts', {})}",
        f"- SevenNet checkpoint: {runtime.get('sevennet_checkpoint', '')}",
        f"- CUDA available in runtime: {runtime.get('torch_cuda_available', '')}",
        "- test labels are evaluation-only; train-only calibration and val-only model selection are used.",
        "",
        "## Ladder",
        "",
        render_table(report_rows, ["level", "experiment", "status", "mae_mev", "rmse_mev", "p95_mev", "claim_strength"]),
        "",
        "## Conclusion",
        "",
        f"- {conclusion}",
        f"- {ft_conclusion}",
        "- M1/M2 reuse completed 272 real FE-GX evidence; M3-M6 are SevenNet-specific upper-bound probes.",
        "- M5 is head-only: only final readout/rescale modules are trainable. M6 is partial: M5 plus the final interaction block.",
        "",
        "## Files",
        "",
        f"- ladder: {args.out_dir / 'sevennet_capability_ladder_metrics.csv'}",
        f"- embedding/scalar comparison: {args.out_dir / 'sevennet_embedding_vs_scalar_comparison.csv'}",
        f"- fine-tune comparison: {args.out_dir / 'sevennet_finetune_vs_frozen_comparison.csv'}",
        f"- runtime: {args.out_dir / 'sevennet_runtime_status.json'}",
    ]
    text = "\n".join(lines) + "\n"
    (args.out_dir / "279B_sevennet_upper_bound.md").write_text(text, encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.run_all:
        args.run_m3 = args.run_m4 = args.run_m5 = args.run_m6 = True
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime_status(args)
    (args.out_dir / "sevennet_runtime_status.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n")
    graph_summary = prepare_graph_cache(args)
    graphs, meta = load_graphs(args)

    summaries: dict[str, Any] = {}
    if not args.summarize_only:
        if args.run_m3 or args.run_m4:
            summaries["M3"] = run_m3_embedding_adapter(args, graphs, meta)
        if args.run_m4:
            summaries["M4"] = run_m4_embedding_plus_fegx(args, summaries.get("M3"))
            (args.out_dir / "m4_summary.json").write_text(json.dumps(summaries["M4"], indent=2, ensure_ascii=False) + "\n")
        if args.run_m5:
            summaries["M5"] = train_finetune_level(args, "M5", graphs, meta)
        if args.run_m6:
            summaries["M6"] = train_finetune_level(args, "M6", graphs, meta)

    ladder_rows = build_ladder(args, runtime, summaries)
    fields = ["level", "experiment", "status", "split", "n", "mae_mev", "rmse_mev", "p95_mev", "source", "selection_policy", "notes", "claim_strength"]
    write_csv(args.out_dir / "sevennet_capability_ladder_metrics.csv", ladder_rows, fields)
    write_comparisons(args, ladder_rows)
    write_report(args, runtime, graph_summary, ladder_rows, summaries)
    summary = {
        "status": "complete",
        "out_dir": str(args.out_dir),
        "report": str(args.out_dir / "279B_sevennet_upper_bound.md"),
        "copied_report": str(REPORT),
        "data_scope": "full_medium" if not any(split_limits(args).values()) else "limited",
        "graph_summary": graph_summary,
        "runtime": runtime,
        "summaries": summaries,
    }
    (args.out_dir / "279B_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
