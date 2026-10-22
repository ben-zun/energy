#!/home/yxzhang/envs/pet_forme/bin/python
from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rclfa_v3.eqv3_native import (
    build_structure_records,
    configure_eqv3_gradient_checkpointing,
    configure_eqv3_tuning_mode,
    configure_eqv3_energy_scaling,
    ensure_dir,
    ensure_eqv3_imports,
    iter_graph_batches,
    save_json,
    seed_everything,
    set_cuda_device,
    summarize_parameter_stats,
)
from rclfa_v3.reference import regression_metrics


ROOT = Path("/home/yxzhang/energy")
DEFAULT_OUT = ROOT / "runs" / "rclfa_v3" / "eqv3_native_upper_bound_20260725"


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EqV3 native formation-energy adaptation with matched direct/residual routes."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workspace-root", default="/home/yxzhang/energy")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default="eqv3_native_upper_bound")
    parser.add_argument(
        "--checkpoint",
        default="/home/yxzhang/energy/checkpoints/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt",
    )
    parser.add_argument("--eqv3-repo", default="/home/yxzhang/energy/external_repos/equiformer_v3-main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--tuning-type", choices=["head", "lora", "all"], default="all")
    parser.add_argument("--reference-csv", default="")
    parser.add_argument("--reference-column", default="pred_element_reference")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def build_optimizer(
    model: nn.Module,
    *,
    lr_head: float,
    lr_backbone: float,
    lr_lora: float,
    weight_decay: float,
) -> AdamW:
    head_params = []
    backbone_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".parametrizations.weight.0.lora_" in name:
            lora_params.append(param)
        elif name.startswith("energy_block."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters selected.")
    return AdamW(param_groups)


def run_epoch(
    model: nn.Module,
    records,
    *,
    batch_size: int,
    scale_divisor: float,
    residual_mode: bool,
    device: torch.device,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: GradScaler | None,
    clip_grad_norm: float,
    seed: int,
    amp_dtype: torch.dtype | None,
    grad_accum_steps: int,
    stop_requested: list[bool],
):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    pred_rows: list[dict[str, object]] = []
    batch_count = 0
    total_batches = max(1, (len(records) + batch_size - 1) // batch_size)
    for batch_idx, bundle in enumerate(
        iter_graph_batches(
            records,
            batch_size=batch_size,
            scale_divisor=scale_divisor,
            residual_mode=residual_mode,
            shuffle=is_train,
            seed=seed,
        ),
        start=1,
    ):
        batch_count += 1
        batch = bundle.batch.to(device)
        target_scaled = bundle.target_scaled.to(device)
        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                scaled_pred = model(batch)["energy"].reshape(-1)
                loss = criterion(scaled_pred, target_scaled)
            if is_train:
                loss_to_backprop = loss / float(max(1, grad_accum_steps))
                if scaler is not None:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                should_step = (batch_idx % max(1, grad_accum_steps) == 0) or (batch_idx == total_batches)
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        scaled_pred_np = scaled_pred.detach().float().cpu().numpy()
        pred_total = scaled_pred_np * float(scale_divisor)
        if residual_mode:
            pred_total = pred_total + bundle.reference_total_ev
        pred_per_atom = pred_total / bundle.natoms
        losses.append(float(loss.detach().cpu()))
        for idx, sample_id in enumerate(bundle.ids):
            pred_rows.append(
                {
                    "split": bundle.splits[idx],
                    "id": sample_id,
                    "cif_path": bundle.cif_paths[idx],
                    "natoms": int(bundle.natoms[idx]),
                    "reference_per_atom_ev": float(bundle.reference_per_atom_ev[idx]),
                    "reference_total_ev": float(bundle.reference_total_ev[idx]),
                    "y_true_total_ev": float(bundle.y_true_total_ev[idx]),
                    "y_pred_total_ev": float(pred_total[idx]),
                    "y_true_per_atom_ev": float(bundle.y_true_per_atom_ev[idx]),
                    "y_pred_per_atom_ev": float(pred_per_atom[idx]),
                    "abs_error_mev_atom": float(abs(pred_per_atom[idx] - bundle.y_true_per_atom_ev[idx]) * 1000.0),
                }
            )
        if stop_requested[0]:
            print("[eqv3-native] stop requested; finishing current epoch early", flush=True)
            break

    pred_frame = pd.DataFrame(pred_rows)
    metrics = regression_metrics(
        pred_frame["y_true_per_atom_ev"].to_numpy(dtype=np.float64),
        pred_frame["y_pred_per_atom_ev"].to_numpy(dtype=np.float64),
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["num_batches"] = int(batch_count)
    return metrics, pred_frame


def evaluate_all_splits(
    model: nn.Module,
    records_by_split: dict[str, list],
    *,
    batch_size: int,
    scale_divisor: float,
    residual_mode: bool,
    device: torch.device,
    criterion: nn.Module,
    amp_dtype: torch.dtype | None,
):
    metric_rows = []
    pred_frames = []
    for split in ("train", "val", "test"):
        if not records_by_split.get(split):
            continue
        metrics, frame = run_epoch(
            model,
            records_by_split[split],
            batch_size=batch_size,
            scale_divisor=scale_divisor,
            residual_mode=residual_mode,
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            clip_grad_norm=0.0,
            seed=0,
            amp_dtype=amp_dtype,
            grad_accum_steps=1,
            stop_requested=[False],
        )
        metrics["split"] = split
        metric_rows.append(metrics)
        pred_frames.append(frame)
    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def resolve_amp_dtype(device: torch.device, amp_name: str) -> tuple[torch.dtype | None, GradScaler | None]:
    if device.type != "cuda" or amp_name == "none":
        return None, None
    if amp_name == "bfloat16":
        return torch.bfloat16, None
    if amp_name == "float16":
        return torch.float16, GradScaler("cuda")
    raise ValueError(f"Unsupported amp dtype: {amp_name}")


def save_epoch_artifacts(
    *,
    run_dir: Path,
    run_name: str,
    history_rows: list[dict[str, object]],
    model: nn.Module,
    args: argparse.Namespace,
    scale_divisor: float,
    parameter_stats: dict[str, float],
    lora_modules: list[str],
    best_epoch: int,
    best_val: float,
    checkpointing_blocks: list[int],
    epoch: int,
) -> None:
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / f"{run_name}_history.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "scale_divisor": scale_divisor,
            "parameter_stats": parameter_stats,
            "lora_modules": lora_modules,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "last_epoch": epoch,
        },
        run_dir / f"{run_name}_latest.pt",
    )
    save_json(
        run_dir / f"{run_name}_latest.json",
        {
            "status": "running",
            "run_name": run_name,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_mae_mev_atom": best_val,
            "history_csv": str(run_dir / f"{run_name}_history.csv"),
            "latest_checkpoint_path": str(run_dir / f"{run_name}_latest.pt"),
            "gradient_checkpointing_blocks": checkpointing_blocks,
        },
    )


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(args.out_dir.resolve())
    log_path = run_dir / f"{args.run_name}.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(orig_stdout, log_handle)
    sys.stderr = TeeStream(orig_stderr, log_handle)
    t0 = time.time()

    try:
        _main_impl(args, run_dir, log_path, t0)
    except Exception as exc:
        save_json(
            run_dir / f"{args.run_name}_summary.json",
            {
                "status": "failed",
                "run_name": args.run_name,
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "workspace_root": str(Path(args.workspace_root).resolve()),
                "device": args.device,
                "target_mode": args.target_mode,
                "tuning_type": args.tuning_type,
                "runtime_s": time.time() - t0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "log_path": str(log_path),
            },
        )
        raise
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_handle.close()


def _main_impl(args: argparse.Namespace, run_dir: Path, log_path: Path, turn_t0: float) -> None:
    if args.target_mode == "residual" and not args.reference_csv:
        raise ValueError("--target-mode residual requires --reference-csv")
    if args.device.startswith("cuda"):
        set_cuda_device(args.device)
    ensure_eqv3_imports(Path(args.eqv3_repo))
    from fairchem.core import OCPCalculator

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    seed_everything(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") else "cpu")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"EqV3 checkpoint not found: {checkpoint}")

    calc = OCPCalculator(checkpoint_path=checkpoint, cpu=(device.type != "cuda"), seed=args.seed, only_output=["energy"])
    calc.trainer.scaler = None
    model = getattr(calc.trainer, "_unwrapped_model", None) or calc.trainer.model
    scale_divisor = configure_eqv3_energy_scaling(model)
    checkpointing_blocks = configure_eqv3_gradient_checkpointing(
        model,
        enabled=not args.disable_gradient_checkpointing,
    )
    model = model.to(device)
    lora_modules = configure_eqv3_tuning_mode(
        model,
        tuning_type=args.tuning_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    parameter_stats = summarize_parameter_stats(model)

    records_by_split = build_structure_records(
        dataset_root=Path(args.dataset_root),
        workspace_root=Path(args.workspace_root).resolve(),
        a2g=calc.a2g,
        reference_csv=Path(args.reference_csv) if args.reference_csv else None,
        reference_column=args.reference_column,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )
    split_counts = {split: len(records) for split, records in records_by_split.items()}
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = build_optimizer(
        model,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        lr_lora=args.lr_lora,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 2))
    amp_dtype, scaler = resolve_amp_dtype(device, args.amp_dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    residual_mode = args.target_mode == "residual"
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history_rows: list[dict[str, object]] = []
    t0 = time.time()
    stop_requested = [False]

    def _request_stop(signum, _frame):
        stop_requested[0] = True
        print(f"[eqv3-native] received signal {signum}; stopping after current epoch", flush=True)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        train_metrics, _ = run_epoch(
            model,
            records_by_split["train"],
            batch_size=args.batch_size,
            scale_divisor=scale_divisor,
            residual_mode=residual_mode,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad_norm=args.clip_grad_norm,
            seed=args.seed + epoch,
            amp_dtype=amp_dtype,
            grad_accum_steps=args.grad_accum_steps,
            stop_requested=stop_requested,
        )
        val_metrics = {"loss": float("nan"), "mae_mev_atom": float("nan")}
        if records_by_split.get("val"):
            val_metrics, _ = run_epoch(
                model,
                records_by_split["val"],
                batch_size=args.batch_size,
                scale_divisor=scale_divisor,
                residual_mode=residual_mode,
                device=device,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                clip_grad_norm=0.0,
                seed=0,
                amp_dtype=amp_dtype,
                grad_accum_steps=1,
                stop_requested=[False],
            )
            scheduler.step(float(val_metrics["loss"]))
            current_val = float(val_metrics["mae_mev_atom"])
            if current_val < best_val:
                best_val = current_val
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            scheduler.step(float(train_metrics["loss"]))

        lr_state = {
            f"lr_group_{idx}": float(group["lr"])
            for idx, group in enumerate(optimizer.param_groups)
        }
        epoch_peak_vram_mb = 0.0
        if device.type == "cuda":
            epoch_peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_mae_mev_atom": float(train_metrics.get("mae_mev_atom", float("nan"))),
                "train_num_batches": int(train_metrics.get("num_batches", 0)),
                "val_loss": float(val_metrics["loss"]),
                "val_mae_mev_atom": float(val_metrics.get("mae_mev_atom", float("nan"))),
                "val_num_batches": int(val_metrics.get("num_batches", 0)),
                "epoch_time_s": float(time.time() - epoch_t0),
                "epoch_peak_vram_mb": epoch_peak_vram_mb,
                **lr_state,
            }
        )
        save_epoch_artifacts(
            run_dir=run_dir,
            run_name=args.run_name,
            history_rows=history_rows,
            model=model,
            args=args,
            scale_divisor=scale_divisor,
            parameter_stats=parameter_stats,
            lora_modules=lora_modules,
            best_epoch=best_epoch,
            best_val=best_val,
            checkpointing_blocks=checkpointing_blocks,
            epoch=epoch,
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[eqv3-native] epoch={epoch} train_mae={float(train_metrics.get('mae_mev_atom', float('nan'))):.3f} "
                f"val_mae={float(val_metrics.get('mae_mev_atom', float('nan'))):.3f}",
                flush=True,
            )
        if stop_requested[0]:
            print(f"[eqv3-native] stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
        if records_by_split.get("val") and stale_epochs >= args.patience:
            print(f"[eqv3-native] early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    runtime_s = time.time() - t0
    model.load_state_dict(best_state)
    metrics_df, preds_df = evaluate_all_splits(
        model,
        records_by_split,
        batch_size=args.batch_size,
        scale_divisor=scale_divisor,
        residual_mode=residual_mode,
        device=device,
        criterion=criterion,
        amp_dtype=amp_dtype,
    )
    history_df = pd.DataFrame(history_rows)

    metrics_path = run_dir / f"{args.run_name}_metrics.csv"
    preds_path = run_dir / f"{args.run_name}_predictions.csv"
    history_path = run_dir / f"{args.run_name}_history.csv"
    ckpt_path = run_dir / f"{args.run_name}_best.pt"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    history_df.to_csv(history_path, index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "scale_divisor": scale_divisor,
            "parameter_stats": parameter_stats,
            "lora_modules": lora_modules,
        },
        ckpt_path,
    )

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))

    summary = {
        "status": "complete",
        "run_name": args.run_name,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "workspace_root": str(Path(args.workspace_root).resolve()),
        "device": args.device,
        "seed": args.seed,
        "target_mode": args.target_mode,
        "tuning_type": args.tuning_type,
        "checkpoint": str(checkpoint),
        "eqv3_repo": str(Path(args.eqv3_repo).resolve()),
        "reference_csv": str(Path(args.reference_csv).resolve()) if args.reference_csv else "",
        "reference_column": args.reference_column,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "scale_divisor": float(scale_divisor),
        "parameter_stats": parameter_stats,
        "lora_modules": lora_modules,
        "gradient_checkpointing_blocks": checkpointing_blocks,
        "split_counts": split_counts,
        "runtime_s": runtime_s,
        "peak_vram_mb": peak_vram_mb,
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(preds_path),
        "history_csv": str(history_path),
        "checkpoint_path": str(ckpt_path),
        "log_path": str(log_path),
        "target_contract": (
            "direct: model predicts total formation energy scaled by model.avg_num_nodes"
            if not residual_mode
            else "residual: fixed calibrated reference total energy plus model-predicted residual total energy, both scaled by model.avg_num_nodes"
        ),
    }
    save_json(run_dir / f"{args.run_name}_summary.json", summary)
    print(run_dir / f"{args.run_name}_summary.json")


if __name__ == "__main__":
    main()
