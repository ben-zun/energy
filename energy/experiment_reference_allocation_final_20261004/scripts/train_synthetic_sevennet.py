from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

CONDITIONS = {
    "zero_direct": {"route": "U", "target": "target_zero_ev_atom", "shift": "zero"},
    "reference_direct": {"route": "U", "target": "target_reference_ev_atom", "shift": "reference"},
    "reference_aligned": {"route": "A", "target": "target_reference_ev_atom", "shift": "reference"},
}


def atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def capture_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "random_state_python": random.getstate(),
        "random_state_numpy": np.random.get_state(),
        "random_state_torch": torch.get_rng_state(),
        "random_state_cuda": torch.cuda.get_rng_state_all(),
        "data_loader_generator_state": generator.get_state(),
    }


def restore_rng_state(payload: dict[str, object], generator: torch.Generator) -> None:
    random.setstate(payload["random_state_python"])
    np.random.set_state(payload["random_state_numpy"])
    torch.set_rng_state(payload["random_state_torch"])
    torch.cuda.set_rng_state_all(payload["random_state_cuda"])
    generator.set_state(payload["data_loader_generator_state"])


def centered(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return (series - series.groupby(groups).transform("mean")).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--controlled-stop-after-epoch", type=int)
    args = parser.parse_args()

    energy = args.energy_root.resolve(strict=True)
    experiment = args.experiment_root.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels.resolve(strict=True)
    upstream = energy / "experiment12_20260929_alignment"
    sys.path.insert(0, str(upstream / "code"))
    from alignment_common import (
        BATCH_EVAL,
        BATCH_TRAIN,
        CALIBRATION_ALPHA,
        HUBER_BETA,
        LR,
        PREFIXES,
        TRAINABLE_PARAMETERS,
        WEIGHT_DECAY,
        alignment_offset,
        apply_post_calibration,
        atomic_fractions,
        build_model,
        fit_post_calibration,
        group_labels,
        load_element_reference,
        load_graphs,
        load_meta,
        route_prediction,
        trainable_state_hash,
    )

    condition = CONDITIONS[args.condition]
    run = experiment / "synthetic_reference_control" / ("smoke" if args.smoke else "runs")
    run = run / args.condition / f"seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
        print(status_path.read_text())
        return

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    meta = load_meta()
    labels = pd.read_csv(labels_path, keep_default_na=False)
    if not np.array_equal(meta.id.to_numpy(), labels.id.to_numpy()):
        raise AssertionError("synthetic labels are not exactly aligned to graph metadata")
    target_values = labels[condition["target"]].to_numpy(np.float64)
    pretrained_values = labels.pretrained_ev_atom.to_numpy(np.float64)
    if not np.isfinite(target_values).all() or not np.isfinite(pretrained_values).all():
        raise FloatingPointError("synthetic labels are nonfinite")
    fractions = atomic_fractions(meta)
    groups = group_labels(meta)
    mu, bias = load_element_reference()
    offset = alignment_offset(fractions, mu, bias)
    np.testing.assert_allclose(offset, labels.reference_shift_ev_atom, atol=1e-10, rtol=0)
    split_idx = {split: np.flatnonzero(meta.split.to_numpy() == split) for split in ("train", "val", "test")}
    if {key: len(value) for key, value in split_idx.items()} != {"train": 8192, "val": 1024, "test": 2048}:
        raise AssertionError("unexpected medium split sizes")

    graphs = load_graphs(("train", "val", "test"))
    model = build_model(device)
    initialization_hash = trainable_state_hash(model)
    initial_trainable = {
        name: parameter.detach().cpu().double().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = torch.nn.SmoothL1Loss(beta=HUBER_BETA)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        graphs["train"], batch_size=BATCH_TRAIN, shuffle=True, generator=generator
    )
    eval_loaders = {
        split: DataLoader(graphs[split], batch_size=BATCH_EVAL, shuffle=False)
        for split in ("train", "val")
    }
    target_tensor = torch.tensor(target_values, dtype=torch.float32, device=device)
    offset_tensor = torch.tensor(offset, dtype=torch.float32, device=device)

    def evaluate(split: str) -> np.ndarray:
        result = np.full(len(meta), np.nan, dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for batch in eval_loaders[split]:
                batch = batch.to(device)
                positions = batch.manifest_row.cpu().numpy()
                result[positions] = route_prediction(
                    condition["route"], model, batch, offset_tensor
                ).cpu().numpy()
        return result

    # Initialization checks are executed before loading any resumable state.
    initial_prediction = np.full(len(meta), np.nan, dtype=np.float64)
    initial_prediction[split_idx["val"]] = evaluate("val")[split_idx["val"]]
    val_rows = split_idx["val"]
    init_error = initial_prediction[val_rows] - target_values[val_rows]
    init_ec_drift = centered(initial_prediction[val_rows] - pretrained_values[val_rows], groups[val_rows])
    init_checks = {
        "initialization_hash": initialization_hash,
        "val_initial_target_mae_mev_atom": float(1000 * np.mean(np.abs(init_error))),
        "val_initial_ec_drift_mae_mev_atom": float(1000 * np.mean(np.abs(init_ec_drift))),
        "reference_offset_match_max_abs_ev_atom": float(
            np.max(np.abs(offset - labels.reference_shift_ev_atom.to_numpy()))
        ),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if init_checks["trainable_parameter_count"] != TRAINABLE_PARAMETERS:
        raise AssertionError("trainable parameter count changed")
    if (
        args.condition in {"zero_direct", "reference_aligned"}
        and init_checks["val_initial_target_mae_mev_atom"] > 0.02
    ):
        raise AssertionError(f"oracle-like condition is not initialized at its target: {init_checks}")

    checkpoint_paths = sorted(run.glob("epoch*.pt"))
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_val = float("inf")
    history: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    if checkpoint_paths:
        latest = checkpoint_paths[-1]
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload["condition"] != args.condition or int(payload["seed"]) != seed:
            raise AssertionError("resume checkpoint belongs to another run")
        if payload["initialization_hash"] != initialization_hash:
            raise AssertionError("resume initialization hash mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload, generator)
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_epoch = int(payload["best_epoch"])
        best_val = float(payload["best_val_calibrated_mae_mev_atom"])
        history = pd.read_csv(run / "history.csv").to_dict("records")
        calibration_rows = pd.read_csv(run / "calibration_history.csv").to_dict("records")
        if len(history) != start_epoch - 1 or len(calibration_rows) != start_epoch - 1:
            raise AssertionError("resume history is inconsistent with latest full epoch")

    epochs = 1 if args.smoke else args.epochs
    atomic_json(
        {
            "status": "running",
            "condition": args.condition,
            "seed": seed,
            "route": condition["route"],
            "target_column": condition["target"],
            "max_epochs": epochs,
            "no_early_stopping": True,
            "started_or_resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "initialization_checks": init_checks,
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
            "loss": {"name": "SmoothL1", "beta_ev_atom": HUBER_BETA},
            "post_calibration_alpha": CALIBRATION_ALPHA,
            "trainable_prefixes": list(PREFIXES),
            "resume_from_epoch": start_epoch - 1,
        },
        status_path,
    )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = route_prediction(condition["route"], model, batch, offset_tensor)
            target = target_tensor[batch.manifest_row.long()]
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite synthetic training loss")
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach()))
        training_seconds = time.perf_counter() - epoch_start

        evaluation_start = time.perf_counter()
        native = np.full(len(meta), np.nan, dtype=np.float64)
        for split in ("train", "val"):
            values = evaluate(split)
            native[split_idx[split]] = values[split_idx[split]]
        train_rows = split_idx["train"]
        calibration = fit_post_calibration(
            fractions[train_rows], target_values[train_rows] - native[train_rows]
        )
        val_prediction = native[val_rows]
        val_calibrated = val_prediction + apply_post_calibration(calibration, fractions[val_rows])
        val_target = target_values[val_rows]
        val_ec_drift = centered(val_prediction - pretrained_values[val_rows], groups[val_rows])
        val_calibrated_ec_drift = centered(
            val_calibrated - pretrained_values[val_rows], groups[val_rows]
        )
        parameter_squared = 0.0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                difference = parameter.detach().cpu().double() - initial_trainable[name]
                parameter_squared += float(torch.sum(difference * difference))
        val_calibrated_mae = float(1000 * np.mean(np.abs(val_calibrated - val_target)))
        if val_calibrated_mae < best_val - 1e-12:
            best_val = val_calibrated_mae
            best_epoch = epoch
        history.append(
            {
                "condition": args.condition,
                "seed": seed,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "val_native_mae_mev_atom": float(1000 * np.mean(np.abs(val_prediction - val_target))),
                "val_calibrated_mae_mev_atom": val_calibrated_mae,
                "val_ec_drift_mev_atom": float(1000 * np.mean(np.abs(val_ec_drift))),
                "val_calibrated_ec_drift_mev_atom": float(
                    1000 * np.mean(np.abs(val_calibrated_ec_drift))
                ),
                "parameter_update_l2": float(np.sqrt(parameter_squared)),
                "training_seconds": training_seconds,
                "evaluation_seconds": time.perf_counter() - evaluation_start,
            }
        )
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        calibration_rows.append(
            {
                "condition": args.condition,
                "seed": seed,
                "epoch": epoch,
                **{
                    f"w_{number}": float(value)
                    for number, value in enumerate(calibration.coef_, start=1)
                },
                "b": float(calibration.intercept_),
            }
        )
        pd.DataFrame(calibration_rows).to_csv(run / "calibration_history.csv", index=False)
        state: dict[str, object] = {
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "grad_scaler_state": None,
            "epoch": epoch,
            "global_step": global_step,
            "condition": args.condition,
            "seed": seed,
            "initialization_hash": initialization_hash,
            "best_epoch": best_epoch,
            "best_val_calibrated_mae_mev_atom": best_val,
            **capture_rng_state(generator),
        }
        atomic_torch_save(state, run / f"epoch{epoch:03d}.pt")
        atomic_json(
            {
                "status": "running",
                "condition": args.condition,
                "seed": seed,
                "completed_epochs": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_calibrated_mae_mev_atom": best_val,
                "initialization_checks": init_checks,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            status_path,
        )
        print(json.dumps(history[-1]), flush=True)
        if args.controlled_stop_after_epoch == epoch:
            raise RuntimeError(
                f"controlled stop after complete epoch {epoch}; rerun without "
                "--controlled-stop-after-epoch to exercise exact resume"
            )

    final_status = {
        "status": "smoke_complete" if args.smoke else "complete",
        "condition": args.condition,
        "seed": seed,
        "completed_epochs": epochs,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_val_calibrated_mae_mev_atom": best_val,
        "initialization_checks": init_checks,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(final_status, status_path)
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
