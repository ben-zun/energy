from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from reference_allocation.synthetic import (
    audit_synthetic_targets,
    build_synthetic_targets,
    fold_intercept_into_element_coefficients,
)
from torch_geometric.loader import DataLoader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    energy = args.energy_root.resolve(strict=True)
    upstream = energy / "experiment12_20260929_alignment"
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(upstream / "code"))
    from alignment_common import (
        BATCH_EVAL,
        atomic_fractions,
        build_model,
        group_labels,
        load_element_reference,
        load_graphs,
        load_meta,
        per_atom,
        trainable_state_hash,
    )

    meta = load_meta()
    if meta.id.duplicated().any() or len(meta) != 11264:
        raise AssertionError("unexpected or duplicate medium-split metadata")
    split_counts = meta.split.value_counts().to_dict()
    if split_counts != {"train": 8192, "test": 2048, "val": 1024}:
        raise AssertionError(f"unexpected split counts: {split_counts}")
    fractions = atomic_fractions(meta)
    mu, bias = load_element_reference()
    delta_mu = fold_intercept_into_element_coefficients(mu, bias)
    graphs = load_graphs(("train", "val", "test"))
    model = build_model(torch.device(args.device))
    model.eval()
    pretrained = np.full(len(meta), np.nan, dtype=np.float64)
    with torch.no_grad():
        for split in ("train", "val", "test"):
            for batch in DataLoader(graphs[split], batch_size=BATCH_EVAL, shuffle=False):
                batch = batch.to(args.device)
                pretrained[batch.manifest_row.cpu().numpy()] = per_atom(model, batch).cpu().numpy()
    if not np.isfinite(pretrained).all():
        raise FloatingPointError("teacher prediction is incomplete or nonfinite")
    target_zero, target_reference = build_synthetic_targets(pretrained, fractions, delta_mu)
    groups = group_labels(meta)
    audit = audit_synthetic_targets(
        pretrained_per_atom=pretrained,
        target_reference_per_atom=target_reference,
        fractions=fractions,
        natoms=meta.natoms.to_numpy(),
        delta_mu=delta_mu,
        composition_keys=groups,
    )
    repeated = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy() >= 2
    oracle_centered_drift = (
        pd.Series(target_reference - pretrained)
        - pd.Series(target_reference - pretrained).groupby(groups).transform("mean")
    ).to_numpy()
    audit["oracle_ec_drift_max_abs_ev_atom"] = float(np.max(np.abs(oracle_centered_drift[repeated])))
    if audit["oracle_ec_drift_max_abs_ev_atom"] > 1e-10:
        raise AssertionError("oracle reference changes same-composition relative energy")
    audit["reference_coordinate_dependency"] = "none: label shift uses atomic fractions only"
    audit["teacher_trainable_state_hash"] = trainable_state_hash(model)
    audit["teacher_full_state_hash"] = state_hash(model)

    labels = meta[["id", "split", "formula", "natoms", "y_true"]].copy()
    labels = labels.rename(columns={"y_true": "unused_real_target_ev_atom"})
    labels["composition_key"] = groups
    labels["pretrained_ev_atom"] = pretrained
    labels["reference_shift_ev_atom"] = target_reference - pretrained
    labels["target_zero_ev_atom"] = target_zero
    labels["target_reference_ev_atom"] = target_reference
    label_path = output / "synthetic_labels.csv"
    labels.to_csv(label_path, index=False)
    coefficient_path = output / "reference_coefficients.csv"
    pd.DataFrame(
        {
            "atomic_number": np.arange(1, 119),
            "source_mu": mu,
            "source_bias": bias,
            "delta_mu_with_bias_folded": delta_mu,
        }
    ).to_csv(coefficient_path, index=False)
    audit["synthetic_labels_sha256"] = sha256(label_path)
    audit["reference_coefficients_sha256"] = sha256(coefficient_path)
    audit["status"] = "PASS"
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
