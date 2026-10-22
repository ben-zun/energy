from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase.io import read as ase_read
from torch import nn
from torch.nn.utils import parametrize

from rclfa_v3.data import normalize_split, target_from_row


ROOT = Path("/home/yxzhang/energy")
DEFAULT_EQV3_REPO = ROOT / "external_repos" / "equiformer_v3-main"


def ensure_scipy_sph_harm_compat() -> None:
    try:
        import scipy.special as special
    except Exception:
        return
    if hasattr(special, "sph_harm"):
        return
    if not hasattr(special, "sph_harm_y"):
        return

    def sph_harm(m, n, theta, phi):
        return special.sph_harm_y(n, m, phi, theta)

    special.sph_harm = sph_harm


def ensure_eqv3_imports(eqv3_repo: Path = DEFAULT_EQV3_REPO) -> None:
    eqv3_repo = eqv3_repo.resolve()
    for path in (eqv3_repo / "src", eqv3_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    ensure_scipy_sph_harm_compat()
    import experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


def set_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if ":" in device:
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    else:
        torch.cuda.set_device(0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    pd.DataFrame(rows).to_csv(path, mode="a", header=not exists, index=False)


@dataclass(frozen=True)
class SplitRow:
    split: str
    sample_id: str
    cif_path: str
    target_per_atom_ev: float


@dataclass
class SplitStructureRecord:
    split: str
    sample_id: str
    cif_path: str
    natoms: int
    target_per_atom_ev: float
    target_total_ev: float
    reference_per_atom_ev: float
    reference_total_ev: float
    data_object: Any


@dataclass
class GraphBatchBundle:
    batch: Any
    ids: list[str]
    splits: list[str]
    cif_paths: list[str]
    natoms: np.ndarray
    y_true_per_atom_ev: np.ndarray
    y_true_total_ev: np.ndarray
    reference_per_atom_ev: np.ndarray
    reference_total_ev: np.ndarray
    target_scaled: torch.Tensor


def read_split_rows(
    dataset_root: Path,
    workspace_root: Path,
    split: str,
    max_samples: int | None = None,
) -> list[SplitRow]:
    path = dataset_root / split / f"{split}.csv"
    if not path.exists():
        return []
    rows: list[SplitRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cif_path = Path(row["cif_path"])
            if not cif_path.is_absolute():
                cif_path = workspace_root / cif_path
            rows.append(
                SplitRow(
                    split=split,
                    sample_id=str(row["id"]),
                    cif_path=str(cif_path),
                    target_per_atom_ev=float(target_from_row(row)),
                )
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_dataset_manifest(
    dataset_root: Path,
    workspace_root: Path,
    out_csv: Path,
) -> Path:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        for row in read_split_rows(dataset_root, workspace_root, split, max_samples=None):
            rows.append(
                {
                    "split": row.split,
                    "id": row.sample_id,
                    "cif_path": row.cif_path,
                    "y_true": row.target_per_atom_ev,
                }
            )
    ensure_dir(out_csv.parent)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def load_reference_map(reference_csv: Path, reference_column: str) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(reference_csv)
    required = {"split", "id", reference_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{reference_csv} missing required columns: {missing}")
    frame["split"] = frame["split"].map(normalize_split)
    frame["id"] = frame["id"].astype(str)
    frame[reference_column] = pd.to_numeric(frame[reference_column], errors="coerce")
    bad = frame[reference_column].isna()
    if bool(bad.any()):
        example = frame.loc[bad, ["split", "id"]].head(5).to_dict("records")
        raise RuntimeError(f"{reference_csv} has NaN values in {reference_column}; example rows: {example}")
    return {
        (str(row.split), str(row.id)): float(getattr(row, reference_column))
        for row in frame[["split", "id", reference_column]].itertuples(index=False)
    }


def build_structure_records(
    *,
    dataset_root: Path,
    workspace_root: Path,
    a2g: Any,
    reference_csv: Path | None = None,
    reference_column: str = "pred_element_reference",
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict[str, list[SplitStructureRecord]]:
    reference_map = load_reference_map(reference_csv, reference_column) if reference_csv else {}
    split_limits = {"train": max_train, "val": max_val, "test": max_test}
    out: dict[str, list[SplitStructureRecord]] = {}
    for split in ("train", "val", "test"):
        records: list[SplitStructureRecord] = []
        for idx, row in enumerate(
            read_split_rows(dataset_root, workspace_root, split, max_samples=split_limits[split]),
            start=1,
        ):
            atoms = ase_read(row.cif_path)
            natoms = int(len(atoms))
            if reference_csv is not None:
                key = (split, row.sample_id)
                if key not in reference_map:
                    raise KeyError(f"Missing reference value for split={split!r}, id={row.sample_id!r} in {reference_csv}")
                ref_per_atom = float(reference_map[key])
            else:
                ref_per_atom = 0.0
            data_object = a2g.convert(atoms)
            data_object.sid = row.sample_id
            records.append(
                SplitStructureRecord(
                    split=split,
                    sample_id=row.sample_id,
                    cif_path=row.cif_path,
                    natoms=natoms,
                    target_per_atom_ev=row.target_per_atom_ev,
                    target_total_ev=row.target_per_atom_ev * natoms,
                    reference_per_atom_ev=ref_per_atom,
                    reference_total_ev=ref_per_atom * natoms,
                    data_object=data_object,
                )
            )
            if idx % 500 == 0:
                print(f"[eqv3-native] loaded {split} {idx} structures", flush=True)
        out[split] = records
    return out


def collate_graph_records(
    records: list[SplitStructureRecord],
    *,
    scale_divisor: float,
    residual_mode: bool,
) -> GraphBatchBundle:
    ensure_eqv3_imports()
    from fairchem.core.datasets import data_list_collater

    target_total = np.asarray([record.target_total_ev for record in records], dtype=np.float32)
    reference_total = np.asarray([record.reference_total_ev for record in records], dtype=np.float32)
    if residual_mode:
        target_total = target_total - reference_total
    return GraphBatchBundle(
        batch=data_list_collater([record.data_object for record in records], otf_graph=True),
        ids=[record.sample_id for record in records],
        splits=[record.split for record in records],
        cif_paths=[record.cif_path for record in records],
        natoms=np.asarray([record.natoms for record in records], dtype=np.int64),
        y_true_per_atom_ev=np.asarray([record.target_per_atom_ev for record in records], dtype=np.float32),
        y_true_total_ev=np.asarray([record.target_total_ev for record in records], dtype=np.float32),
        reference_per_atom_ev=np.asarray([record.reference_per_atom_ev for record in records], dtype=np.float32),
        reference_total_ev=reference_total,
        target_scaled=torch.as_tensor(target_total / float(scale_divisor), dtype=torch.float32),
    )


def iter_graph_batches(
    records: list[SplitStructureRecord],
    *,
    batch_size: int,
    scale_divisor: float,
    residual_mode: bool,
    shuffle: bool,
    seed: int,
):
    order = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_records = [records[idx] for idx in order[start : start + batch_size]]
        yield collate_graph_records(batch_records, scale_divisor=scale_divisor, residual_mode=residual_mode)


def require_eqv3_energy_model(model: Any) -> Any:
    if not hasattr(model, "energy_block"):
        raise TypeError("EqV3 native fine-tuning expects a model with a top-level energy_block")
    if not hasattr(model, "blocks"):
        raise TypeError("EqV3 native fine-tuning expects a model with transformer blocks")
    return model.energy_block


def configure_eqv3_energy_scaling(model: Any, scale_divisor: float | None = None) -> float:
    require_eqv3_energy_model(model)
    chosen = float(scale_divisor or getattr(model, "avg_num_nodes", 1.0) or 1.0)
    model.avg_num_nodes = chosen
    if hasattr(model, "regress_forces"):
        model.regress_forces = False
    if hasattr(model, "regress_stress"):
        model.regress_stress = False
    return chosen


def configure_eqv3_gradient_checkpointing(model: Any, enabled: bool = True) -> list[int]:
    num_layers = int(getattr(model, "num_layers", len(getattr(model, "blocks", []))))
    if num_layers < 1 or not hasattr(model, "gradient_checkpointing_block_list"):
        return []
    values = [1] * num_layers if enabled else [0] * num_layers
    model.gradient_checkpointing_block_list = values
    return values


class AdditiveLowRankParametrization(nn.Module):
    def __init__(self, weight_shape: tuple[int, ...], rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.weight_shape = tuple(int(dim) for dim in weight_shape)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        if len(self.weight_shape) == 2:
            out_features, in_features = self.weight_shape
            self.lora_A = nn.Parameter(torch.empty(out_features, self.rank))
            self.lora_B = nn.Parameter(torch.zeros(self.rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5.0))
        elif len(self.weight_shape) == 3:
            slices, out_features, in_features = self.weight_shape
            self.lora_A = nn.Parameter(torch.empty(slices, out_features, self.rank))
            self.lora_B = nn.Parameter(torch.zeros(slices, self.rank, in_features))
            bound = 1.0 / max(float(in_features), 1.0)
            nn.init.uniform_(self.lora_A, -bound, bound)
        else:
            raise ValueError(f"Unsupported LoRA weight shape: {self.weight_shape}")

    def delta(self) -> torch.Tensor:
        if len(self.weight_shape) == 2:
            return self.lora_A @ self.lora_B
        return torch.einsum("sor,sri->soi", self.lora_A, self.lora_B)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return base + self.delta().to(dtype=base.dtype, device=base.device) * self.scale


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def apply_lora_to_eqv3_backbone(
    model: Any,
    *,
    rank: int,
    alpha: float,
    target_prefixes: tuple[str, ...] = ("blocks.",),
) -> list[str]:
    ensure_eqv3_imports()
    from experimental.models.equiformer_v3.so3 import SO3Linear

    attached: list[str] = []
    for module_name, module in model.named_modules():
        if not any(module_name.startswith(prefix) for prefix in target_prefixes):
            continue
        if not isinstance(module, (nn.Linear, SO3Linear)):
            continue
        if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
            continue
        parametrization = AdditiveLowRankParametrization(tuple(module.weight.shape), rank=rank, alpha=alpha)
        parametrize.register_parametrization(module, "weight", parametrization, unsafe=True)
        module.parametrizations.weight.original.requires_grad = False
        attached.append(module_name)
    if not attached:
        raise RuntimeError("LoRA did not attach to any EqV2 backbone modules")
    return attached


def configure_eqv3_tuning_mode(
    model: Any,
    *,
    tuning_type: str,
    lora_rank: int,
    lora_alpha: float,
) -> list[str]:
    energy_head = require_eqv3_energy_model(model)
    freeze_module(model)
    attached_lora: list[str] = []
    if tuning_type == "head":
        unfreeze_module(energy_head)
        return attached_lora
    if tuning_type == "lora":
        attached_lora = apply_lora_to_eqv3_backbone(model, rank=lora_rank, alpha=lora_alpha)
        unfreeze_module(energy_head)
        return attached_lora
    if tuning_type == "all":
        unfreeze_module(model)
        unfreeze_module(energy_head)
        return attached_lora
    raise ValueError(f"Unknown tuning_type: {tuning_type}")


def summarize_parameter_stats(model: nn.Module) -> dict[str, float]:
    stats = {
        "total_parameters": 0.0,
        "trainable_parameters": 0.0,
        "trainable_fraction": 0.0,
        "trainable_backbone_parameters": 0.0,
        "trainable_energy_head_parameters": 0.0,
        "trainable_lora_parameters": 0.0,
        "frozen_other_parameters": 0.0,
    }
    for name, param in model.named_parameters():
        count = float(param.numel())
        stats["total_parameters"] += count
        if param.requires_grad:
            stats["trainable_parameters"] += count
            if name.startswith("blocks.") or name.startswith("sphere_embedding.") or name.startswith("edge_degree_embedding.") or name.startswith("norm.") or name.startswith("so3_rotation.") or name.startswith("distance_expansion.") or name.startswith("envelope_func."):
                stats["trainable_backbone_parameters"] += count
            if name.startswith("energy_block."):
                stats["trainable_energy_head_parameters"] += count
            if ".parametrizations.weight.0.lora_" in name:
                stats["trainable_lora_parameters"] += count
        else:
            stats["frozen_other_parameters"] += count
    if stats["total_parameters"] > 0:
        stats["trainable_fraction"] = stats["trainable_parameters"] / stats["total_parameters"]
    return stats
