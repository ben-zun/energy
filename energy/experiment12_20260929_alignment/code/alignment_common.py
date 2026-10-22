"""Shared definitions for the matched U-vs-A long-budget comparison.

Route U (unaligned target):  y_hat_U = f_theta(X) / N_X
Route A (aligned target):    y_hat_A = f_theta(X) / N_X + d(z)

with d(z) = -reference_offset(z) the frozen element-reference calibration
(= E_base - f_theta0/N, audited in experiment11).  Both routes share one script
and differ only by --alignment.

Data, model and calibration paths are read-only references to earlier
experiments; nothing here writes outside experiment12_20260929_alignment/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Element

ENERGY = Path('/home/yxzhang/energy')
EXP11 = ENERGY / 'experiment11_20260928'
OUT = ENERGY / 'experiment12_20260929_alignment'

# medium split used by experiments 8/9/11: 8192 train / 1024 val / 2048 test
GRAPH_CACHE = (ENERGY / 'runs/rclfa_v3/fegx_component_function_deep_study_292_298_revision'
               / 'fair_finetune_293/graph_cache/medium')
# frozen element-reference calibration (audited: 118 mu_Z + bias, no scale term)
CALIBRATION_CACHE = (ENERGY / 'runs/rclfa_v3/real_fegx_foundation_replacement_272'
                     / 'reference_conversion/sevennet_reference/calibration_coefficients.csv')
# archived experiment11 medium components, used only to cross-check base(X)
EXP11_FROZEN_COMPONENTS = EXP11 / 'interventions_v2/A_20260528/frozen_components.csv'
# frozen confirmation set (2436 structures, 2048 overall / 200 relative groups)
CONFIRMATION = EXP11 / 'confirmation'

PREFIXES = ('3_', '4_', 'reduce_input_to_hidden', 'reduce_hidden_to_energy',
            'rescale_atomic_energy')
TRAINABLE_PARAMETERS = 301300
ROUTES = ('U', 'A')
SEEDS = (20260528, 20260529, 20260530)
BATCH_TRAIN = 16
BATCH_EVAL = 32
LR = 3e-4
WEIGHT_DECAY = 1e-5
HUBER_BETA = 0.05
MAX_EPOCHS = 120
CALIBRATION_ALPHA = 1e-6
BUDGETS = (20, 40, 60, 80, 120)


def load_element_reference(path=CALIBRATION_CACHE):
    """Return (mu[118], bias) for the frozen element_reference calibration."""
    coefficients = pd.read_csv(path)
    values = (coefficients[coefficients.calibration.eq('element_reference')]
              .set_index('coefficient').value)
    mu = np.array([values.get(f'mu_Z{number}', 0.) for number in range(1, 119)], dtype=float)
    bias = float(values.get('bias', 0.))
    return mu, bias


def atomic_fractions(meta: pd.DataFrame) -> np.ndarray:
    """118-dim atomic fractions in Z order, exactly as run_decomposition does."""
    fractions = np.array([[Composition(formula).get_atomic_fraction(Element.from_Z(number))
                           for number in range(1, 119)] for formula in meta.formula])
    counts = fractions * meta.natoms.to_numpy()[:, None]
    if not np.allclose(counts, np.rint(counts), atol=1e-6, rtol=0):
        raise ValueError('Noninteger atom counts')
    counts = np.rint(counts)
    return counts / counts.sum(axis=1)[:, None]


def alignment_offset(fractions: np.ndarray, mu: np.ndarray, bias: float) -> np.ndarray:
    """d(z) = -(x @ mu + bias) = E_base - f_theta0/N, the frozen aligned offset."""
    return -(fractions @ mu + bias)


def per_atom(model, batch):
    """SevenNet total energy per atom. Divides by natoms exactly once."""
    import sevenn._keys as keys
    prediction = model(batch.clone())
    return (prediction[keys.PRED_TOTAL_ENERGY].reshape(-1)
            / prediction[keys.NUM_ATOMS].reshape(-1))


def route_prediction(route, model, batch, offset_tensor):
    """Native prediction that enters the training loss."""
    energy = per_atom(model, batch)
    if route == 'U':
        return energy
    if route == 'A':
        return energy + offset_tensor[batch.manifest_row.long()]
    raise ValueError(f'Unknown route {route!r}')


def centered(values, groups):
    frame = pd.DataFrame({'v': np.asarray(values, float), 'g': np.asarray(groups)})
    return (frame.v - frame.groupby('g').v.transform('mean')).to_numpy()


def group_labels(meta: pd.DataFrame) -> np.ndarray:
    """Exact-composition group = integer element-count vector."""
    fractions = atomic_fractions(meta)
    counts = np.rint(fractions * meta.natoms.to_numpy()[:, None]).astype(int)
    return np.array(['|'.join(map(str, row)) for row in counts])


def ec_mae(truth, prediction, groups):
    groups = np.asarray(groups)
    sizes = pd.Series(groups).value_counts()
    mask = pd.Series(groups).map(sizes).to_numpy() >= 2
    error = centered(np.asarray(prediction) - np.asarray(truth), groups)
    return (float(1000 * np.mean(np.abs(error[mask]))) if mask.any() else float('nan'),
            int(mask.sum()), int((sizes >= 2).sum()))


def fit_post_calibration(fractions_train, residual_train, alpha=CALIBRATION_ALPHA):
    """Ridge on 118 atomic fractions + unpenalized intercept, fit on train only."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(fractions_train, residual_train)
    return model


def apply_post_calibration(model, fractions):
    return model.predict(fractions)


def load_graphs(splits=('train', 'val', 'test')):
    return {split: torch.load(GRAPH_CACHE / f'{split}_graphs.pt', map_location='cpu',
                              weights_only=False) for split in splits}


def load_meta():
    meta = pd.read_csv(GRAPH_CACHE / 'graph_metadata.csv').sort_values('row_index').reset_index(drop=True)
    if not np.array_equal(meta.row_index.to_numpy(), np.arange(len(meta))):
        raise ValueError('graph metadata row_index is not contiguous')
    return meta


def build_model(device):
    import sys
    sys.path.insert(0, str(ENERGY))
    from rclfa_v3.scripts.run_sevennet_upper_bound_279 import load_sevennet_model
    model = load_sevennet_model(remove_force_output=True).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != TRAINABLE_PARAMETERS:
        raise ValueError(f'Unexpected trainable count: {trainable}')
    return model


def trainable_state_hash(model):
    """Stable hash of the trainable parameters, for the U/A initialization check."""
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().astype(np.float64).tobytes())
    return digest.hexdigest()


def run_dir(route, seed):
    return OUT / 'runs' / f'{route}_{seed}'
