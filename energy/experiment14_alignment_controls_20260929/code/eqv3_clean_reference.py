"""Build a composition-only reference column for the CLEAN EqV3 comparison.

Verification result that motivates this file
  pred_element_reference  = f_theta0(X)/N + delta(z)
  within exact-composition groups the difference
      pred_element_reference - model_energy_per_atom
  has spread 1.8e-15 (machine zero) in 0/335 groups, so delta(z) is a pure
  composition function.

The existing residual runs used pred_element_reference as the reference, so their
prediction is f_theta(X) + f_theta0(X)/N + delta(z): it differs from the direct
route by TWO factors (composition offset AND a frozen-model residual baseline).

This script emits a reference CSV whose column `pred_element_offset` is delta(z)
alone, so
    --target-mode residual --reference-column pred_element_offset
predicts  f_theta(X) + delta(z),
which is the exact single-factor analogue of SevenNet route A.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

SOURCE = Path('/home/yxzhang/energy/runs/rclfa_v3/eqv3_matched_benchmark_20260725'
              '/calibration/calibrated_predictions.csv')
OUT = Path('/home/yxzhang/energy/experiment14_alignment_controls_20260929/eqv3_clean_reference.csv')
CHECK_GROUPS = Path('/home/yxzhang/energy/runs/rclfa_v3/fegx_component_function_deep_study_292_298'
                    '_revision/fair_finetune_293/graph_cache/medium/graph_metadata.csv')


def main():
    frame = pd.read_csv(SOURCE)
    frame['pred_element_offset'] = (frame.pred_element_reference
                                    - frame.model_energy_per_atom)
    # sanity: the offset must be a pure composition function
    def counts(row):
        c = Composition(row.formula)
        return '|'.join(str(int(round(c.get_atomic_fraction(Element.from_Z(z)) * row.natoms)))
                        for z in range(1, 119))
    frame['g'] = frame.apply(counts, axis=1)
    sizes = frame.g.value_counts()
    multi = sizes[sizes >= 2].index
    spread = frame[frame.g.isin(multi)].groupby('g').pred_element_offset.agg(
        lambda s: s.max() - s.min())
    print(f'rows={len(frame)} multi-groups={len(spread)} '
          f'max_within_group_spread={spread.max():.3e} '
          f'groups_with_spread={int((spread > 1e-9).sum())}')
    if spread.max() > 1e-9:
        raise AssertionError('pred_element_offset is not composition-only')
    keep = ['split', 'id', 'formula', 'cif_path', 'natoms', 'nelements', 'y_true',
            'pred_element_offset', 'pred_composition_only_linear']
    out = frame[keep]
    missing = set(out.id) ^ set(pd.read_csv(SOURCE).id)
    if missing:
        raise AssertionError('id set changed')
    out.to_csv(OUT, index=False)
    print('wrote', OUT, out.shape)
    print(out.groupby('split').size().to_string())


if __name__ == '__main__':
    main()
