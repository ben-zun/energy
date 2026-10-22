"""Step 2 -- train EqV3 with the last full computation module + energy head.

The archived trainer only offers --tuning-type head|lora|all, and its argparse
choices reject anything else, so this wrapper does NOT edit the archived library:

  * it strips two custom flags from sys.argv
        --unfreeze-last-blocks N     (default 1)
  * rewrites --tuning-type lastblock  ->  --tuning-type head   (a legal choice,
    which keeps the frozen backbone and unfreezes the head)
  * installs a monkeypatch of configure_eqv3_tuning_mode that consults the env
    vars and, for the lastblock mode, additionally unfreezes
        model.blocks[-N:]        the trailing transformer blocks
        model.norm               the final norm that feeds the energy head
        model.energy_block       the energy output head
  * then calls the archived trainer's main() unchanged.

Architecture (from the run logs): blocks 0-6 are TransBlockV3, then `norm`
(EquivariantMergeLayerNorm), then `energy_block` (ScalarFeedForwardNetwork).
"Last full computation module" is therefore blocks[-1]; `norm` is included because
it is part of the same readout path.  Everything else stays frozen.

Every run records its trainable parameter count to the summary so the two tuning
modes can be compared without pretending their budgets match.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path('/home/yxzhang/energy/rclfa_v3/scripts')
sys.path.insert(0, str(SCRIPTS))


def _extract(argv, flag, default=None):
    if flag in argv:
        index = argv.index(flag)
        value = argv[index + 1]
        del argv[index:index + 2]
        return value
    return default


def main():
    argv = sys.argv[:]
    n_blocks = int(_extract(argv, '--unfreeze-last-blocks', '1'))
    include_norm = _extract(argv, '--unfreeze-final-norm', 'true').lower() != 'false'
    if '--tuning-type' in argv:
        requested = argv[argv.index('--tuning-type') + 1]
        if requested == 'lastblock':
            argv[argv.index('--tuning-type') + 1] = 'head'   # legal placeholder
    sys.argv = argv

    os.environ['EQV3_UNFREEZE_LAST_BLOCKS'] = str(n_blocks)
    os.environ['EQV3_UNFREEZE_FINAL_NORM'] = '1' if include_norm else '0'

    from rclfa_v3 import eqv3_native

    original = eqv3_native.configure_eqv3_tuning_mode

    def configure_eqv3_tuning_mode(model, *, tuning_type, lora_rank, lora_alpha):
        requested = os.environ.get('EQV3_UNFREEZE_LAST_BLOCKS')
        if requested is None or int(requested) <= 0:
            return original(model, tuning_type=tuning_type, lora_rank=lora_rank,
                            lora_alpha=lora_alpha)
        n = int(requested)
        energy_head = eqv3_native.require_eqv3_energy_model(model)
        blocks = list(model.blocks)
        if n > len(blocks):
            raise ValueError(f'asked to unfreeze {n} blocks but the model has {len(blocks)}')
        # freeze/unfreeze return None -- they act by side effect, so build the
        # list of unfrozen module names ourselves
        eqv3_native.freeze_module(model)
        targets = [(f'blocks.{len(blocks) - 1 - i}', block)
                   for i, block in enumerate(reversed(blocks)) if i < n]
        include_norm = os.environ.get('EQV3_UNFREEZE_FINAL_NORM') == '1'
        final_norm = getattr(model, 'norm', None)
        if include_norm and final_norm is not None:
            targets.append(('norm', final_norm))
        targets.append(('energy_block', energy_head))
        for _, module in targets:
            eqv3_native.unfreeze_module(module)
        names = [name for name, _ in targets]
        print(f'[lastblock] unfroze {len(names)} modules: last {n} block(s)'
              f'{", final norm" if include_norm else ""}, energy head -> {names}', flush=True)
        return names

    eqv3_native.configure_eqv3_tuning_mode = configure_eqv3_tuning_mode

    import run_eqv3_native_upper_bound_smoke_20260725 as trainer
    # the trainer imported the symbol at module load, so rebind it there too
    trainer.eqv3_native = eqv3_native
    if hasattr(trainer, 'configure_eqv3_tuning_mode'):
        trainer.configure_eqv3_tuning_mode = configure_eqv3_tuning_mode
    trainer.main()


if __name__ == '__main__':
    main()
