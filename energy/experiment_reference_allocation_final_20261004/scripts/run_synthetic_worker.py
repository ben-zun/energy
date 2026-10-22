from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ASSIGNMENTS = {
    0: [
        ("zero_direct", 20260528),
        ("zero_direct", 20260529),
        ("zero_direct", 20260530),
    ],
    1: [
        ("reference_direct", 20260528),
        ("reference_direct", 20260529),
    ],
    2: [
        ("reference_direct", 20260530),
        ("reference_aligned", 20260528),
    ],
    3: [
        ("reference_aligned", 20260529),
        ("reference_aligned", 20260530),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=sorted(ASSIGNMENTS), required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--energy-root", type=Path, default=Path("/home/yxzhang/energy"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment_root.resolve(strict=True)
    energy = args.energy_root.resolve(strict=True)
    python = args.python.resolve(strict=True)
    training_script = experiment / "scripts/train_synthetic_sevennet.py"
    labels = experiment / "synthetic_reference_control/labels_v1/synthetic_labels.csv"
    if not training_script.is_file() or not labels.is_file():
        raise FileNotFoundError("training script or frozen synthetic labels are missing")
    commands: list[list[str]] = []
    for condition, seed in ASSIGNMENTS[args.worker]:
        commands.append(
            [
                str(python),
                str(training_script),
                "--condition", condition,
                "--seed", str(seed),
                "--energy-root", str(energy),
                "--experiment-root", str(experiment),
                "--labels", str(labels),
                "--device", f"cuda:{args.worker}",
                "--epochs", "120",
            ]
        )
    print(json.dumps({"worker": args.worker, "commands": commands}, indent=2), flush=True)
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment)
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    print(json.dumps({"worker": args.worker, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
