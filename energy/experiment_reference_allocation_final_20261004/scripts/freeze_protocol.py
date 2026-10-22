from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "PROTOCOL.md"
    manifest = root / "manifests" / "protocol.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists() and destination.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite a different frozen protocol: {destination}")
    destination.write_bytes(payload)
    record = {
        "source_basename": source.name,
        "source_sha256": digest,
        "frozen_path": str(destination),
        "frozen_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "byte_count": len(payload),
    }
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    if record["source_sha256"] != record["frozen_sha256"]:
        raise AssertionError("Frozen protocol differs from source")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
