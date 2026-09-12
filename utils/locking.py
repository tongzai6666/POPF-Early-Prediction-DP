from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bundle_dir: str | Path, files: Iterable[str]) -> None:
    bundle = Path(bundle_dir)
    manifest = {name: sha256(bundle / name) for name in files}
    with open(bundle / "release_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def verify_manifest(bundle_dir: str | Path) -> bool:
    bundle = Path(bundle_dir)
    with open(bundle / "release_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return all(sha256(bundle / name) == expected for name, expected in manifest.items())
