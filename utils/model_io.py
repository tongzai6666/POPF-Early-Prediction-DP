from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from data.preprocessing import DataPreprocessor, load_feature_schema
from models.dual_task_classifier import build_model
from utils.locking import verify_manifest


def load_locked_bundle(bundle_dir: str | Path, device: torch.device):
    bundle = Path(bundle_dir)
    if not verify_manifest(bundle):
        raise RuntimeError("Release manifest verification failed. Refusing to run an altered bundle.")
    schema = load_feature_schema(bundle / "feature_schema.yaml")
    pre = DataPreprocessor.load(bundle / "preprocessor.json", schema)
    with open(bundle / "model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(bundle / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    ckpt = torch.load(bundle / "model.pt", map_location=device)
    model = build_model(
        pre.state.feature_descriptions,
        int(ckpt["num_categories"]),
        str(bundle / "model_config.yaml"),
        float(ckpt["primary_pos_weight"]),
        float(ckpt["severity_pos_weight"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, pre, cfg, thresholds, schema
