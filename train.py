"""Train and lock the temporal dual-task POPF model.

This script deliberately keeps the internal/external evaluation cohorts out of model
selection. A tuning split is created *within the training cohort* for early stopping.
The selected epoch is then used to refit the model on the full training cohort, after
which the published training-derived decision thresholds are locked in the release
bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.preprocessing import DataPreprocessor, derive_class_weights, load_feature_schema
from data.data_loader import POPFDataset, make_loader
from models.dual_task_classifier import build_model
from utils.reproducibility import set_seed
from utils.locking import write_manifest


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def run_epoch(model, loader, optimizer, device, gradient_clip_norm: float) -> Dict[str, float]:
    model.train()
    totals = {"total_loss": 0.0, "primary_loss": 0.0, "severity_loss": 0.0}
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        losses = model.compute_loss(out, batch)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        for k in totals:
            totals[k] += float(losses[k].detach().cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_loss(model, loader, device) -> Dict[str, float]:
    model.eval()
    totals = {"total_loss": 0.0, "primary_loss": 0.0, "severity_loss": 0.0}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        losses = model.compute_loss(out, batch)
        for k in totals:
            totals[k] += float(losses[k].cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def make_optimizer(model, cfg):
    base_lr = float(cfg["training"]["base_learning_rate"])
    bert_mult = float(cfg["training"]["bert_learning_rate_multiplier"])
    wd = float(cfg["training"]["weight_decay"])
    bert_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bert_params if "semantic_encoder.bert" in name else other_params).append(p)
    return torch.optim.AdamW(
        [
            {"params": bert_params, "lr": base_lr * bert_mult},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=wd,
    )


def three_class_strata(df: pd.DataFrame, leakage_col: str, cr_col: str) -> np.ndarray:
    leak = df[leakage_col].astype(int).to_numpy()
    cr = df[cr_col].astype(int).to_numpy()
    # 0 = no leakage, 1 = BL, 2 = CR-POPF
    return np.where(leak == 0, 0, np.where(cr == 1, 2, 1))


def select_epoch(train_df: pd.DataFrame, cfg, schema, device: torch.device, work_dir: Path) -> int:
    labels = cfg["labels"]
    strata = three_class_strata(train_df, labels["leakage"], labels["cr_popf"])
    tune_fraction = float(cfg["training"]["tuning_fraction"])
    subtrain, tune = train_test_split(
        train_df,
        test_size=tune_fraction,
        random_state=int(cfg["seed"]),
        stratify=strata,
    )

    pre = DataPreprocessor(schema).fit(subtrain)
    p_w, s_w = derive_class_weights(subtrain, labels["leakage"], labels["cr_popf"])
    model = build_model(pre.state.feature_descriptions, pre.num_categories, str(work_dir / "model_config.yaml"), p_w, s_w).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["training"]["scheduler_factor"]),
        patience=int(cfg["training"]["scheduler_patience"]),
    )

    train_loader = make_loader(
        POPFDataset(subtrain, pre, labels["patient_id"], labels["leakage"], labels["cr_popf"]),
        int(cfg["training"]["batch_size"]), True, int(cfg["training"]["num_workers"]),
    )
    tune_loader = make_loader(
        POPFDataset(tune, pre, labels["patient_id"], labels["leakage"], labels["cr_popf"]),
        int(cfg["training"]["batch_size"]), False, int(cfg["training"]["num_workers"]),
    )

    best_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    max_epochs = int(cfg["training"]["max_epochs"])
    patience = int(cfg["training"]["early_stopping_patience"])
    grad_clip = float(cfg["training"]["gradient_clip_norm"])

    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, grad_clip)
        va = validate_loss(model, tune_loader, device)
        scheduler.step(va["total_loss"])
        print(f"epoch={epoch:03d} train_loss={tr['total_loss']:.5f} tune_loss={va['total_loss']:.5f}")
        if va["total_loss"] < best_loss - 1e-7:
            best_loss = va["total_loss"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_epoch


def refit_full_training(train_df: pd.DataFrame, cfg, schema, device: torch.device, work_dir: Path, epochs: int):
    labels = cfg["labels"]
    pre = DataPreprocessor(schema).fit(train_df)
    p_w, s_w = derive_class_weights(train_df, labels["leakage"], labels["cr_popf"])
    model = build_model(pre.state.feature_descriptions, pre.num_categories, str(work_dir / "model_config.yaml"), p_w, s_w).to(device)
    optimizer = make_optimizer(model, cfg)
    train_loader = make_loader(
        POPFDataset(train_df, pre, labels["patient_id"], labels["leakage"], labels["cr_popf"]),
        int(cfg["training"]["batch_size"]), True, int(cfg["training"]["num_workers"]),
    )
    grad_clip = float(cfg["training"]["gradient_clip_norm"])
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, grad_clip)
        print(f"refit_epoch={epoch:03d}/{epochs:03d} loss={tr['total_loss']:.5f}")
    return model, pre, p_w, s_w


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "model_config.yaml")
    shutil.copy(args.schema, out / "feature_schema.yaml")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    schema = load_feature_schema(args.schema)
    if int(cfg["model"]["num_time_stages"]) != 4:
        raise ValueError("This implementation requires exactly four perioperative stages.")

    set_seed(int(cfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_df = pd.read_csv(args.train_csv)

    required = [cfg["labels"]["leakage"], cfg["labels"]["cr_popf"]]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required label columns: {missing}")

    # Phase 1: model selection is confined to the training cohort.
    best_epoch = select_epoch(train_df, cfg, schema, device, out)
    print(f"Selected epoch from training-only tuning split: {best_epoch}")

    # Phase 2: refit preprocessing and model on the complete training cohort.
    set_seed(int(cfg["seed"]))
    model, pre, primary_w, severity_w = refit_full_training(train_df, cfg, schema, device, out, best_epoch)
    pre.save(out / "preprocessor.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_categories": pre.num_categories,
            "primary_pos_weight": primary_w,
            "severity_pos_weight": severity_w,
            "selected_epoch": best_epoch,
        },
        out / "model.pt",
    )

    # The study thresholds are locked values derived from the training cohort.
    thresholds = {
        "leakage_probability": float(cfg["thresholds"]["leakage_probability"]),
        "cr_popf_probability": float(cfg["thresholds"]["cr_popf_probability"]),
        "provenance": cfg["thresholds"]["provenance"],
    }
    with open(out / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    training_metadata = {
        "selected_epoch": best_epoch,
        "seed": int(cfg["seed"]),
        "primary_pos_weight": primary_w,
        "severity_pos_weight": severity_w,
        "n_training": int(len(train_df)),
        "external_data_used_for_training_or_tuning": False,
    }
    with open(out / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    write_manifest(
        out,
        ["model.pt", "preprocessor.json", "thresholds.json", "model_config.yaml", "feature_schema.yaml", "training_metadata.json"],
    )
    print(f"Locked release bundle written to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="Training cohort CSV only (e.g., n=609 in the reported study).")
    p.add_argument("--output_dir", default="./release_bundle")
    p.add_argument("--config", default="./config/model_config.yaml")
    p.add_argument("--schema", default="./config/feature_schema.yaml")
    p.add_argument("--device", default=None)
    main(p.parse_args())
