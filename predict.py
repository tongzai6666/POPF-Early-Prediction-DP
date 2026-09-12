"""Inference from the locked public release bundle.

The published thresholds are 0.40 for the broader pancreatic leakage probability
and 0.31 for the final CR-POPF probability. No preprocessing parameter, model
weight, or threshold is re-estimated at inference time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data.data_loader import POPFDataset, make_loader
from utils.model_io import load_locked_bundle


def move_batch(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_dataframe(df: pd.DataFrame, bundle_dir: str, device_str: str | None = None) -> pd.DataFrame:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, pre, cfg, thresholds, _ = load_locked_bundle(bundle_dir, device)
    labels = cfg["labels"]
    dataset = POPFDataset(df, pre, labels["patient_id"], labels["leakage"], labels["cr_popf"])
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), False, int(cfg["training"]["num_workers"]))

    rows = []
    for batch in loader:
        patient_ids = batch["patient_id"]
        batch = move_batch(batch, device)
        pred = model.predict(
            batch,
            leakage_threshold=float(thresholds["leakage_probability"]),
            cr_threshold=float(thresholds["cr_popf_probability"]),
        )
        for i, pid in enumerate(patient_ids):
            rows.append({
                "patient_id": pid,
                "pancreatic_leakage_probability": float(pred["leakage_probability"][i].cpu()),
                "cr_popf_probability": float(pred["cr_popf_probability"][i].cpu()),
                "joint_cr_popf_probability": float(pred["joint_cr_popf_probability"][i].cpu()),
                "predicted_grade": pred["grade"][i],
                "leakage_threshold": float(thresholds["leakage_probability"]),
                "cr_popf_threshold": float(thresholds["cr_popf_probability"]),
            })
    return pd.DataFrame(rows)


def main(args):
    df = pd.read_csv(args.data_csv)
    out = predict_dataframe(df, args.bundle_dir, args.device)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.head())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--data_csv", required=True)
    p.add_argument("--output_csv", default="./predictions.csv")
    p.add_argument("--device", default=None)
    main(p.parse_args())
