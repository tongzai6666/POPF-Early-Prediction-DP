"""Audit utility to derive Youden thresholds from training-cohort predictions.

For the reported manuscript release, the locked thresholds remain 0.40 and 0.31.
This utility exists to document/reproduce the selection rule on the training cohort.
"""
from __future__ import annotations
import argparse
import pandas as pd
from predict import predict_dataframe
from utils.metrics import youden_threshold

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--leakage_label", default="label_leakage")
    p.add_argument("--cr_label", default="label_cr_popf")
    args = p.parse_args()
    df = pd.read_csv(args.train_csv)
    pr = predict_dataframe(df, args.bundle_dir)
    x = df.merge(pr, on="patient_id")
    print("Youden leakage threshold:", youden_threshold(x[args.leakage_label].values, x["pancreatic_leakage_probability"].values))
    print("Youden CR-POPF threshold:", youden_threshold(x[args.cr_label].values, x["cr_popf_probability"].values))
