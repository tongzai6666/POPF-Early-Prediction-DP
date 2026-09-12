from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from predict import predict_dataframe
from utils.metrics import binary_metrics


def main(args):
    df = pd.read_csv(args.data_csv)
    pred = predict_dataframe(df, args.bundle_dir, args.device)
    merged = df.merge(pred, on="patient_id", how="inner")
    leakage = binary_metrics(
        merged[args.leakage_label].astype(int).to_numpy(),
        merged["pancreatic_leakage_probability"].to_numpy(),
        float(merged["leakage_threshold"].iloc[0]),
    )
    cr = binary_metrics(
        merged[args.cr_label].astype(int).to_numpy(),
        merged["cr_popf_probability"].to_numpy(),
        float(merged["cr_popf_threshold"].iloc[0]),
    )
    result = {"broader_pancreatic_leakage": leakage, "cr_popf": cr}
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--data_csv", required=True)
    p.add_argument("--leakage_label", default="label_leakage")
    p.add_argument("--cr_label", default="label_cr_popf")
    p.add_argument("--output_json", default=None)
    p.add_argument("--device", default=None)
    main(p.parse_args())
