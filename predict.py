"""Inference from the locked public release bundle.

The published thresholds are:

- 0.40 for the broader pancreatic leakage probability
- 0.31 for the final population-level CR-POPF probability

The model provides three probability outputs:

1. pancreatic_leakage_probability
   = P(pancreatic leakage)

2. conditional_cr_popf_probability
   = P(CR-POPF | pancreatic leakage)

3. cr_popf_probability
   = P(CR-POPF)
   = P(pancreatic leakage)
     × P(CR-POPF | pancreatic leakage)

The published thresholds remain locked at 0.40 and 0.31.
No preprocessing parameter, model weight, class weight,
or decision threshold is re-estimated at inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data.data_loader import POPFDataset, make_loader
from utils.model_io import load_locked_bundle


def move_batch(batch, device):
    """
    Move tensor components of a batch to the selected device.
    Non-tensor objects, such as patient identifiers, remain unchanged.
    """
    return {
        k: (
            v.to(device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


@torch.no_grad()
def predict_dataframe(
    df: pd.DataFrame,
    bundle_dir: str,
    device_str: str | None = None,
) -> pd.DataFrame:
    """
    Generate patient-level predictions using the locked release bundle.

    Probability definitions
    -----------------------

    pancreatic_leakage_probability
        P(pancreatic leakage)

    conditional_cr_popf_probability
        P(CR-POPF | pancreatic leakage)

    cr_popf_probability
        Population-level CR-POPF probability:

        P(CR-POPF)
        =
        P(pancreatic leakage)
        ×
        P(CR-POPF | pancreatic leakage)

    joint_cr_popf_probability
        Backward-compatible alias of cr_popf_probability.

    Published decision thresholds
    -----------------------------

    Pancreatic leakage:
        0.40

    CR-POPF:
        0.31
    """

    # -------------------------------------------------------------
    # Select computation device
    # -------------------------------------------------------------
    device = torch.device(
        device_str
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    # -------------------------------------------------------------
    # Load locked model bundle
    #
    # This contains:
    # - model weights
    # - preprocessing state
    # - configuration
    # - locked thresholds
    # -------------------------------------------------------------
    model, pre, cfg, thresholds, _ = load_locked_bundle(
        bundle_dir,
        device,
    )

    labels = cfg["labels"]

    # -------------------------------------------------------------
    # Build dataset using the locked preprocessing state
    # -------------------------------------------------------------
    dataset = POPFDataset(
        df,
        pre,
        labels["patient_id"],
        labels["leakage"],
        labels["cr_popf"],
    )

    loader = make_loader(
        dataset,
        int(cfg["training"]["batch_size"]),
        False,
        int(cfg["training"]["num_workers"]),
    )

    rows = []

    # -------------------------------------------------------------
    # Perform inference
    # -------------------------------------------------------------
    for batch in loader:

        patient_ids = batch["patient_id"]

        batch = move_batch(
            batch,
            device,
        )

        pred = model.predict(
            batch,
            leakage_threshold=float(
                thresholds["leakage_probability"]
            ),
            cr_threshold=float(
                thresholds["cr_popf_probability"]
            ),
        )

        # ---------------------------------------------------------
        # Convert predictions into patient-level output
        # ---------------------------------------------------------
        for i, pid in enumerate(patient_ids):

            rows.append(
                {
                    # -------------------------------------------------
                    # Patient identifier
                    # -------------------------------------------------
                    "patient_id":
                        pid,

                    # -------------------------------------------------
                    # Stage 1 probability
                    #
                    # P(pancreatic leakage)
                    #
                    # Pancreatic leakage includes:
                    # biochemical leak + CR-POPF
                    # -------------------------------------------------
                    "pancreatic_leakage_probability":
                        float(
                            pred[
                                "leakage_probability"
                            ][i].cpu()
                        ),

                    # -------------------------------------------------
                    # Stage 2 conditional probability
                    #
                    # P(CR-POPF | pancreatic leakage)
                    # -------------------------------------------------
                    "conditional_cr_popf_probability":
                        float(
                            pred[
                                "conditional_cr_popf_probability"
                            ][i].cpu()
                        ),

                    # -------------------------------------------------
                    # Final population-level CR-POPF probability
                    #
                    # P(CR-POPF)
                    # =
                    # P(leakage)
                    # ×
                    # P(CR-POPF | leakage)
                    #
                    # This is the formal CR-POPF probability
                    # reported by the model.
                    # -------------------------------------------------
                    "cr_popf_probability":
                        float(
                            pred[
                                "cr_popf_probability"
                            ][i].cpu()
                        ),

                    # -------------------------------------------------
                    # Backward-compatible alias
                    #
                    # This value is identical to
                    # cr_popf_probability.
                    # -------------------------------------------------
                    "joint_cr_popf_probability":
                        float(
                            pred[
                                "joint_cr_popf_probability"
                            ][i].cpu()
                        ),

                    # -------------------------------------------------
                    # Final threshold-based category
                    # -------------------------------------------------
                    "predicted_grade":
                        pred[
                            "grade"
                        ][i],

                    # -------------------------------------------------
                    # Published locked threshold:
                    # pancreatic leakage = 0.40
                    # -------------------------------------------------
                    "leakage_threshold":
                        float(
                            thresholds[
                                "leakage_probability"
                            ]
                        ),

                    # -------------------------------------------------
                    # Published locked threshold:
                    # CR-POPF = 0.31
                    # -------------------------------------------------
                    "cr_popf_threshold":
                        float(
                            thresholds[
                                "cr_popf_probability"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main(args):

    # -------------------------------------------------------------
    # Read patient-level input data
    # -------------------------------------------------------------
    df = pd.read_csv(
        args.data_csv
    )

    # -------------------------------------------------------------
    # Generate predictions
    # -------------------------------------------------------------
    out = predict_dataframe(
        df,
        args.bundle_dir,
        args.device,
    )

    # -------------------------------------------------------------
    # Create output directory if necessary
    # -------------------------------------------------------------
    Path(
        args.output_csv
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------
    out.to_csv(
        args.output_csv,
        index=False,
    )

    # -------------------------------------------------------------
    # Display first rows
    # -------------------------------------------------------------
    print(
        out.head()
    )


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument(
        "--bundle_dir",
        required=True,
    )

    p.add_argument(
        "--data_csv",
        required=True,
    )

    p.add_argument(
        "--output_csv",
        default="./predictions.csv",
    )

    p.add_argument(
        "--device",
        default=None,
    )

    main(
        p.parse_args()
    )
