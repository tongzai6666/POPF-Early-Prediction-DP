"""
Prediction/Inference Script
===========================
Script for model inference on new patient data.
Supports single patient prediction and batch processing.

Usage:
    python predict.py --model_path ./checkpoints/best_model.pth --data_path ./data/test.csv --output_path ./predictions.csv
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Union
import csv

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.bc_bert_encoder import BC_BERT_Encoder
from models.temporal_embedding import TemporalEmbedding
from models.gated_transformer import ContextAwareGatedTransformer
from models.attention_mask import MissingValueHandler
from models.dual_task_classifier import (
    PrimaryMLPClassifier, 
    HierarchicalGatedMLP, 
    DualTaskFramework
)
from data.data_loader import POPFDataset, collate_fn
from data.preprocessing import DataPreprocessor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device = None
) -> DualTaskFramework:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Computation device

    Returns:
        Loaded DualTaskFramework model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {checkpoint_path}")

    # Create model architecture (same as training)
    d_model = 768

    bc_bert_encoder = BC_BERT_Encoder(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        hidden_size=d_model,
        freeze_layers=6,
        dropout=0.1
    )

    temporal_embedding = TemporalEmbedding(
        num_time_stages=3,
        embedding_dim=d_model,
        dropout=0.1
    )

    gated_transformer = ContextAwareGatedTransformer(
        d_model=d_model,
        num_heads=8,
        num_layers=4,
        d_ff=3072,
        dropout=0.1,
        use_look_ahead=False
    )

    missing_handler = MissingValueHandler(
        embedding_dim=d_model,
        missing_token_init="normal"
    )

    primary_classifier = PrimaryMLPClassifier(
        input_dim=d_model,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        use_batch_norm=True
    )

    severity_classifier = HierarchicalGatedMLP(
        input_dim=d_model,
        primary_feature_dim=128,
        hidden_dims=[256, 128],
        dropout=0.3,
        use_batch_norm=True,
        gate_threshold=0.5
    )

    model = DualTaskFramework(
        bc_bert_encoder=bc_bert_encoder,
        temporal_embedding=temporal_embedding,
        gated_transformer=gated_transformer,
        primary_classifier=primary_classifier,
        severity_classifier=severity_classifier,
        missing_handler=missing_handler,
        primary_loss_weight=1.0,
        severity_loss_weight=1.0,
        use_severity_gate=True
    ).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    logger.info(f"Validation AUC: {checkpoint.get('val_primary_auc', 'unknown')}")

    return model


def predict_single_patient(
    model: DualTaskFramework,
    patient_data: Dict,
    device: torch.device,
    threshold: float = 0.4875
) -> Dict:
    """
    Predict POPF risk for a single patient.

    Args:
        model: Trained DualTaskFramework model
        patient_data: Dictionary with keys:
            - 'timestamps': list of str
            - 'variable_names': list of str
            - 'values': torch.Tensor or list of float
            - 'attention_mask': torch.Tensor or list of float
            - 'center_id': int (optional)
        device: Computation device
        threshold: Classification threshold (Youden index optimal)

    Returns:
        Prediction dictionary with risk probabilities and classification
    """
    # Prepare batch (single patient)
    if isinstance(patient_data['values'], list):
        patient_data['values'] = torch.tensor(patient_data['values'], dtype=torch.float32)
    if isinstance(patient_data['attention_mask'], list):
        patient_data['attention_mask'] = torch.tensor(patient_data['attention_mask'], dtype=torch.float32)

    # Add batch dimension
    batch_data = {
        'timestamps': [patient_data['timestamps']],
        'variable_names': [patient_data['variable_names']],
        'values': patient_data['values'].unsqueeze(0).to(device),
        'attention_mask': patient_data['attention_mask'].unsqueeze(0).to(device),
        'center_ids': torch.tensor([patient_data.get('center_id', 1)], dtype=torch.long).to(device)
    }

    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model.predict(batch_data, primary_threshold=threshold)

    return {
        'popf_probability': float(predictions['popf_probability'][0]),
        'severity_probability': float(predictions['severity_probability'][0]),
        'popf_risk': predictions['popf_risk'][0],
        'severity_grade': predictions['severity_grade'][0],
        'threshold_used': threshold
    }


def predict_batch(
    model: DualTaskFramework,
    data_loader,
    device: torch.device,
    threshold: float = 0.4875
) -> List[Dict]:
    """
    Predict POPF risk for a batch of patients.

    Args:
        model: Trained DualTaskFramework model
        data_loader: PyTorch DataLoader
        device: Computation device
        threshold: Classification threshold

    Returns:
        List of prediction dictionaries
    """
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Move tensors to device
            batch_data = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device)
                else:
                    batch_data[key] = value

            # Predict
            predictions = model.predict(batch_data, primary_threshold=threshold)

            # Collect results
            batch_size = len(predictions['popf_risk'])
            for i in range(batch_size):
                pred = {
                    'patient_id': batch['patient_ids'][i] if 'patient_ids' in batch else f"patient_{i}",
                    'popf_probability': float(predictions['popf_probability'][i]),
                    'severity_probability': float(predictions['severity_probability'][i]),
                    'popf_risk': predictions['popf_risk'][i],
                    'severity_grade': predictions['severity_grade'][i],
                    'threshold_used': threshold
                }
                all_predictions.append(pred)

    return all_predictions


def save_predictions(
    predictions: List[Dict],
    output_path: str
):
    """
    Save predictions to CSV file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    # Print summary statistics
    high_risk_count = sum(1 for p in predictions if p['popf_risk'] == 'High')
    cr_popf_count = sum(1 for p in predictions if p['severity_grade'] == 'CR-POPF')

    logger.info(f"Total patients: {len(predictions)}")
    logger.info(f"High-risk patients: {high_risk_count} ({high_risk_count/len(predictions)*100:.1f}%)")
    logger.info(f"Predicted CR-POPF: {cr_popf_count} ({cr_popf_count/len(predictions)*100:.1f}%)")


def main(
    model_path: str,
    data_path: str,
    output_path: str,
    threshold: float = 0.4875,
    device: str = None
):
    """
    Main prediction function.

    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to input data (CSV format)
        output_path: Path to save predictions (CSV format)
        threshold: Classification threshold
        device: Device string ("cuda" or "cpu")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)

    # Load data
    logger.info(f"Loading data from {data_path}")
    dataset = POPFDataset(data_path=data_path)

    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Predict
    logger.info(f"Running predictions with threshold={threshold}")
    predictions = predict_batch(model, data_loader, device, threshold)

    # Save results
    save_predictions(predictions, output_path)

    # Print sample predictions
    logger.info("\nSample predictions:")
    for i, pred in enumerate(predictions[:5]):
        logger.info(
            f"  {pred['patient_id']}: "
            f"POPF_prob={pred['popf_probability']:.4f}, "
            f"Risk={pred['popf_risk']}, "
            f"Grade={pred['severity_grade']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict POPF risk using trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--output_path", type=str, default="./predictions.csv", help="Path to output CSV")
    parser.add_argument("--threshold", type=float, default=0.4875, help="Classification threshold (Youden index)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        threshold=args.threshold,
        device=args.device
    )
