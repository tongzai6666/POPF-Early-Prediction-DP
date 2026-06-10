"""
Training Script
===============
Main training script for the dual-task POPF prediction framework.
Implements training loop with early stopping, learning rate scheduling,
and model checkpointing.

Paper Section 2.5: "Internal validation was performed on the split single-center dataset."
"""

import os
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.bc_bert_encoder import BC_BERT_Encoder
from models.temporal_embedding import TemporalEmbedding, ValueEmbedding
from models.gated_transformer import ContextAwareGatedTransformer
from models.attention_mask import MissingValueHandler
from models.dual_task_classifier import (
    PrimaryMLPClassifier, 
    HierarchicalGatedMLP, 
    DualTaskFramework
)
from data.data_loader import POPFDataset, collate_fn, MulticenterDataLoader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(
    d_model: int = 768,
    num_heads: int = 8,
    num_transformer_layers: int = 4,
    d_ff: int = 3072,
    dropout: float = 0.1,
    primary_loss_weight: float = 1.0,
    severity_loss_weight: float = 1.0,
    device: torch.device = None
) -> DualTaskFramework:
    """
    Create and initialize the complete dual-task framework.

    Args:
        d_model: Model dimension (must match BERT hidden_size)
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        primary_loss_weight: Weight for primary task loss
        severity_loss_weight: Weight for severity task loss
        device: Computation device

    Returns:
        Initialized DualTaskFramework model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    bc_bert_encoder = BC_BERT_Encoder(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        hidden_size=d_model,
        freeze_layers=6,
        dropout=dropout
    )

    temporal_embedding = TemporalEmbedding(
        num_time_stages=3,
        embedding_dim=d_model,
        dropout=dropout
    )

    gated_transformer = ContextAwareGatedTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_transformer_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_look_ahead=False  # Non-causal for perioperative data
    )

    missing_handler = MissingValueHandler(
        embedding_dim=d_model,
        missing_token_init="normal"
    )

    primary_classifier = PrimaryMLPClassifier(
        input_dim=d_model,
        hidden_dims=[512, 256, 128],
        dropout=dropout * 3,  # Higher dropout for classifier
        use_batch_norm=True
    )

    severity_classifier = HierarchicalGatedMLP(
        input_dim=d_model,
        primary_feature_dim=128,
        hidden_dims=[256, 128],
        dropout=dropout * 3,
        use_batch_norm=True,
        gate_threshold=0.5
    )

    # Assemble framework
    model = DualTaskFramework(
        bc_bert_encoder=bc_bert_encoder,
        temporal_embedding=temporal_embedding,
        gated_transformer=gated_transformer,
        primary_classifier=primary_classifier,
        severity_classifier=severity_classifier,
        missing_handler=missing_handler,
        primary_loss_weight=primary_loss_weight,
        severity_loss_weight=severity_loss_weight,
        use_severity_gate=True
    ).to(device)

    return model


def train_epoch(
    model: DualTaskFramework,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: DualTaskFramework model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Computation device
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    model.train()

    total_loss = 0.0
    total_primary_loss = 0.0
    total_severity_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        # Move batch to device (handle string lists separately)
        batch_tensors = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_tensors[key] = value.to(device)
            else:
                batch_tensors[key] = value

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_tensors)

        # Compute loss
        labels = {
            'label_popf': batch_tensors['label_popf'],
            'label_severity': batch_tensors['label_severity']
        }
        losses = model.compute_loss(outputs, labels, use_severity_mask=True)

        loss = losses['total_loss']

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_primary_loss += losses['primary_loss'].item()
        total_severity_loss += losses['severity_loss'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'primary': f"{losses['primary_loss'].item():.4f}",
            'severity': f"{losses['severity_loss'].item():.4f}"
        })

    avg_loss = total_loss / num_batches
    avg_primary = total_primary_loss / num_batches
    avg_severity = total_severity_loss / num_batches

    return {
        'loss': avg_loss,
        'primary_loss': avg_primary,
        'severity_loss': avg_severity
    }


def validate_epoch(
    model: DualTaskFramework,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: DualTaskFramework model
        val_loader: Validation data loader
        device: Computation device
        epoch: Current epoch number

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_primary_loss = 0.0
    total_severity_loss = 0.0
    num_batches = 0

    all_primary_probs = []
    all_primary_labels = []
    all_severity_probs = []
    all_severity_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            batch_tensors = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_tensors[key] = value.to(device)
                else:
                    batch_tensors[key] = value

            outputs = model(batch_tensors)

            labels = {
                'label_popf': batch_tensors['label_popf'],
                'label_severity': batch_tensors['label_severity']
            }
            losses = model.compute_loss(outputs, labels, use_severity_mask=True)

            total_loss += losses['total_loss'].item()
            total_primary_loss += losses['primary_loss'].item()
            total_severity_loss += losses['severity_loss'].item()
            num_batches += 1

            # Collect predictions for metrics
            all_primary_probs.append(outputs['primary_prob'].cpu())
            all_primary_labels.append(batch_tensors['label_popf'].cpu())

            # Severity: only for positive cases
            severity_mask = (batch_tensors['label_popf'] == 1).squeeze(-1)
            if severity_mask.any():
                all_severity_probs.append(outputs['severity_prob'][severity_mask].cpu())
                all_severity_labels.append(batch_tensors['label_severity'][severity_mask].cpu())

    avg_loss = total_loss / num_batches
    avg_primary = total_primary_loss / num_batches
    avg_severity = total_severity_loss / num_batches

    # Compute AUC for primary task
    from sklearn.metrics import roc_auc_score

    primary_probs = torch.cat(all_primary_probs).numpy().flatten()
    primary_labels = torch.cat(all_primary_labels).numpy().flatten()

    try:
        primary_auc = roc_auc_score(primary_labels, primary_probs)
    except:
        primary_auc = 0.5

    # Compute AUC for severity task (if enough samples)
    severity_auc = 0.5
    if len(all_severity_probs) > 0 and len(all_severity_probs[0]) > 0:
        severity_probs = torch.cat(all_severity_probs).numpy().flatten()
        severity_labels = torch.cat(all_severity_labels).numpy().flatten()
        if len(np.unique(severity_labels)) > 1:
            try:
                severity_auc = roc_auc_score(severity_labels, severity_probs)
            except:
                pass

    return {
        'loss': avg_loss,
        'primary_loss': avg_primary,
        'severity_loss': avg_severity,
        'primary_auc': primary_auc,
        'severity_auc': severity_auc
    }


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 15,
    device: str = None,
    seed: int = 42
):
    """
    Main training function.

    Args:
        data_dir: Directory containing center data files
        output_dir: Directory to save model checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
        device: Device string ("cuda" or "cpu")
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Create data loaders
    data_manager = MulticenterDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        train_val_split=0.7,
        random_seed=seed
    )

    loaders = data_manager.get_all_loaders()
    train_loader = loaders['train']
    val_loader = loaders['val']

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = create_model(device=device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer - use different learning rates for BERT and other components
    bert_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': lr * 0.1},  # Lower LR for pre-trained BERT
        {'params': other_params, 'lr': lr}
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_primary_auc': [],
        'val_severity_auc': []
    }

    logger.info("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Log metrics
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Primary AUC: {val_metrics['primary_auc']:.4f} | "
            f"Val Severity AUC: {val_metrics['severity_auc']:.4f}"
        )

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_primary_auc'].append(val_metrics['primary_auc'])
        history['val_severity_auc'].append(val_metrics['severity_auc'])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0

            checkpoint_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_primary_auc': val_metrics['primary_auc'],
                'val_severity_auc': val_metrics['severity_auc']
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Save final training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train POPF prediction model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed
    )
