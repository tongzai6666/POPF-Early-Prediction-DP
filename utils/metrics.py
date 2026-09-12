from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.metrics import recall_score


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    j = tpr - fpr
    return float(thresholds[int(np.nanargmax(j))])


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }
