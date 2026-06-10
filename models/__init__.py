"""
POPF Early Prediction Models
============================
Core model components for the dual-task deep learning framework.

Modules:
    bc_bert_encoder: Bio-Clinical BERT semantic encoder
    temporal_embedding: Time-aware embedding layer
    gated_transformer: Context-aware gated transformer
    attention_mask: Missing value handling via attention masking
    dual_task_classifier: Primary MLP + Hierarchical Gated MLP for dual-task prediction
"""

from .bc_bert_encoder import BC_BERT_Encoder
from .temporal_embedding import TemporalEmbedding, PositionalEncoding, ValueEmbedding
from .gated_transformer import (
    GatedMultiHeadAttention, 
    GatedFeedForward,
    ContextAwareGatedTransformer,
    TemporalPooling
)
from .attention_mask import (
    VariableAttentionMask,
    MissingValueHandler,
    CenterAwareMask
)
from .dual_task_classifier import (
    PrimaryMLPClassifier,
    HierarchicalGatedMLP,
    DualTaskFramework
)

__all__ = [
    'BC_BERT_Encoder',
    'TemporalEmbedding',
    'PositionalEncoding',
    'ValueEmbedding',
    'GatedMultiHeadAttention',
    'GatedFeedForward',
    'ContextAwareGatedTransformer',
    'TemporalPooling',
    'VariableAttentionMask',
    'MissingValueHandler',
    'CenterAwareMask',
    'PrimaryMLPClassifier',
    'HierarchicalGatedMLP',
    'DualTaskFramework'
]
