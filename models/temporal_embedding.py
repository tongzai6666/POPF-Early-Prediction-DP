from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    def __init__(self, num_time_stages: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_time_stages, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stage_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(stage_ids))


class ValueEmbedding(nn.Module):
    """Embed standardized numeric values and training-vocabulary categorical values."""

    def __init__(self, d_model: int, num_categories: int):
        super().__init__()
        self.numeric = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.categorical = nn.Embedding(num_categories, d_model, padding_idx=0)

    def forward(
        self,
        numeric_values: torch.Tensor,
        category_ids: torch.Tensor,
        feature_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        # numeric_values/category_ids: [B, F]; feature_type_ids: [B, F] or [F]
        numeric_emb = self.numeric(numeric_values.unsqueeze(-1))
        categorical_emb = self.categorical(category_ids)
        if feature_type_ids.dim() == 1:
            feature_type_ids = feature_type_ids.unsqueeze(0).expand(numeric_values.size(0), -1)
        is_cat = feature_type_ids.bool().unsqueeze(-1)
        return torch.where(is_cat, categorical_emb, numeric_emb)


class ContextFusion(nn.Module):
    """Fuse semantic, value, and temporal embeddings with a learned value gate."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic: torch.Tensor, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        gate = self.value_gate(torch.cat([semantic, value, time], dim=-1))
        x = semantic + gate * value + time
        return self.dropout(self.norm(x))
