"""
Temporal Embedding Module
=========================
Encodes temporal information (preoperative / intraoperative / postoperative_24h)
into learnable time embeddings that are added to variable semantic embeddings.

Paper Section 2.5: "All temporal EHR items are jointly encoded with their 
corresponding variable names and timestamps, enabling the model to delineate 
the dynamic evolution of key indicators over time."
"""

import torch
import torch.nn as nn
from typing import Dict, List


class TemporalEmbedding(nn.Module):
    """
    Learnable temporal embedding layer that maps discrete time stages
    to continuous vector representations.

    Time stages:
        - preoperative (术前)
        - intraoperative (术中)  
        - postoperative_24h (术后24h)
    """

    def __init__(
        self,
        num_time_stages: int = 3,
        embedding_dim: int = 768,
        dropout: float = 0.1
    ):
        """
        Args:
            num_time_stages: Number of discrete time points (default: 3)
            embedding_dim: Embedding dimension (must match BERT hidden_size)
            dropout: Dropout rate
        """
        super(TemporalEmbedding, self).__init__()

        self.num_time_stages = num_time_stages
        self.embedding_dim = embedding_dim

        # Time stage mapping: string -> index
        self.time_stage_map = {
            "preoperative": 0,
            "preop": 0,
            "intraoperative": 1,
            "intraop": 1,
            "postoperative_24h": 2,
            "postop_24h": 2,
            "postoperative": 2,
            "postop": 2
        }

        # Learnable time embeddings
        self.time_embeddings = nn.Embedding(num_time_stages, embedding_dim)

        # Positional encoding within same time stage (for multiple variables at same time)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small values
        nn.init.normal_(self.time_embeddings.weight, mean=0, std=0.02)

    def get_time_index(self, timestamp: str) -> int:
        """Convert timestamp string to embedding index."""
        timestamp = timestamp.lower().strip()
        if timestamp in self.time_stage_map:
            return self.time_stage_map[timestamp]
        # Default to preoperative if unknown
        return 0

    def forward(
        self,
        semantic_embeddings: torch.Tensor,
        timestamps: List[List[str]],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Add temporal embeddings to semantic embeddings.

        Args:
            semantic_embeddings: (batch_size, max_vars, embedding_dim)
                Semantic embeddings from BC-BERT encoder
            timestamps: List of lists, timestamps[i][j] = time stage string for variable j of patient i
            attention_mask: (batch_size, max_vars) binary mask

        Returns:
            Temporally-enhanced embeddings: (batch_size, max_vars, embedding_dim)
        """
        batch_size, max_vars, emb_dim = semantic_embeddings.shape
        device = semantic_embeddings.device

        # Create time index tensor
        time_indices = torch.zeros(batch_size, max_vars, dtype=torch.long, device=device)
        for i in range(batch_size):
            for j in range(max_vars):
                if attention_mask[i, j] == 1:
                    time_indices[i, j] = self.get_time_index(timestamps[i][j])

        # Get time embeddings
        time_emb = self.time_embeddings(time_indices)  # (batch_size, max_vars, emb_dim)

        # Add positional encoding for order within same time stage
        time_emb = self.positional_encoding(time_emb)

        # Combine semantic + temporal embeddings
        combined = semantic_embeddings + time_emb

        # Apply layer norm and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        # Apply mask
        combined = combined * attention_mask.unsqueeze(-1).float()

        return combined


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for variables within the same time stage.
    Helps distinguish between multiple variables collected at the same time point.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class ValueEmbedding(nn.Module):
    """
    Numerical value embedding layer.
    Converts continuous numerical values into learnable embeddings through
    a multi-layer perceptron, preserving magnitude information.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_bins: int = 256,
        dropout: float = 0.1
    ):
        super(ValueEmbedding, self).__init__()

        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        # Value binning embeddings (learnable quantization)
        self.value_embeddings = nn.Embedding(num_bins, embedding_dim)

        # Continuous value projection (for preserving exact magnitude)
        self.value_projector = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        nn.init.normal_(self.value_embeddings.weight, mean=0, std=0.02)

    def discretize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Discretize continuous value into bins.
        Uses percentile-based binning to handle wide-ranging clinical values.

        Args:
            value: (batch_size, max_vars) continuous values
        Returns:
            bin_indices: (batch_size, max_vars) integer bin indices
        """
        # Simple percentile-based binning (can be replaced with learned binning)
        # Clip extreme values and map to bins
        normalized = torch.clamp(value, min=-1e6, max=1e6)

        # Use log scale for skewed clinical distributions
        log_val = torch.log1p(torch.abs(normalized)) * torch.sign(normalized)

        # Map to [0, num_bins-1]
        # Assume reasonable clinical value range: [-10, 10] in log scale
        bin_indices = ((log_val + 10) / 20 * (self.num_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, min=0, max=self.num_bins - 1)

        return bin_indices

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert numerical values to embeddings.

        Args:
            values: (batch_size, max_vars) continuous numerical values
            attention_mask: (batch_size, max_vars) binary mask

        Returns:
            Value embeddings: (batch_size, max_vars, embedding_dim)
        """
        batch_size, max_vars = values.shape
        device = values.device

        # Discretize and get bin embeddings
        bin_indices = self.discretize_value(values)
        bin_emb = self.value_embeddings(bin_indices)  # (batch_size, max_vars, emb_dim)

        # Continuous value projection
        val_proj = self.value_projector(values.unsqueeze(-1))  # (batch_size, max_vars, emb_dim//2)

        # Concatenate and fuse
        combined = torch.cat([bin_emb, val_proj], dim=-1)
        embeddings = self.fusion(combined)

        # Apply mask
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test temporal embedding
    temp_emb = TemporalEmbedding(num_time_stages=3, embedding_dim=768).to(device)

    semantic_emb = torch.randn(2, 10, 768).to(device)  # batch=2, max_vars=10
    timestamps = [
        ["preop"] * 4 + ["intraop"] * 3 + ["postop_24h"] * 3,
        ["preop"] * 5 + ["intraop"] * 2 + ["postop_24h"] * 3
    ]
    attention_mask = torch.ones(2, 10).to(device)
    attention_mask[0, 8:] = 0  # Patient 0 has only 8 variables

    output = temp_emb(semantic_emb, timestamps, attention_mask)
    print(f"Temporal embedding output shape: {output.shape}")

    # Test value embedding
    val_emb = ValueEmbedding(embedding_dim=768).to(device)
    values = torch.randn(2, 10).to(device) * 100  # Clinical-scale values
    val_output = val_emb(values, attention_mask)
    print(f"Value embedding output shape: {val_output.shape}")

    print("Temporal Embedding test passed!")
