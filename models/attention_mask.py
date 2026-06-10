"""
Attention Mask Module
=====================
Implements masking strategy within self-attention mechanism to mitigate
the interference of missing values on representation learning.

Paper Section 2.5: "A masking strategy within the self-attention mechanism 
is utilized to mitigate the interference of missing values on representation learning."
"""

import torch
import torch.nn as nn
from typing import Optional


class VariableAttentionMask(nn.Module):
    """
    Generates attention masks for handling missing variables in perioperative EHR data.

    In multicenter settings, different centers may record different variable sets.
    This module creates masks that prevent the model from attending to missing variables,
    allowing dynamic weight redistribution to available features.
    """

    def __init__(self, fill_value: float = -1e9):
        """
        Args:
            fill_value: Value to fill for masked positions (default: large negative for softmax)
        """
        super(VariableAttentionMask, self).__init__()
        self.fill_value = fill_value

    def create_padding_mask(
        self,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create padding mask from binary attention mask.

        Args:
            attention_mask: (batch_size, seq_len) binary tensor
                1 = valid variable, 0 = missing variable

        Returns:
            padding_mask: (batch_size, 1, 1, seq_len) boolean tensor
                True = mask (missing), False = keep (valid)
        """
        # Invert: 1 (valid) -> False (don't mask), 0 (missing) -> True (mask)
        padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
        return padding_mask  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Create look-ahead mask to prevent attending to future time points.
        Used if enforcing temporal causality (optional in this framework).

        Args:
            size: Sequence length
            device: Computation device

        Returns:
            look_ahead_mask: (1, 1, size, size) boolean tensor
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.bool().unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)

    def combine_masks(
        self,
        padding_mask: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine padding mask and look-ahead mask.

        Args:
            padding_mask: (batch_size, 1, 1, seq_len)
            look_ahead_mask: (1, 1, seq_len, seq_len) or None

        Returns:
            combined_mask: (batch_size, 1, seq_len, seq_len)
        """
        if look_ahead_mask is not None:
            # Expand padding mask to match look-ahead dimensions
            combined = padding_mask | look_ahead_mask
            return combined
        else:
            # Expand padding mask for broadcasting in attention
            return padding_mask.expand(-1, -1, padding_mask.size(-1), -1)

    def apply_mask(
        self,
        attention_scores: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mask to attention scores before softmax.

        Args:
            attention_scores: (batch_size, num_heads, seq_len, seq_len)
            mask: (batch_size, 1, seq_len, seq_len) boolean tensor
                True = mask, False = keep

        Returns:
            masked_scores: (batch_size, num_heads, seq_len, seq_len)
        """
        # Expand mask to match number of heads
        if mask.size(1) == 1:
            mask = mask.expand(-1, attention_scores.size(1), -1, -1)

        # Fill masked positions with large negative value
        masked_scores = attention_scores.masked_fill(mask, self.fill_value)
        return masked_scores


class MissingValueHandler(nn.Module):
    """
    Handles missing values in input features before attention computation.

    Strategies:
    1. Learnable missing token embedding (for completely missing variables)
    2. Zero masking with attention weight redistribution
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        missing_token_init: str = "zero"
    ):
        super(MissingValueHandler, self).__init__()

        self.embedding_dim = embedding_dim

        # Learnable missing token embedding
        if missing_token_init == "zero":
            self.missing_token = nn.Parameter(torch.zeros(embedding_dim))
        elif missing_token_init == "normal":
            self.missing_token = nn.Parameter(torch.randn(embedding_dim) * 0.02)
        else:
            self.missing_token = nn.Parameter(torch.zeros(embedding_dim))

    def fill_missing(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fill missing variable positions with learnable missing token.

        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            attention_mask: (batch_size, seq_len) binary mask

        Returns:
            filled_embeddings: (batch_size, seq_len, embedding_dim)
        """
        # Expand mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)

        # For valid positions: keep original embedding
        # For missing positions: use missing token
        filled = embeddings * mask_expanded + self.missing_token * (1 - mask_expanded)

        return filled

    def compute_available_variable_ratio(
        self,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ratio of available variables per patient.
        Useful for adaptive normalization or confidence estimation.

        Args:
            attention_mask: (batch_size, seq_len)

        Returns:
            ratios: (batch_size,) float tensor in [0, 1]
        """
        return attention_mask.float().mean(dim=1)


class CenterAwareMask(nn.Module):
    """
    Center-aware masking for handling multicenter data heterogeneity.

    Different centers may have different variable availability patterns.
    This module can optionally apply center-specific masking rules.
    """

    def __init__(
        self,
        num_centers: int = 6,
        embedding_dim: int = 768
    ):
        super(CenterAwareMask, self).__init__()

        self.num_centers = num_centers
        self.embedding_dim = embedding_dim

        # Center-specific missing token adaptations
        self.center_adaptations = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_centers)
        ])

    def adapt_for_center(
        self,
        embeddings: torch.Tensor,
        center_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply center-specific adaptation to missing value handling.

        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            center_ids: (batch_size,) integer center identifiers (0-5)
            attention_mask: (batch_size, seq_len)

        Returns:
            adapted_embeddings: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device

        result = embeddings.clone()

        for center_idx in range(self.num_centers):
            center_mask = (center_ids == center_idx)  # (batch_size,)
            if center_mask.any():
                center_emb = embeddings[center_mask]  # (num_center_patients, seq_len, emb_dim)
                center_attn = attention_mask[center_mask]

                # Apply center-specific adaptation to missing positions
                missing_positions = (center_attn == 0).unsqueeze(-1).float()
                adapted = self.center_adaptations[center_idx](center_emb)

                # Only update missing positions
                result[center_mask] = center_emb * (1 - missing_positions) + adapted * missing_positions

        return result


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test VariableAttentionMask
    mask_module = VariableAttentionMask().to(device)

    batch_size, seq_len = 2, 10
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    attention_mask[0, 5:] = 0  # Patient 0 missing last 5 variables
    attention_mask[1, 7:] = 0  # Patient 1 missing last 3 variables

    padding_mask = mask_module.create_padding_mask(attention_mask)
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask[0, 0, 0]: {padding_mask[0, 0, 0]}")

    # Test attention score masking
    attn_scores = torch.randn(batch_size, 8, seq_len, seq_len).to(device)
    combined_mask = mask_module.combine_masks(padding_mask)
    masked_scores = mask_module.apply_mask(attn_scores, combined_mask)
    print(f"Masked scores shape: {masked_scores.shape}")

    # Test MissingValueHandler
    mv_handler = MissingValueHandler(embedding_dim=768).to(device)
    embeddings = torch.randn(batch_size, seq_len, 768).to(device)
    filled = mv_handler.fill_missing(embeddings, attention_mask)
    print(f"Filled embeddings shape: {filled.shape}")

    ratio = mv_handler.compute_available_variable_ratio(attention_mask)
    print(f"Available variable ratios: {ratio}")

    print("Attention Mask test passed!")
