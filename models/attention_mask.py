from __future__ import annotations

import torch
import torch.nn as nn


class MissingValueHandler(nn.Module):
    """Learned missing token plus observed-value mask.

    Missing features are replaced by a learned token for numerical stability, then
    excluded as keys/values in Transformer attention and from masked pooling.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.missing_token = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        token = self.missing_token.view(1, 1, -1)
        return torch.where(observed_mask.unsqueeze(-1), x, token)
