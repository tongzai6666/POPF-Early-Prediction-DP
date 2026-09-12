from __future__ import annotations

import torch
import torch.nn as nn


class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~observed_mask.bool()
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        g1 = self.attn_gate(torch.cat([x, attn_out], dim=-1))
        x = self.norm1(x + self.dropout(g1 * attn_out))
        ff_out = self.ff(x)
        g2 = self.ff_gate(torch.cat([x, ff_out], dim=-1))
        x = self.norm2(x + self.dropout(g2 * ff_out))
        return x


class ContextAwareGatedTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, observed_mask)
        return x


def masked_mean_pool(x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    mask = observed_mask.unsqueeze(-1).to(x.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom
