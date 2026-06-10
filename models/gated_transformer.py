"""
Context-Aware Gated Transformer Module
======================================
Captures temporal trajectory of individual parameters and nonlinear interactions
among different variables using multi-head self-attention with gating mechanism.

Paper Section 2.5: "Gated Transformer layers are employed to capture the temporal 
trajectory of individual parameters as well as nonlinear interactions among different 
variables. The Transformer... excels at capturing long-range dependencies."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention_mask import VariableAttentionMask


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with gating mechanism.

    The gate controls information flow based on feature importance,
    enhancing relevant temporal dependencies while suppressing noise.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_activation: str = "sigmoid"
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            gate_activation: Gate activation function ("sigmoid" or "tanh")
        """
        super(GatedMultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Gating mechanism: learns which features to emphasize
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_activation = nn.Sigmoid() if gate_activation == "sigmoid" else nn.Tanh()

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Feed-forward network with gating
        self.ffn = GatedFeedForward(d_model, d_model * 4, dropout)

        # Mask handler
        self.mask_handler = VariableAttentionMask()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o, self.gate_linear]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with gating.

        Args:
            x: (batch_size, seq_len, d_model) input embeddings
            attention_mask: (batch_size, seq_len) binary mask (1=valid, 0=missing)
            look_ahead_mask: Optional causal mask

        Returns:
            output: (batch_size, seq_len, d_model) gated attention output
        """
        batch_size, seq_len, _ = x.shape

        # Pre-norm architecture
        residual = x
        x = self.layer_norm1(x)

        # Compute Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # (batch_size, num_heads, seq_len, seq_len)

        # Apply mask
        padding_mask = self.mask_handler.create_padding_mask(attention_mask)
        combined_mask = self.mask_handler.combine_masks(padding_mask, look_ahead_mask)
        scores = self.mask_handler.apply_mask(scores, combined_mask)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout(attn_output)

        # Gating mechanism: compute gate values
        gate = self.gate_activation(self.gate_linear(x))

        # Apply gate to attention output
        gated_output = gate * attn_output + (1 - gate) * residual

        # Second sub-layer: FFN with gating
        ffn_output = self.ffn(self.layer_norm2(gated_output))

        # Final residual connection
        output = gated_output + ffn_output

        # Apply attention mask to output (zero out missing positions)
        output = output * attention_mask.unsqueeze(-1).float()

        return output


class GatedFeedForward(nn.Module):
    """
    Feed-forward network with gating mechanism.

    Uses GELU activation with a learned gate to control information flow,
    similar to Gated Linear Units (GLU) but adapted for transformer architecture.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super(GatedFeedForward, self).__init__()

        # Two parallel branches: transformation and gate
        self.linear_transform = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.xavier_uniform_(self.linear_gate.weight)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.linear_transform.bias)
        nn.init.zeros_(self.linear_gate.bias)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Transformation branch
        transform = self.activation(self.linear_transform(x))

        # Gate branch
        gate = torch.sigmoid(self.linear_gate(x))

        # Gated combination
        gated = transform * gate
        gated = self.dropout(gated)

        # Output projection
        output = self.output_linear(gated)

        return output


class ContextAwareGatedTransformer(nn.Module):
    """
    Stack of Gated Transformer layers with context-aware processing.

    This is the core temporal modeling module that captures:
    1. Temporal trajectory of individual parameters across perioperative stages
    2. Nonlinear interactions among different clinical variables
    3. Long-range dependencies through multi-head self-attention
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 3072,
        dropout: float = 0.1,
        use_look_ahead: bool = False
    ):
        """
        Args:
            d_model: Model dimension (must match BERT output)
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            use_look_ahead: Whether to enforce temporal causality
        """
        super(ContextAwareGatedTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_look_ahead = use_look_ahead

        # Stack of gated transformer layers
        self.layers = nn.ModuleList([
            GatedMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Optional: temporal aggregation for global patient representation
        self.temporal_pooling = TemporalPooling(d_model, pooling_type="attention")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            x: (batch_size, seq_len, d_model) input embeddings (from BC-BERT + Temporal)
            attention_mask: (batch_size, seq_len) binary mask
            return_all_layers: If True, return outputs from all layers

        Returns:
            If return_all_layers=False:
                output: (batch_size, seq_len, d_model) final layer output
            If return_all_layers=True:
                all_outputs: List of (batch_size, seq_len, d_model) for each layer
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Optional look-ahead mask for temporal causality
        look_ahead_mask = None
        if self.use_look_ahead:
            look_ahead_mask = VariableAttentionMask().create_look_ahead_mask(seq_len, device)

        all_outputs = []
        current = x

        for layer in self.layers:
            current = layer(current, attention_mask, look_ahead_mask)
            all_outputs.append(current)

        # Final normalization
        output = self.final_norm(current)

        if return_all_layers:
            return output, all_outputs
        return output

    def get_patient_representation(
        self,
        layer_outputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate variable-level representations into patient-level representation.

        Args:
            layer_outputs: (batch_size, seq_len, d_model) from final transformer layer
            attention_mask: (batch_size, seq_len)

        Returns:
            patient_repr: (batch_size, d_model) global patient representation
        """
        return self.temporal_pooling(layer_outputs, attention_mask)


class TemporalPooling(nn.Module):
    """
    Temporal pooling mechanism to aggregate variable-level features into patient-level representation.

    Uses attention-based pooling to give higher weight to clinically important variables.
    """

    def __init__(
        self,
        d_model: int = 768,
        pooling_type: str = "attention"
    ):
        super(TemporalPooling, self).__init__()

        self.pooling_type = pooling_type

        if pooling_type == "attention":
            # Learnable attention pooling
            self.attention_vector = nn.Parameter(torch.randn(d_model))
            nn.init.normal_(self.attention_vector, mean=0, std=0.02)
        elif pooling_type == "mean":
            pass  # Simple mean pooling
        elif pooling_type == "max":
            pass  # Max pooling

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, d_model)
        """
        if self.pooling_type == "attention":
            # Compute attention scores
            scores = torch.matmul(x, self.attention_vector)  # (batch_size, seq_len)

            # Mask and softmax
            scores = scores.masked_fill((attention_mask == 0), -1e9)
            weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)

            # Weighted sum
            pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, d_model)

        elif self.pooling_type == "mean":
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_x = (x * mask_expanded).sum(dim=1)  # (batch_size, d_model)
            count = attention_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            pooled = sum_x / torch.clamp(count, min=1)

        elif self.pooling_type == "max":
            # Masked max pooling
            x_masked = x.masked_fill((attention_mask == 0).unsqueeze(-1), -1e9)
            pooled = x_masked.max(dim=1)[0]  # (batch_size, d_model)

        return pooled


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size, seq_len, d_model = 2, 15, 768

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    attention_mask[0, 10:] = 0  # Patient 0 has 10 variables
    attention_mask[1, 12:] = 0  # Patient 1 has 12 variables

    # Test single layer
    single_layer = GatedMultiHeadAttention(d_model=768, num_heads=8).to(device)
    output = single_layer(x, attention_mask)
    print(f"Single layer output shape: {output.shape}")

    # Test full transformer stack
    transformer = ContextAwareGatedTransformer(
        d_model=768,
        num_heads=8,
        num_layers=4,
        d_ff=3072
    ).to(device)

    final_output, all_layers = transformer(x, attention_mask, return_all_layers=True)
    print(f"Final output shape: {final_output.shape}")
    print(f"Number of layers: {len(all_layers)}")

    # Test patient representation
    patient_repr = transformer.get_patient_representation(final_output, attention_mask)
    print(f"Patient representation shape: {patient_repr.shape}")

    print("Gated Transformer test passed!")
