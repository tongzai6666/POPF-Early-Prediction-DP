"""
Bio-Clinical BERT Encoder Module
================================
Encodes each perioperative clinical variable (timestamp + variable_name + value)
into a semantic embedding vector using Bio-Clinical BERT pre-trained weights.

Reference: Lee et al. BioBERT (Bioinformatics 2020)
Paper Section 2.5: "Each perioperative variable... is encoded into a modular embedding vector.
Variables are independently encapsulated to preserve their temporal relationships and clinical semantics."
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional


class BC_BERT_Encoder(nn.Module):
    """
    Bio-Clinical BERT encoder for structured EHR variables.

    Each clinical variable is represented as a text string:
        "{timestamp} {variable_name} {value}"

    The BERT encoder extracts [CLS] token embedding as the semantic representation.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        hidden_size: int = 768,
        freeze_layers: int = 6,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace model identifier for Bio-Clinical BERT
            hidden_size: BERT hidden dimension (768 for base model)
            freeze_layers: Number of bottom transformer layers to freeze
            dropout: Dropout rate for regularization
        """
        super(BC_BERT_Encoder, self).__init__()

        self.hidden_size = hidden_size

        # Load pre-trained Bio-Clinical BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze bottom layers to preserve biomedical semantic knowledge
        # while allowing top layers to adapt to perioperative EHR structure
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Projection layer to standardize output dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

    def encode_variable(
        self,
        timestamp: str,
        variable_name: str,
        value: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode a single clinical variable into semantic embedding.

        Args:
            timestamp: Time of collection (e.g., "preoperative", "postoperative_24h")
            variable_name: Clinical variable name (e.g., "DFA", "PT")
            value: String representation of value (e.g., "2984", "14.8")
            device: Computation device

        Returns:
            Semantic embedding vector of shape (hidden_size,)
        """
        # Construct natural language-like text for BERT encoding
        text = f"{timestamp} {variable_name} is {value}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        with torch.no_grad() if not any(p.requires_grad for p in self.bert.parameters()) else torch.enable_grad():
            outputs = self.bert(**inputs)
            # Extract [CLS] token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)

        # Project to standardized space
        projected = self.projection(cls_embedding)  # (1, hidden_size)

        return projected.squeeze(0)  # (hidden_size,)

    def forward(
        self,
        variable_texts: List[str],
        attention_mask: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Batch encoding of multiple clinical variables.

        Args:
            variable_texts: List of formatted text strings, each representing one variable
                Format: "{timestamp} {variable_name} {value}"
            attention_mask: Binary mask indicating valid (1) vs missing (0) variables
                Shape: (batch_size, max_num_variables)
            device: Computation device

        Returns:
            Variable embeddings of shape (batch_size, max_num_variables, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = len(variable_texts)
        max_vars = max(len(v) for v in variable_texts) if isinstance(variable_texts[0], list) else len(variable_texts)

        # Tokenize all variables in batch
        encoded = self.tokenizer(
            variable_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        # Forward through BERT
        outputs = self.bert(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (total_variables, hidden_size)

        # Project
        projected = self.projection(cls_embeddings)  # (total_variables, hidden_size)

        # Reshape to (batch_size, max_num_variables, hidden_size)
        if attention_mask is not None:
            # Use attention_mask to reshape properly
            embeddings = torch.zeros(batch_size, max_vars, self.hidden_size, device=device)
            idx = 0
            for i in range(batch_size):
                num_valid = int(attention_mask[i].sum().item())
                embeddings[i, :num_valid] = projected[idx:idx + num_valid]
                idx += num_valid
            return embeddings
        else:
            return projected.unsqueeze(0)  # (1, num_vars, hidden_size)

    def get_semantic_embeddings(
        self,
        batch_data: Dict[str, torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        """
        High-level interface: convert structured batch data to semantic embeddings.

        Args:
            batch_data: Dictionary containing:
                - 'timestamps': (batch_size, max_vars) token IDs or strings
                - 'variable_names': (batch_size, max_vars) variable name strings
                - 'values': (batch_size, max_vars) value strings
                - 'attention_mask': (batch_size, max_vars) binary mask
            device: Computation device

        Returns:
            Semantic embeddings: (batch_size, max_vars, hidden_size)
        """
        batch_size = batch_data['attention_mask'].size(0)
        max_vars = batch_data['attention_mask'].size(1)

        # Flatten all valid variables for batch processing
        all_texts = []
        for i in range(batch_size):
            for j in range(max_vars):
                if batch_data['attention_mask'][i, j] == 1:
                    ts = batch_data['timestamps'][i, j]
                    vn = batch_data['variable_names'][i, j]
                    val = batch_data['values'][i, j]

                    # Convert tensor to string if needed
                    if isinstance(ts, torch.Tensor):
                        ts = str(ts.item())
                    if isinstance(vn, torch.Tensor):
                        vn = str(vn.item())
                    if isinstance(val, torch.Tensor):
                        val = f"{val.item():.2f}"

                    all_texts.append(f"{ts} {vn} {val}")

        if len(all_texts) == 0:
            return torch.zeros(batch_size, max_vars, self.hidden_size, device=device)

        # Batch encode
        encoded = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        outputs = self.bert(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embeddings)

        # Reconstruct batch structure
        embeddings = torch.zeros(batch_size, max_vars, self.hidden_size, device=device)
        idx = 0
        for i in range(batch_size):
            for j in range(max_vars):
                if batch_data['attention_mask'][i, j] == 1:
                    embeddings[i, j] = projected[idx]
                    idx += 1

        return embeddings


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BC_BERT_Encoder().to(device)

    test_texts = [
        "preoperative DFA 150",
        "postoperative_24h DFA 2984",
        "preoperative PT 14.8"
    ]

    embeddings = encoder(test_texts, device=device)
    print(f"Test embeddings shape: {embeddings.shape}")
    print("BC-BERT Encoder test passed!")
