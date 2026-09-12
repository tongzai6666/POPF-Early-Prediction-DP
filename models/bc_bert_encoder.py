from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BCBERTEncoder(nn.Module):
    """Bio-Clinical BERT encoder for variable semantics.

    Variable descriptions are fixed across patients. They are tokenized once and
    encoded on each forward pass so the unfrozen upper BERT layers remain trainable.
    """

    def __init__(self, model_name: str, descriptions: List[str], freeze_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        tok = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )
        self.register_buffer("description_input_ids", tok["input_ids"], persistent=False)
        self.register_buffer("description_attention_mask", tok["attention_mask"], persistent=False)

    def forward(self) -> torch.Tensor:
        out = self.bert(
            input_ids=self.description_input_ids,
            attention_mask=self.description_attention_mask,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.dropout(cls)
