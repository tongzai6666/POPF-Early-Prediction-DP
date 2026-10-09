from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import yaml

from .bc_bert_encoder import BCBERTEncoder
from .temporal_embedding import TemporalEmbedding, ValueEmbedding, ContextFusion
from .attention_mask import MissingValueHandler
from .gated_transformer import ContextAwareGatedTransformer, masked_mean_pool


def _make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float,
    batch_norm: bool
) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = input_dim

    for h in hidden_dims:
        layers.append(nn.Linear(d, h))

        if batch_norm:
            layers.append(nn.BatchNorm1d(h))

        layers.extend(
            [
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )

        d = h

    return nn.Sequential(*layers)


class PrimaryMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_batch_norm: bool,
    ):
        super().__init__()

        self.backbone = _make_mlp(
            input_dim,
            hidden_dims,
            dropout,
            use_batch_norm,
        )

        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h = self.backbone(x)

        return self.output(h).squeeze(-1), h


class HierarchicalGatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        primary_feature_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_batch_norm: bool,
    ):
        super().__init__()

        self.primary_gate = nn.Sequential(
            nn.Linear(
                primary_feature_dim + 1,
                primary_feature_dim,
            ),
            nn.Sigmoid(),
        )

        severity_input = (
            input_dim
            + primary_feature_dim
            + 1
        )

        self.backbone = _make_mlp(
            severity_input,
            hidden_dims,
            dropout,
            use_batch_norm,
        )

        self.output = nn.Linear(
            hidden_dims[-1],
            1,
        )

    def forward(
        self,
        patient_repr: torch.Tensor,
        primary_feature: torch.Tensor,
        primary_prob: torch.Tensor,
    ) -> torch.Tensor:

        p = primary_prob.unsqueeze(-1)

        gate = self.primary_gate(
            torch.cat(
                [
                    primary_feature,
                    p,
                ],
                dim=-1,
            )
        )

        x = torch.cat(
            [
                patient_repr,
                gate * primary_feature,
                p,
            ],
            dim=-1,
        )

        return self.output(
            self.backbone(x)
        ).squeeze(-1)


class DualTaskFramework(nn.Module):
    """
    Hierarchical soft-sequential model.

    First-stage output:
        p_leakage = P(pancreatic leakage)

    where pancreatic leakage includes:
        biochemical leak (BL) + Grade B/C CR-POPF.

    Second-stage output:
        p_conditional_cr =
        P(CR-POPF | pancreatic leakage)

    The final population-level CR-POPF probability is:

        P(CR-POPF)
        =
        P(pancreatic leakage)
        *
        P(CR-POPF | pancreatic leakage)

    The severity loss is evaluated only among leakage-positive
    patients during training.

    The second-stage classifier remains soft-sequential because
    it receives:
        1. the shared patient representation,
        2. the primary hidden representation,
        3. the first-stage leakage probability.
    """

    def __init__(
        self,
        feature_descriptions: List[str],
        num_categories: int,
        model_cfg: Dict,
        primary_pos_weight: float = 1.0,
        severity_pos_weight: float = 1.0,
    ):
        super().__init__()

        d_model = int(
            model_cfg["d_model"]
        )

        self.primary_loss_weight = float(
            model_cfg.get(
                "primary_loss_weight",
                1.0,
            )
        )

        self.severity_loss_weight = float(
            model_cfg.get(
                "severity_loss_weight",
                1.0,
            )
        )

        # ---------------------------------------------------------
        # Bio-Clinical BERT semantic encoder
        # ---------------------------------------------------------
        self.semantic_encoder = BCBERTEncoder(
            model_name=model_cfg["bert_model_name"],
            descriptions=feature_descriptions,
            freeze_layers=int(
                model_cfg["freeze_bert_layers"]
            ),
            dropout=float(
                model_cfg["transformer_dropout"]
            ),
        )

        # ---------------------------------------------------------
        # Numerical / categorical value embedding
        # ---------------------------------------------------------
        self.value_embedding = ValueEmbedding(
            d_model,
            num_categories,
        )

        # ---------------------------------------------------------
        # Four-stage temporal embedding
        # ---------------------------------------------------------
        self.temporal_embedding = TemporalEmbedding(
            int(
                model_cfg["num_time_stages"]
            ),
            d_model,
            float(
                model_cfg["transformer_dropout"]
            ),
        )

        # ---------------------------------------------------------
        # Semantic + value + temporal fusion
        # ---------------------------------------------------------
        self.fusion = ContextFusion(
            d_model,
            float(
                model_cfg["transformer_dropout"]
            ),
        )

        # ---------------------------------------------------------
        # Missing-value handling
        # ---------------------------------------------------------
        self.missing_handler = MissingValueHandler(
            d_model
        )

        # ---------------------------------------------------------
        # Context-aware gated Transformer
        # ---------------------------------------------------------
        self.transformer = ContextAwareGatedTransformer(
            d_model=d_model,
            num_heads=int(
                model_cfg["num_attention_heads"]
            ),
            num_layers=int(
                model_cfg["num_transformer_layers"]
            ),
            d_ff=int(
                model_cfg["d_ff"]
            ),
            dropout=float(
                model_cfg["transformer_dropout"]
            ),
        )

        # ---------------------------------------------------------
        # First-stage leakage classifier
        # ---------------------------------------------------------
        self.primary_classifier = PrimaryMLPClassifier(
            d_model,
            list(
                model_cfg["primary_hidden_dims"]
            ),
            float(
                model_cfg["classifier_dropout"]
            ),
            bool(
                model_cfg["use_batch_norm"]
            ),
        )

        # ---------------------------------------------------------
        # Second-stage CR-POPF severity classifier
        # ---------------------------------------------------------
        self.severity_classifier = HierarchicalGatedMLP(
            d_model,
            int(
                model_cfg[
                    "primary_hidden_dims"
                ][-1]
            ),
            list(
                model_cfg[
                    "severity_hidden_dims"
                ]
            ),
            float(
                model_cfg[
                    "classifier_dropout"
                ]
            ),
            bool(
                model_cfg[
                    "use_batch_norm"
                ]
            ),
        )

        # ---------------------------------------------------------
        # Class weights
        # ---------------------------------------------------------
        self.register_buffer(
            "primary_pos_weight",
            torch.tensor(
                float(
                    primary_pos_weight
                )
            ),
        )

        self.register_buffer(
            "severity_pos_weight",
            torch.tensor(
                float(
                    severity_pos_weight
                )
            ),
        )

    def encode(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        bsz, _ = batch[
            "numeric_values"
        ].shape

        # ---------------------------------------------------------
        # Semantic identity of each clinical variable
        # ---------------------------------------------------------
        semantic = (
            self.semantic_encoder()
            .unsqueeze(0)
            .expand(
                bsz,
                -1,
                -1,
            )
        )

        # ---------------------------------------------------------
        # Patient-specific observed values
        # ---------------------------------------------------------
        value = self.value_embedding(
            batch["numeric_values"],
            batch["category_ids"],
            batch["feature_type_ids"],
        )

        # ---------------------------------------------------------
        # Perioperative temporal stage
        # ---------------------------------------------------------
        stage_ids = batch[
            "stage_ids"
        ]

        if stage_ids.dim() == 1:
            stage_ids = (
                stage_ids
                .unsqueeze(0)
                .expand(
                    bsz,
                    -1,
                )
            )

        time = self.temporal_embedding(
            stage_ids
        )

        # ---------------------------------------------------------
        # Fuse semantic, value, and temporal representations
        # ---------------------------------------------------------
        x = self.fusion(
            semantic,
            value,
            time,
        )

        # ---------------------------------------------------------
        # Replace missing inputs internally and apply mask
        # ---------------------------------------------------------
        x = self.missing_handler(
            x,
            batch["observed_mask"],
        )

        # ---------------------------------------------------------
        # Context-aware gated Transformer
        # ---------------------------------------------------------
        x = self.transformer(
            x,
            batch["observed_mask"],
        )

        # ---------------------------------------------------------
        # Patient-level representation
        # ---------------------------------------------------------
        return masked_mean_pool(
            x,
            batch["observed_mask"],
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # ---------------------------------------------------------
        # Shared patient representation
        # ---------------------------------------------------------
        patient_repr = self.encode(
            batch
        )

        # ---------------------------------------------------------
        # Stage 1:
        # P(pancreatic leakage)
        #
        # Leakage includes:
        # BL + Grade B/C CR-POPF
        # ---------------------------------------------------------
        (
            primary_logit,
            primary_feature,
        ) = self.primary_classifier(
            patient_repr
        )

        primary_prob = torch.sigmoid(
            primary_logit
        )

        # ---------------------------------------------------------
        # Stage 2:
        # P(CR-POPF | pancreatic leakage)
        # ---------------------------------------------------------
        severity_logit = (
            self.severity_classifier(
                patient_repr,
                primary_feature,
                primary_prob,
            )
        )

        severity_prob = torch.sigmoid(
            severity_logit
        )

        # ---------------------------------------------------------
        # Final population-level CR-POPF probability
        #
        # P(CR-POPF)
        # =
        # P(leakage)
        # ×
        # P(CR-POPF | leakage)
        # ---------------------------------------------------------
        final_cr_prob = (
            primary_prob
            * severity_prob
        )

        return {
            "patient_repr":
                patient_repr,

            "primary_logit":
                primary_logit,

            # P(leakage)
            "primary_prob":
                primary_prob,

            "severity_logit":
                severity_logit,

            # P(CR-POPF | leakage)
            "severity_prob":
                severity_prob,

            # -----------------------------------------------------
            # Formal CR-POPF endpoint:
            # population-level P(CR-POPF)
            # -----------------------------------------------------
            "cr_prob":
                final_cr_prob,

            # Alias retained for backward compatibility
            "joint_cr_prob":
                final_cr_prob,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # ---------------------------------------------------------
        # True labels
        # ---------------------------------------------------------
        leakage = batch[
            "label_leakage"
        ].float()

        cr = batch[
            "label_cr_popf"
        ].float()

        # ---------------------------------------------------------
        # Stage-1 loss:
        # calculated across the entire training cohort
        # ---------------------------------------------------------
        primary_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                outputs[
                    "primary_logit"
                ],
                leakage,
                pos_weight=self.primary_pos_weight,
            )
        )

        # ---------------------------------------------------------
        # Stage-2 loss:
        # only among true leakage-positive patients
        #
        # Within this subgroup:
        # BL = 0
        # CR-POPF = 1
        # ---------------------------------------------------------
        leakage_mask = (
            leakage == 1
        )

        if leakage_mask.any():

            severity_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    outputs[
                        "severity_logit"
                    ][leakage_mask],
                    cr[
                        leakage_mask
                    ],
                    pos_weight=self.severity_pos_weight,
                )
            )

        else:

            severity_loss = (
                outputs[
                    "severity_logit"
                ].sum()
                * 0.0
            )

        # ---------------------------------------------------------
        # Joint optimization
        # ---------------------------------------------------------
        total = (
            self.primary_loss_weight
            * primary_loss
            +
            self.severity_loss_weight
            * severity_loss
        )

        return {
            "total_loss":
                total,

            "primary_loss":
                primary_loss,

            "severity_loss":
                severity_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        leakage_threshold: float,
        cr_threshold: float,
    ) -> Dict:

        self.eval()

        out = self.forward(
            batch
        )

        # ---------------------------------------------------------
        # Population-level leakage probability
        # ---------------------------------------------------------
        p_leak = out[
            "primary_prob"
        ]

        # ---------------------------------------------------------
        # Conditional CR-POPF probability:
        # P(CR-POPF | leakage)
        # ---------------------------------------------------------
        p_conditional_cr = out[
            "severity_prob"
        ]

        # ---------------------------------------------------------
        # Final population-level CR-POPF probability:
        #
        # P(CR-POPF)
        # =
        # P(leakage)
        # ×
        # P(CR-POPF | leakage)
        # ---------------------------------------------------------
        p_cr = out[
            "cr_prob"
        ]

        # Backward-compatible alias
        p_joint = out[
            "joint_cr_prob"
        ]

        # ---------------------------------------------------------
        # Threshold-based classifications
        # ---------------------------------------------------------
        leak_positive = (
            p_leak
            >= leakage_threshold
        )

        cr_positive = (
            p_cr
            >= cr_threshold
        )

        # ---------------------------------------------------------
        # Patient-level category labels
        #
        # The locked pancreatic leakage threshold is 0.40.
        # The locked CR-POPF threshold is 0.31 and is applied
        # to the final population-level CR-POPF probability.
        # ---------------------------------------------------------
        grades = []

        for leak, cr_flag in zip(
            leak_positive
            .cpu()
            .tolist(),
            cr_positive
            .cpu()
            .tolist(),
        ):

            if cr_flag:

                grades.append(
                    "CR-POPF"
                )

            elif leak:

                grades.append(
                    "BL"
                )

            else:

                grades.append(
                    "No leakage"
                )

        return {
            # P(leakage)
            "leakage_probability":
                p_leak,

            # P(CR-POPF | leakage)
            "conditional_cr_popf_probability":
                p_conditional_cr,

            # Final population-level P(CR-POPF)
            "cr_popf_probability":
                p_cr,

            # Same value retained for compatibility
            "joint_cr_popf_probability":
                p_joint,

            "leakage_positive":
                leak_positive,

            "cr_popf_positive":
                cr_positive,

            "grade":
                grades,
        }


def build_model(
    feature_descriptions: List[str],
    num_categories: int,
    config_path: str,
    primary_pos_weight: float = 1.0,
    severity_pos_weight: float = 1.0,
) -> DualTaskFramework:

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(
            f
        )

    return DualTaskFramework(
        feature_descriptions=
            feature_descriptions,

        num_categories=
            num_categories,

        model_cfg=
            cfg["model"],

        primary_pos_weight=
            primary_pos_weight,

        severity_pos_weight=
            severity_pos_weight,
    )
