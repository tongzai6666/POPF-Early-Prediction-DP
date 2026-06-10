"""
Dual-Task Cascaded Classifier Module
====================================
Implements the two synergistic prediction tasks:
1. Primary MLP: Overall POPF occurrence prediction (binary classification)
2. Hierarchical Gated MLP: Severity stratification among positive cases (BL vs CR-POPF)

Paper Section 2.5: "The model output adopts a cascaded dual-classifier architecture:
first, a multi-layer perceptron (MLP) outputs the overall probability of POPF occurrence;
then, a hierarchical gated MLP fuses this probability with the shared feature embedding
to further classify POPF-positive cases into BL and CR-POPF."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrimaryMLPClassifier(nn.Module):
    """
    Primary classifier: Predicts overall POPF occurrence probability.

    Input: Shared feature embedding from Gated Transformer (patient-level representation)
    Output: P(POPF) - probability of any grade POPF (Grade A/B/C vs no POPF)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input feature embedding
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(PrimaryMLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer: binary classification (POPF vs no POPF)
        self.classifier = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim) patient-level feature embedding

        Returns:
            logits: (batch_size, 1) raw logits for POPF probability
            features: (batch_size, hidden_dims[-1]) intermediate features for secondary task
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits, features


class HierarchicalGatedMLP(nn.Module):
    """
    Hierarchical Gated MLP: Severity stratification among POPF-positive cases.

    Fuses the primary prediction probability with shared feature embedding
to classify severity: BL (Grade A, biochemical leak) vs CR-POPF (Grade B/C, clinically relevant).

    The gating mechanism dynamically controls information flow based on primary prediction confidence,
    ensuring severity prediction is only activated for high-confidence positive cases.
    """

    def __init__(
        self,
        input_dim: int = 768,
        primary_feature_dim: int = 128,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        gate_threshold: float = 0.5
    ):
        """
        Args:
            input_dim: Dimension of shared feature embedding from transformer
            primary_feature_dim: Dimension of features from primary classifier
            hidden_dims: List of hidden layer dimensions for severity MLP
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            gate_threshold: Threshold for primary prediction to activate severity branch
        """
        super(HierarchicalGatedMLP, self).__init__()

        self.input_dim = input_dim
        self.primary_feature_dim = primary_feature_dim
        self.gate_threshold = gate_threshold

        # Fusion layer: combines shared embedding + primary features + primary probability
        fusion_dim = input_dim + primary_feature_dim + 1  # +1 for probability scalar

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gating mechanism: learns to weight features based on severity-relevant patterns
        self.gate_controller = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

        # Severity MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.severity_extractor = nn.Sequential(*layers)

        # Output layer: binary classification (BL=0 vs CR-POPF=1)
        self.severity_classifier = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        shared_embedding: torch.Tensor,
        primary_features: torch.Tensor,
        primary_prob: torch.Tensor,
        primary_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            shared_embedding: (batch_size, input_dim) from transformer
            primary_features: (batch_size, primary_feature_dim) from primary MLP intermediate layer
            primary_prob: (batch_size, 1) sigmoid probability from primary classifier
            primary_logits: (batch_size, 1) raw logits from primary classifier

        Returns:
            severity_logits: (batch_size, 1) logits for CR-POPF probability
                Note: Only meaningful for POPF-positive cases; should be masked during training
        """
        batch_size = shared_embedding.size(0)

        # Concatenate all information sources
        combined = torch.cat([
            shared_embedding,           # (batch_size, input_dim)
            primary_features,           # (batch_size, primary_feature_dim)
            primary_prob                # (batch_size, 1)
        ], dim=-1)                      # (batch_size, input_dim + primary_feature_dim + 1)

        # Fusion
        fused = self.fusion_layer(combined)  # (batch_size, input_dim)

        # Gating: dynamically control information flow
        gate = self.gate_controller(fused)   # (batch_size, input_dim)
        gated_features = gate * fused        # (batch_size, input_dim)

        # Severity feature extraction
        severity_features = self.severity_extractor(gated_features)

        # Severity classification
        severity_logits = self.severity_classifier(severity_features)

        return severity_logits

    def get_severity_mask(self, primary_prob: torch.Tensor) -> torch.Tensor:
        """
        Create mask for severity loss computation.
        Only compute severity loss for patients with high primary prediction confidence.

        Args:
            primary_prob: (batch_size, 1) primary POPF probability

        Returns:
            mask: (batch_size, 1) binary mask (1 = compute severity loss, 0 = ignore)
        """
        # Use soft gating: weight severity loss by primary probability
        # Hard threshold can also be used for training stability
        mask = (primary_prob > self.gate_threshold).float()
        return mask


class DualTaskFramework(nn.Module):
    """
    Complete dual-task framework combining all components.

    Architecture flow:
    1. BC-BERT Encoder: Encode each variable into semantic embedding
    2. Temporal Embedding: Add time information
    3. Gated Transformer: Capture temporal dynamics and interactions
    4. Primary MLP: Predict overall POPF occurrence
    5. Hierarchical Gated MLP: Predict severity (BL vs CR-POPF) for positive cases
    """

    def __init__(
        self,
        bc_bert_encoder,
        temporal_embedding,
        gated_transformer,
        primary_classifier,
        severity_classifier,
        missing_handler,
        primary_loss_weight: float = 1.0,
        severity_loss_weight: float = 1.0,
        use_severity_gate: bool = True
    ):
        """
        Args:
            bc_bert_encoder: BC-BERT encoder module
            temporal_embedding: Temporal embedding module
            gated_transformer: Context-aware gated transformer
            primary_classifier: Primary MLP for POPF occurrence
            severity_classifier: Hierarchical gated MLP for severity
            missing_handler: Missing value handler
            primary_loss_weight: Weight for primary task loss
            severity_loss_weight: Weight for severity task loss
            use_severity_gate: Whether to use gating for severity task activation
        """
        super(DualTaskFramework, self).__init__()

        self.bc_bert_encoder = bc_bert_encoder
        self.temporal_embedding = temporal_embedding
        self.gated_transformer = gated_transformer
        self.primary_classifier = primary_classifier
        self.severity_classifier = severity_classifier
        self.missing_handler = missing_handler

        self.primary_loss_weight = primary_loss_weight
        self.severity_loss_weight = severity_loss_weight
        self.use_severity_gate = use_severity_gate

    def forward(
        self,
        batch_data: dict,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass through the complete dual-task framework.

        Args:
            batch_data: Dictionary containing:
                - 'timestamps': (batch_size, max_vars) time stage strings
                - 'variable_names': (batch_size, max_vars) variable name strings
                - 'values': (batch_size, max_vars) numerical values
                - 'attention_mask': (batch_size, max_vars) binary mask
                - 'center_ids': (batch_size,) center identifiers (optional)
            return_features: If True, return intermediate representations

        Returns:
            outputs: Dictionary containing:
                - 'primary_logits': (batch_size, 1) POPF occurrence logits
                - 'primary_prob': (batch_size, 1) POPF occurrence probability
                - 'severity_logits': (batch_size, 1) CR-POPF severity logits
                - 'severity_prob': (batch_size, 1) CR-POPF probability
                - 'patient_repr': (batch_size, d_model) patient-level representation
                - Optional: 'all_features' if return_features=True
        """
        device = next(self.parameters()).device

        # Step 1: BC-BERT encoding
        semantic_embeddings = self.bc_bert_encoder.get_semantic_embeddings(batch_data, device)
        # (batch_size, max_vars, hidden_size)

        # Step 2: Handle missing values
        semantic_embeddings = self.missing_handler.fill_missing(
            semantic_embeddings, 
            batch_data['attention_mask']
        )

        # Step 3: Add temporal embeddings
        temporal_embeddings = self.temporal_embedding(
            semantic_embeddings,
            batch_data['timestamps'],
            batch_data['attention_mask']
        )

        # Step 4: Gated Transformer processing
        transformer_output = self.gated_transformer(
            temporal_embeddings,
            batch_data['attention_mask']
        )
        # (batch_size, max_vars, d_model)

        # Step 5: Patient-level representation
        patient_repr = self.gated_transformer.get_patient_representation(
            transformer_output,
            batch_data['attention_mask']
        )
        # (batch_size, d_model)

        # Step 6: Primary classification (POPF occurrence)
        primary_logits, primary_features = self.primary_classifier(patient_repr)
        primary_prob = torch.sigmoid(primary_logits)

        # Step 7: Severity classification (BL vs CR-POPF)
        # Only process if we have positive cases or during inference
        severity_logits = self.severity_classifier(
            patient_repr,
            primary_features,
            primary_prob,
            primary_logits
        )
        severity_prob = torch.sigmoid(severity_logits)

        outputs = {
            'primary_logits': primary_logits,
            'primary_prob': primary_prob,
            'severity_logits': severity_logits,
            'severity_prob': severity_prob,
            'patient_repr': patient_repr
        }

        if return_features:
            outputs['semantic_embeddings'] = semantic_embeddings
            outputs['transformer_output'] = transformer_output
            outputs['primary_features'] = primary_features

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        labels: dict,
        use_severity_mask: bool = True
    ) -> dict:
        """
        Compute combined loss for both tasks.

        Args:
            outputs: Output dictionary from forward pass
            labels: Dictionary containing:
                - 'label_popf': (batch_size, 1) binary labels for POPF occurrence
                - 'label_severity': (batch_size, 1) binary labels for severity (0=BL, 1=CR-POPF)
                    Only valid for POPF-positive cases; can be -1 for negative cases
            use_severity_mask: Whether to mask severity loss for negative cases

        Returns:
            losses: Dictionary containing:
                - 'primary_loss': BCE loss for POPF occurrence
                - 'severity_loss': BCE loss for severity (masked)
                - 'total_loss': Weighted combination
        """
        # Primary task loss: Binary Cross Entropy
        primary_loss = F.binary_cross_entropy_with_logits(
            outputs['primary_logits'],
            labels['label_popf'].float(),
            reduction='mean'
        )

        # Severity task loss: Only for POPF-positive cases
        if use_severity_mask:
            # Create mask: 1 for positive cases, 0 for negative cases
            severity_mask = (labels['label_popf'] == 1).float()

            # Also mask out invalid severity labels (-1)
            valid_severity = (labels['label_severity'] >= 0).float()
            severity_mask = severity_mask * valid_severity

            if severity_mask.sum() > 0:
                severity_loss = F.binary_cross_entropy_with_logits(
                    outputs['severity_logits'],
                    labels['label_severity'].float(),
                    weight=severity_mask,
                    reduction='sum'
                ) / severity_mask.sum()
            else:
                severity_loss = torch.tensor(0.0, device=primary_loss.device)
        else:
            severity_loss = F.binary_cross_entropy_with_logits(
                outputs['severity_logits'],
                labels['label_severity'].float(),
                reduction='mean'
            )

        # Combined loss
        total_loss = (
            self.primary_loss_weight * primary_loss +
            self.severity_loss_weight * severity_loss
        )

        return {
            'primary_loss': primary_loss,
            'severity_loss': severity_loss,
            'total_loss': total_loss
        }

    def predict(
        self,
        batch_data: dict,
        primary_threshold: float = 0.4875  # Youden index optimal threshold from paper
    ) -> dict:
        """
        Inference method with risk stratification.

        Args:
            batch_data: Input data dictionary
            primary_threshold: Threshold for POPF positive/negative classification

        Returns:
            predictions: Dictionary containing:
                - 'popf_risk': High/Low risk classification
                - 'popf_probability': Raw probability
                - 'severity_probability': CR-POPF probability (if POPF positive)
                - 'severity_grade': Predicted grade (None/BL/CR-POPF)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch_data)

        primary_prob = outputs['primary_prob']
        severity_prob = outputs['severity_prob']

        # Risk stratification
        popf_positive = (primary_prob >= primary_threshold).squeeze(-1)

        predictions = {
            'popf_probability': primary_prob.squeeze(-1).cpu().numpy(),
            'severity_probability': severity_prob.squeeze(-1).cpu().numpy(),
            'popf_risk': ['High' if p else 'Low' for p in popf_positive.cpu().numpy()],
            'severity_grade': []
        }

        for i, is_positive in enumerate(popf_positive):
            if is_positive:
                # Severity classification: threshold at 0.5
                if severity_prob[i] >= 0.5:
                    predictions['severity_grade'].append('CR-POPF')
                else:
                    predictions['severity_grade'].append('BL')
            else:
                predictions['severity_grade'].append(None)

        return predictions


if __name__ == "__main__":
    # Quick test with dummy modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy components (normally imported from other modules)
    class DummyEncoder(nn.Module):
        def get_semantic_embeddings(self, batch_data, device):
            return torch.randn(batch_data['attention_mask'].size(0), 
                             batch_data['attention_mask'].size(1), 768).to(device)

    class DummyTempEmb(nn.Module):
        def forward(self, emb, ts, mask):
            return emb

    class DummyTransformer(nn.Module):
        def forward(self, x, mask):
            return x
        def get_patient_representation(self, x, mask):
            return x.mean(dim=1)

    class DummyMissingHandler(nn.Module):
        def fill_missing(self, emb, mask):
            return emb

    # Build framework
    framework = DualTaskFramework(
        bc_bert_encoder=DummyEncoder(),
        temporal_embedding=DummyTempEmb(),
        gated_transformer=DummyTransformer(),
        primary_classifier=PrimaryMLPClassifier(input_dim=768),
        severity_classifier=HierarchicalGatedMLP(input_dim=768, primary_feature_dim=128),
        missing_handler=DummyMissingHandler()
    ).to(device)

    # Test data
    batch_size, max_vars = 4, 15
    batch_data = {
        'timestamps': [["preop"] * 5 + ["intraop"] * 5 + ["postop_24h"] * 5 for _ in range(batch_size)],
        'variable_names': [["var"] * max_vars for _ in range(batch_size)],
        'values': torch.randn(batch_size, max_vars).to(device),
        'attention_mask': torch.ones(batch_size, max_vars).to(device),
        'center_ids': torch.randint(0, 6, (batch_size,)).to(device)
    }

    # Forward pass
    outputs = framework(batch_data)
    print(f"Primary logits shape: {outputs['primary_logits'].shape}")
    print(f"Primary prob shape: {outputs['primary_prob'].shape}")
    print(f"Severity logits shape: {outputs['severity_logits'].shape}")
    print(f"Patient repr shape: {outputs['patient_repr'].shape}")

    # Loss computation
    labels = {
        'label_popf': torch.randint(0, 2, (batch_size, 1)).float().to(device),
        'label_severity': torch.randint(0, 2, (batch_size, 1)).float().to(device)
    }
    losses = framework.compute_loss(outputs, labels)
    print(f"Total loss: {losses['total_loss'].item():.4f}")

    # Prediction
    preds = framework.predict(batch_data)
    print(f"Sample predictions: {preds['popf_risk'][:2]}")

    print("Dual-Task Framework test passed!")
