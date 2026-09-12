# Supplementary Methods 1. Technical implementation of the temporal dual-task framework

## Reporting framework and reproducibility

The technical description below is provided to support reproducibility of the temporal dual-task prediction framework. Reporting was structured with reference to TRIPOD+AI, the current reporting guidance for clinical prediction model studies using regression or machine-learning methods, and with consideration of the analysis principles highlighted in PROBAST+AI. The public implementation separates model development from evaluation, records all preprocessing parameters from the training cohort, locks model weights and decision thresholds before evaluation, and provides a version-pinned software environment. No patient-level data from the internal or external evaluation cohorts are used to fit preprocessing parameters, class weights, model parameters, or decision thresholds.

## Input representation and four-stage temporal organization

Each patient is represented as an ordered set of heterogeneous perioperative variables assigned to one of four clinically defined stages: baseline characteristics, preoperative information, intraoperative information, and postoperative measurements available within 24 h after surgery. These stages are encoded as stage IDs 0, 1, 2, and 3, respectively. The public feature schema records, for each variable, its column name, clinical description, data type, temporal stage, and whether it belongs to the core input set used by the web prototype.

The input table uses one row per patient. Continuous variables remain continuous and categorical variables remain categorical. Missing measurements are retained as missing and are not replaced by clinically observed zero values. Labels are represented as two binary targets: a first-stage pancreatic leakage label (biochemical leak or Grade B/C POPF versus no leakage) and a CR-POPF label (Grade B/C versus all other patients). The latter is used for the second-stage grading task, with the second-stage loss restricted to leakage-positive patients during training.

## Training-only preprocessing of numerical and categorical variables

All preprocessing parameters are estimated from the training cohort only. For each continuous variable, the mean and standard deviation are calculated from non-missing training observations. Observed values are standardized as

z = (x - mean_train) / SD_train.

A standard deviation smaller than 1e-8 is replaced by 1 to avoid numerical instability. Missing continuous observations are stored internally as zero after standardization, while the corresponding observed-value mask remains zero; therefore, this internal tensor value is not interpreted as an observed clinical value by the Transformer.

Categorical variables are encoded using a training-derived vocabulary. Category identifiers are defined by variable-specific keys of the form `variable::category`, which prevents identically named categories from different clinical variables from sharing an unintended embedding. Identifier 0 is reserved for missing or previously unseen categories. The numerical normalization statistics and categorical vocabulary are serialized to `preprocessor.json` and reused unchanged during internal validation, external validation, and subsequent inference.

## Contextual encoding of heterogeneous clinical information

The model uses Bio-Clinical BERT (`emilyalsentzer/Bio_ClinicalBERT`) to encode the semantic identity of each clinical variable. The textual description associated with each variable in the feature schema is tokenized and passed through Bio-Clinical BERT. The contextual representation of the [CLS] token is used as the 768-dimensional semantic embedding for that variable. The first six Transformer encoder layers of Bio-Clinical BERT are frozen, while the remaining layers can be updated during model training.

Observed values are represented separately from the semantic embedding. A standardized continuous value is projected from one dimension to the 768-dimensional model space through a linear layer followed by GELU activation and layer normalization. Categorical values are mapped to learned 768-dimensional embeddings using the training-derived category vocabulary. A learned four-level temporal embedding represents the perioperative stage of each variable.

For variable i, let s_i denote the Bio-Clinical BERT semantic embedding, v_i the numerical or categorical value embedding, and t_i the temporal-stage embedding. A learned element-wise gate is computed as

g_i = sigmoid(W_g [s_i ; v_i ; t_i] + b_g),

and the fused representation is

e_i = LayerNorm(s_i + g_i ⊙ v_i + t_i).

This formulation allows the clinical meaning of a variable, its observed patient-specific value, and its temporal position to be represented in a common space without converting the entire patient record into an unordered tabular vector.

## Missing-input handling

An observed-value mask is generated for every patient-variable pair. Missing variables are replaced internally by a learned missing token for stable tensor construction and are excluded as keys and values from effective self-attention through the key-padding mask. The same mask is used during patient-level pooling so missing variables do not contribute as if they were observed measurements. Consequently, the proposed model can accept variable-length and incomplete input sets without retraining or center-specific imputation. Naturally occurring missingness in the study cohorts and controlled variable-removal experiments are analyzed separately in the manuscript.

## Context-aware gated Transformer

The fused variable embeddings are passed through four context-aware gated Transformer blocks with a model dimension of 768, eight attention heads, a feed-forward dimension of 3072, and dropout of 0.10. Each block contains multi-head self-attention followed by a feed-forward network. Learned gates regulate the contribution of both the attention update and the feed-forward update before residual addition and layer normalization. No causal look-ahead mask is applied because all variables supplied to the model are already restricted to information available by the 24-h assessment landmark. After the four Transformer blocks, a masked mean over observed variables produces a shared patient-level temporal representation.

## Hierarchical soft-sequential dual-task classifiers

The shared patient representation is passed to a primary multilayer perceptron with hidden dimensions 512, 256, and 128. Batch normalization, GELU activation, and dropout of 0.30 are applied within the classifier. The primary output is the probability of the broader pancreatic leakage phenotype.

The second-stage classifier receives the shared patient representation, the 128-dimensional hidden representation from the primary classifier, and the primary probability. A learned gate modulates the primary hidden representation before concatenation with the shared representation and primary probability. The resulting vector is processed by a second multilayer perceptron with hidden dimensions 256 and 128 and dropout of 0.30, producing the CR-POPF grading probability. This structure is soft-sequential because the second stage uses information from the first stage but remains differentiable and is trained jointly with the shared representation.

For transparency, the public implementation also returns the product of the first- and second-stage probabilities as an optional sensitivity output. The prespecified manuscript endpoint and the locked CR-POPF threshold are applied to the second-stage CR-POPF grading probability.

## Class imbalance and loss function

Task-specific class weights are estimated from the training cohort only. For the first-stage task, the positive-class weight is calculated as the number of no-leakage patients divided by the number of leakage-positive patients. For the second-stage task, the positive-class weight is calculated within leakage-positive patients as the number of biochemical leaks divided by the number of CR-POPF cases. Weighted binary cross-entropy with logits is then used for both tasks.

The second-stage severity loss is calculated only for leakage-positive patients. The total training objective is

L_total = lambda_1 L_leakage + lambda_2 L_CR,

with lambda_1 = 1.0 and lambda_2 = 1.0 in the reported configuration. This approach preserves the hierarchical clinical structure while preventing the more frequent first-stage class distribution from dominating optimization.

## Training procedure, model selection, and hyperparameters

Random seeds for Python-compatible numerical operations, NumPy, and PyTorch are fixed at 42, and deterministic cuDNN behavior is requested when CUDA is available. Model selection is confined to the designated training cohort. A stratified tuning split is created within the training cohort while preserving the three outcome strata of no leakage, biochemical leak, and CR-POPF. This tuning subset is used only to determine the early-stopping epoch and does not contain patients from the internal or external evaluation cohorts. After the stopping epoch is selected, preprocessing is refit on the complete training cohort, the model is reinitialized using the same seed, and training is repeated on the complete training cohort for the selected number of epochs.

The final configuration uses a maximum of 100 epochs, batch size 32, AdamW optimization, weight decay 0.01, and gradient clipping at a maximum norm of 1.0. The base learning rate for non-BERT parameters is 2e-5, whereas trainable Bio-Clinical BERT parameters use one tenth of the base rate. A ReduceLROnPlateau scheduler halves the learning rate after five epochs without improvement in tuning loss. Early stopping uses a patience of 15 epochs. The final architecture and training hyperparameters are stored in `config/model_config.yaml` and are reproduced in Supplementary Table S5.

## Decision-threshold selection

Decision thresholds are determined from the training cohort using the Youden index and are not optimized in internal or external evaluation data. The locked manuscript thresholds are 0.40 for the broader pancreatic leakage output and 0.31 for the CR-POPF grading output. The public repository includes an audit utility that can recompute Youden thresholds from training-cohort predictions; however, the released evaluation and inference scripts use the locked study thresholds and do not recalibrate them on evaluation data.

## Model locking before internal and external evaluation

After training is complete, the release bundle contains the final model weights, the fitted preprocessing state, feature schema, model configuration, task-specific class weights, training metadata, and decision thresholds. SHA-256 hashes are calculated for these files and written to a release manifest. Evaluation and inference routines first verify this manifest and then load the frozen artifacts in evaluation mode. No normalization statistics, categorical vocabularies, model parameters, class weights, or thresholds are updated during internal or external evaluation. This procedure provides a reproducible record of the model state that existed before evaluation of independent cohorts.

## Software environment and public implementation

The public implementation is written in Python 3.11 and PyTorch and uses the Hugging Face Transformers library for Bio-Clinical BERT. The GitHub release pins package versions in `requirements.txt` and provides a Conda environment definition in `environment.yml`. The principal pinned versions are PyTorch 2.3.1, Transformers 4.41.2, Tokenizers 0.19.1, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, and scikit-learn 1.5.0. The repository contains separate scripts for model training and locking, evaluation of locked cohorts, batch or single-cohort inference, threshold auditing, release-manifest verification, and the research web prototype. Real patient-level data are not distributed in the repository.

## References for reporting guidance

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.
2. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505. doi:10.1136/bmj-2024-082505.
