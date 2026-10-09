# POPF-Early-Prediction-DP

Reproducible implementation of a temporal dual-task framework for 24-hour postoperative risk stratification after distal pancreatectomy.

## Overview

The repository provides the implementation used for the temporal dual-task framework, including:

1. **Four perioperative stages**: baseline, preoperative, intraoperative, and postoperative 24 h.
2. **Two prespecified study thresholds**: 0.40 for the broader pancreatic leakage output and 0.31 for the final population-level CR-POPF output.
3. **Task-specific class-imbalance handling** using training-derived positive-class weights, with the severity loss evaluated only among leakage-positive patients.
4. **Heterogeneous clinical-data encoding** using training-only numerical standardization, variable-specific categorical vocabularies, Bio-Clinical BERT semantic embeddings, numerical/categorical value embeddings, temporal embeddings, gated fusion, missing-value masks, and a context-aware gated Transformer.
5. **Model locking and reproducibility controls** covering preprocessing state, model weights, thresholds, configuration, feature schema, class weights, and training metadata.
6. **Pinned software environments** in `requirements.txt` and `environment.yml`.

## Model outputs

The model returns:

- `pancreatic_leakage_probability = P(BL or Grade B/C)`
- `conditional_cr_popf_probability = P(CR-POPF | pancreatic leakage)`
- `cr_popf_probability = P(pancreatic leakage) × P(CR-POPF | pancreatic leakage)`, representing the final population-level CR-POPF probability
- `joint_cr_popf_probability`: backward-compatible alias of `cr_popf_probability`

Risk categories use the training-derived locked thresholds:

- pancreatic leakage: **0.40**
- CR-POPF: **0.31**

The CR-POPF threshold is applied to the final population-level `cr_popf_probability`.

## Four-stage feature organization

The feature schema is defined in `config/feature_schema.yaml` with stage IDs:

- 0: baseline
- 1: preoperative
- 2: intraoperative
- 3: postoperative 24 h

The schema is intentionally configurable so local EHR column names can be adapted without changing the model implementation.

## Installation

conda env create -f environment.yml conda activate popf-early-prediction

or

python -m venv .venv source .venv/bin/activate pip install -r requirements.txt

The first run requires access to the Hugging Face checkpoint `emilyalsentzer/Bio_ClinicalBERT`.

## Data format

One row per patient in wide CSV format. Required labels for training are:

- `label_leakage`: 1 = BL or Grade B/C, 0 = no leakage
- `label_cr_popf`: 1 = Grade B/C, 0 = otherwise

The first-stage task predicts pancreatic leakage in the full cohort. The second-stage severity loss is evaluated only among leakage-positive patients, where the model distinguishes BL from CR-POPF.

Real patient data are not included. See `data/README.md` and `examples/example_input.csv`.

## Training and locking

Only the designated training cohort should be supplied to `train.py`.

python train.py \   --train_csv /path/to/train_609.csv \   --output_dir ./release_bundle

The script creates a stratified tuning split *inside the training cohort* for early-stopping epoch selection, then refits preprocessing and model parameters on the complete training cohort. It writes a locked bundle containing the model, preprocessing state, thresholds, configuration, and SHA-256 manifest. Evaluation cohorts are not used in this process.

The first-stage output estimates `P(pancreatic leakage)`, whereas the second-stage output estimates `P(CR-POPF | pancreatic leakage)`. The final population-level CR-POPF probability is calculated as the product of these two probabilities.

## External/internal evaluation

python evaluate.py \   --bundle_dir ./release_bundle \   --data_csv /path/to/evaluation.csv \   --output_json ./results/evaluation.json

No preprocessing parameters, class weights, model weights, or thresholds are updated during evaluation.

For CR-POPF evaluation, `cr_popf_probability` represents the final population-level probability:

`P(CR-POPF) = P(pancreatic leakage) × P(CR-POPF | pancreatic leakage)`.

## Inference

python predict.py \   --bundle_dir ./release_bundle \   --data_csv /path/to/new_patients.csv \   --output_csv ./predictions.csv

The prediction output includes pancreatic leakage probability, conditional CR-POPF probability, and final population-level CR-POPF probability.

Incomplete patient inputs are allowed. Unavailable variables remain missing and are masked from effective Transformer attention; they are not converted into observed zero values.

## Web prototype

streamlit run app.py -- --bundle_dir ./release_bundle

The prototype supports core-variable or all-variable entry and simultaneously displays pancreatic leakage and final population-level CR-POPF probabilities. It is intended for research demonstration and not for autonomous clinical decision making.

## Reproducibility and reporting

Technical details are provided in `docs/Supplementary_Methods_1.md`. `docs/TRIPOD_AI_reporting_notes.md` maps the public implementation to reporting elements emphasized by TRIPOD+AI and PROBAST+AI.

The public implementation distinguishes between the second-stage conditional CR-POPF probability and the final population-level CR-POPF probability used for the reported CR-POPF prediction task.

## License

MIT License. See `LICENSE`.
