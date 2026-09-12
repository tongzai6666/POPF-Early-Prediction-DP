# Data format

Real patient-level data are not distributed in this repository.

The training/evaluation scripts expect one row per patient in a wide CSV. Required columns:

- `patient_id`
- `label_leakage`: 1 for BL or Grade B/C; 0 for no leakage
- `label_cr_popf`: 1 for Grade B/C; 0 otherwise
- feature columns defined in `config/feature_schema.yaml`

Feature columns may be missing for an individual patient. Missing values should be empty/NA, not zero-filled. Numeric normalization and categorical vocabularies are fit only on the training cohort and saved in the locked release bundle.
