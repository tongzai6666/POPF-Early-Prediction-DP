# Reporting notes for the public model release

The manuscript should be reported using TRIPOD+AI (BMJ 2024;385:e078378), which supersedes TRIPOD 2015 for prediction-model studies using regression or machine learning. PROBAST+AI (BMJ 2025;388:e082505) can be used as a complementary framework to evaluate quality, risk of bias, and applicability.

Repository elements supporting transparent reporting include:

- explicit participant/cohort partitioning in the manuscript;
- four-stage predictor timing in `config/feature_schema.yaml`;
- training-only preprocessing in `data/preprocessing.py`;
- explicit missing-input handling and attention masking;
- task-specific class weighting derived only from the training cohort;
- complete architecture and final hyperparameters in `config/model_config.yaml`;
- training-only early-stopping epoch selection and full-training refit in `train.py`;
- locked thresholds and SHA-256 release manifest before evaluation;
- separate `evaluate.py` and `predict.py` paths that do not refit preprocessing or thresholds;
- pinned dependency versions;
- no real patient-level data included in the public repository.

The code release does not replace reporting of participant flow, outcome definitions, predictor measurement methods, model performance with uncertainty, calibration, clinical utility, subgroup analyses, or human-AI study design in the manuscript and supplementary materials.
