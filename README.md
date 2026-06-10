# POPF-Early-Prediction-DP
A Deep Learning Framework Using Temporal Perioperative Data for Early Prediction of Postoperative Pancreatic Fistula After Distal Pancreatectomy: A Multicenter Retrospective Study
A deep learning framework using temporal perioperative data for early prediction of postoperative pancreatic fistula after distal pancreatectomy.

## Overview

This repository contains the core implementation of the dual-task deep learning framework described in our multicenter retrospective study. The model integrates perioperative electronic health record (EHR) data from the preoperative period to 24 hours postoperatively to predict:

1. **Overall POPF occurrence** (Grade A/B/C vs. no POPF)
2. **Severity stratification** among positive cases (BL vs. CR-POPF)

The framework is built upon **Bio-Clinical BERT** with a **context-aware gated Transformer module** to capture temporal evolution and nonlinear interactions of clinical parameters.

## Architecture

```
Input (Temporal EHR Items)
    |
    ├──> Bio-Clinical BERT Encoder ──> Semantic Embeddings
    |
    ├──> Temporal Embedding (timestamp + variable name + value)
    |
    └──> Gated Transformer with Attention Mask ──> Contextualized Features
              |
              ├──> Primary MLP ──> POPF Probability (Task 1)
              |
              └──> Hierarchical Gated MLP ──> BL vs. CR-POPF (Task 2)
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.30
- See `requirements.txt` for full dependencies

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Early-POPF-Prediction-DP.git
cd Early-POPF-Prediction-DP
pip install -r requirements.txt
```

## Data Format

The model expects structured perioperative EHR data with the following fields for each patient:

| Field | Description | Example |
|-------|-------------|---------|
| `patient_id` | Unique patient identifier | `P001` |
| `timestamp` | Time of data collection (relative to surgery) | `preop`, `intraop`, `postop_24h` |
| `variable_name` | Clinical variable name | `DFA`, `PT`, `CT_pancreas`, `PNI` |
| `value` | Numerical or categorical value | `2984`, `14.8`, `42.3` |
| `center_id` | Hospital center identifier (1-6) | `1` |

**Note**: Due to patient privacy regulations and institutional review board restrictions, the original clinical dataset cannot be publicly shared. Researchers may request access through the corresponding author with appropriate ethical approvals.

## Model Input Specification

### Temporal Variables (within 24h post-surgery)

| Category | Variables |
|----------|-----------|
| **Baseline** | age, BMI, sex, hypertension, diabetes |
| **Preoperative Imaging** | MPDD, PT, CT attenuation (pancreas/spleen/liver/psoas), P/S, P/L, P/PM |
| **Preoperative Lab** | CBC, albumin, CRP, PNI, NLR, CAR |
| **Intraoperative** | surgical approach, operative time, blood loss, transfusion, stump management |
| **Postoperative 24h** | DFA, CBC, albumin, CRP, PNI, NLR, CAR |

### Target Labels

- `label_popf`: 0 = no POPF, 1 = POPF (Grade A/B/C)
- `label_severity`: 0 = BL (Grade A), 1 = CR-POPF (Grade B/C) [only for POPF-positive cases]

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/training/data \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-5
```

### Inference

```bash
python predict.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path /path/to/test/data.csv \
    --output_path ./predictions.csv
```

### Example Output

```python
{
    'patient_id': 'P001',
    'popf_probability': 0.847,        # Overall POPF risk
    'severity_probability': 0.723,     # CR-POPF probability (given POPF-positive)
    'risk_stratum': 'high-risk'        # Based on threshold 0.4875 (Youden index)
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_name_2026_popf,
  title={A Deep Learning Framework Using Temporal Perioperative Data for Early Prediction of Postoperative Pancreatic Fistula After Distal Pancreatectomy: A Multicenter Retrospective Study},
  journal={},
  year={2026},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions regarding the code or data access requests, please contact the first author.

