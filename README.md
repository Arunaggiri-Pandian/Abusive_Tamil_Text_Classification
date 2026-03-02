# Abusive Tamil Text Detection - CHMOD_777

**DravidianLangTech@ACL 2026 Shared Task**
- **Competition:** [Codabench #11326](https://www.codabench.org/competitions/11326/)
- **Team:** **CHMOD_777**
  - Arunaggiri Pandian Karunanidhi (Micron Technology)
  - Prabalakshmi Arumugam (Boise State University)

This repository contains our code for detecting abusive Tamil text targeting women on social media.

## Results

| Model | Dev Macro F1 |
|-------|--------------|
| **MuRIL v2 (256)** | **82.76%** |
| MuRIL v1 (128) | 82.50% |
| XLM-RoBERTa | 81.95% |
| IndicBERT-v3 | 74.02% |

### Key Findings

1. **MuRIL outperforms multilingual models:** Language-specific pre-training on Indian languages is crucial
2. **Longer context helps:** 256 tokens > 128 tokens for Tamil social media text
3. **Balanced performance:** Both Abusive (82.05% F1) and Non-Abusive (83.46% F1) classes perform well

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- CUDA 11.8+ (optional, for GPU training)

## Project Structure

```
abusive_tamil_text_detection/
├── src/
│   ├── train_transformer.py   # Model training
│   ├── inference.py           # Single-model inference
│   ├── batch_inference.py     # Batch prediction generation
│   ├── analyze.py             # Model analysis & visualization
│   └── data_utils.py          # Data loading utilities
├── configs/                   # Model configurations
├── data/                      # Dataset (not included)
├── outputs/
│   ├── models/               # Trained models
│   └── predictions/          # Submission files
├── requirements.txt
├── run.sh                    # Convenience script
├── paper.md                  # Paper draft
└── README.md
```

## Usage

### 1. Training

```bash
# Train MuRIL (best model)
./run.sh train-transformer --config configs/muril_base.json

# Or with custom parameters
python src/train_transformer.py \
    --model_name google/muril-base-cased \
    --data_dir data \
    --output_dir outputs/models/muril_v2 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-5 \
    --focal_gamma 2.0 \
    --early_stopping_patience 10
```

### 2. Inference

```bash
# Generate predictions
python src/batch_inference.py \
    --model_dir outputs/models/muril_v2 \
    --test_file data/test.csv \
    --output_file outputs/predictions/run1.csv
```

### 3. Analysis

```bash
# Analyze model performance
./run.sh analyze-model --model outputs/models/muril_v2 --data data/dev.csv
```

## Models Used

- **MuRIL** (google/muril-base-cased): 236M parameters, pre-trained on 17 Indian languages
- **XLM-RoBERTa** (xlm-roberta-base): 278M parameters, multilingual model
- **IndicBERT-v3** (ai4bharat/indic-bert): 270M parameters

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | google/muril-base-cased |
| Max length | 256 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Loss function | Focal Loss (gamma=2.0) |
| Epochs | 50 (early stopping) |
| Patience | 10 epochs |

<p align="center">Author: Arunaggiri Pandian Karunanidhi</p>
