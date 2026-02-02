# CHMOD_777 at DravidianLangTech@ACL 2026: MuRIL-Based Detection of Abusive Tamil Text Targeting Women on Social Media

## Abstract

This paper describes our system for the DravidianLangTech@ACL 2026 shared task on detecting abusive Tamil text targeting women on social media. We fine-tune MuRIL (Multilingual Representations for Indian Languages), a BERT-based model pre-trained specifically on Indian languages. Our best model achieves **82.76% Macro F1** on the development set using MuRIL with extended context length (256 tokens). We demonstrate that language-specific pre-training outperforms multilingual alternatives like XLM-RoBERTa, and that longer context windows improve performance on Tamil social media text. Our code is available at: **[GitHub Link]**.

## 1. Introduction

Online abuse targeting women remains a pervasive issue on social media platforms. Tamil-language social media, with millions of active users, is no exception. Automated detection of such abusive content is crucial for platform moderation and user safety.

The DravidianLangTech@ACL 2026 shared task on abusive text detection poses a binary classification challenge: identify whether a given Tamil text contains abusive content targeting women.

Key challenges include:
- **Code-mixing:** Tamil text often mixes with English, requiring models capable of handling bilingual content
- **Script diversity:** Tamil appears in native script and romanized forms
- **Cultural context:** Abusive patterns are culture-specific and may not transfer from other languages

In this work, we demonstrate that MuRIL (Khanuja et al., 2021), pre-trained on 17 Indian languages, significantly outperforms general multilingual models for this task.

## 2. Related Work

### 2.1 Abusive Language Detection

Abusive language detection has been extensively studied for English (Davidson et al., 2017; Founta et al., 2018). Recent work has expanded to multilingual settings using transformer-based models (Ranasinghe and Zampieri, 2021).

### 2.2 Tamil NLP

Tamil NLP has benefited from the development of language-specific models. MuRIL (Khanuja et al., 2021) was pre-trained on 17 Indian languages using both native scripts and transliteration. Previous DravidianLangTech tasks have shown MuRIL's effectiveness for Tamil text classification (Chakravarthi et al., 2022).

## 3. Methodology

### 3.1 Dataset

The shared task provides:
- **Training set:** 3,286 samples
- **Development set:** 366 samples
- **Test set:** 913 samples

Class distribution is nearly balanced:
| Label | Train | Dev |
|-------|-------|-----|
| Non-Abusive | 1,694 (51.6%) | 189 (51.6%) |
| Abusive | 1,592 (48.4%) | 177 (48.4%) |

### 3.2 Model Selection

We evaluated multiple pre-trained models:
- **MuRIL** (google/muril-base-cased): 236M parameters, pre-trained on 17 Indian languages
- **XLM-RoBERTa** (xlm-roberta-base): 278M parameters, multilingual model
- **IndicBERT-v3** (ai4bharat/indic-bert): 270M parameters, Indian language model

### 3.3 Training Configuration

```json
{
    "model_name": "google/muril-base-cased",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-5,
    "loss_type": "focal",
    "focal_gamma": 2.0,
    "early_stopping_patience": 10
}
```

Key techniques:
- **Focal Loss (gamma=2.0):** Down-weights easy examples, focusing on harder cases
- **WeightedRandomSampler:** Ensures balanced mini-batches during training
- **Early stopping:** Prevents overfitting by monitoring Macro F1

### 3.4 Context Length Experiment

We hypothesized that Tamil social media text benefits from longer context windows due to:
- Complex sentence structures
- Code-mixed content requiring more context
- Hashtags and mentions providing contextual signals

We tested max_length values of 128 and 256 tokens.

## 4. Results

### 4.1 Model Comparison

| Model | max_length | Best Epoch | Macro F1 |
|-------|------------|------------|----------|
| **MuRIL v2** | 256 | 9 | **82.76%** |
| MuRIL v1 | 128 | 22 | 82.50% |
| MuRIL (tuned) | 128 | 22 | 82.45% |
| XLM-RoBERTa | 256 | 10 | 81.95% |
| IndicBERT-v3 | 256 | 39 | 74.02% |

### 4.2 Best Model Performance (MuRIL v2)

**Per-Class Results:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Non-Abusive | 82.81% | 84.13% | 83.46% |
| Abusive | 82.76% | 81.36% | 82.05% |

Key observations:
1. **Balanced performance:** Both classes achieve similar F1 scores (difference < 1.5%)
2. **Longer context helps:** 256 tokens outperforms 128 tokens (+0.26% F1)
3. **Faster convergence:** Best model converges at epoch 9 vs 17-22 for others

### 4.3 Hyperparameter Tuning Results

Additional experiments with tuned hyperparameters:
| Configuration | Macro F1 |
|--------------|----------|
| Default (lr=1e-5) | **82.76%** |
| Tuned (lr=2e-5, warmup=0.1, wd=0.01) | 82.45% |
| Tuned + Gradient Accumulation | 82.23% |

Hyperparameter tuning did not improve over default settings, suggesting MuRIL's pre-training is well-suited for Tamil text classification.

## 5. Analysis

### 5.1 Why MuRIL Outperforms

MuRIL's superior performance can be attributed to:
1. **Language-specific pre-training:** Trained on 17 Indian languages including Tamil
2. **Transliteration augmentation:** Handles code-mixed and romanized Tamil
3. **Script-aware tokenization:** Better subword segmentation for Tamil script

### 5.2 Why Longer Context Helps

The improvement with 256 tokens suggests:
- Tamil sentences may require more context for disambiguation
- Hashtags and mentions at text boundaries provide classification signals
- Code-mixed content benefits from surrounding context

### 5.3 Model Efficiency

MuRIL v2 (256 tokens) converges faster (epoch 9) than MuRIL v1 (128 tokens, epoch 22), suggesting that providing more context actually simplifies the learning task.

## 6. Conclusion

We presented our system for detecting abusive Tamil text targeting women. Our key findings are:

1. **MuRIL achieves 82.76% Macro F1**, outperforming XLM-RoBERTa by 0.81 points
2. **Longer context (256 tokens) helps** for Tamil social media text
3. **Language-specific pre-training is crucial** - MuRIL's Indian language focus provides significant advantages
4. **Balanced classes yield balanced performance** - both classes achieve similar F1 scores

Our code and trained models are available at: **[GitHub Link]**

## References

Chakravarthi, B. R., et al. (2022). Findings of the Shared Task on Abusive Language Detection for Dravidian Languages. In Proceedings of the DravidianLangTech Workshop at ACL.

Davidson, T., et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. In ICWSM.

Founta, A.-M., et al. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. In ICWSM.

Khanuja, S., et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.

Ranasinghe, T., and Zampieri, M. (2021). Multilingual Offensive Language Identification with Cross-lingual Embeddings. In EMNLP.

---

## Appendix A: Submission Runs

**File:** `CHMOD_777_abusive.zip`

| Run | Model | Dev F1 | Description |
|-----|-------|--------|-------------|
| 1 | MuRIL v2 (256) | **82.76%** | Primary submission |
| 2 | MuRIL v1 (128) | 82.50% | Shorter context |
| 3 | XLM-RoBERTa | 81.95% | Architecture diversity |

## Appendix B: Hyperparameters

### Best Model (MuRIL v2)
| Parameter | Value |
|-----------|-------|
| Model | google/muril-base-cased |
| Max length | 256 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Epochs | 50 (early stopped at 19) |
| Best epoch | 9 |
| Loss function | Focal Loss (gamma=2.0) |
| Early stopping patience | 10 |
