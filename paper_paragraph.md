# CHMOD_777 at DravidianLangTech@ACL 2026: MuRIL-Based Detection of Abusive Tamil Text Targeting Women on Social Media

## Abstract

This paper describes our system for the DravidianLangTech@ACL 2026 shared task on detecting abusive Tamil text targeting women on social media. We fine-tune MuRIL (Multilingual Representations for Indian Languages), a BERT-based model pre-trained specifically on Indian languages. Our best model achieves **82.76% Macro F1** on the development set using MuRIL with extended context length (256 tokens). We demonstrate that language-specific pre-training outperforms multilingual alternatives like XLM-RoBERTa, and that longer context windows improve performance on Tamil social media text. Our code is available at: **[GitHub Link]**.

## 1. Introduction

Online abuse targeting women remains a pervasive issue on social media platforms. Tamil-language social media, with millions of active users, is no exception to this troubling trend. Automated detection of such abusive content is crucial for platform moderation and user safety, enabling rapid response to harmful content at scale.

The DravidianLangTech@ACL 2026 shared task on abusive text detection poses a binary classification challenge: identify whether a given Tamil text contains abusive content targeting women. This task presents several unique challenges that distinguish it from similar tasks in English or other high-resource languages. Code-mixing is prevalent, with Tamil text frequently incorporating English words and phrases, requiring models capable of handling bilingual content effectively. Script diversity adds complexity, as Tamil appears both in its native script and in romanized forms. Additionally, abusive patterns are often culture-specific and may not transfer reliably from models trained on other languages.

In this work, we demonstrate that MuRIL (Khanuja et al., 2021), pre-trained on 17 Indian languages, significantly outperforms general multilingual models for this task, validating the importance of language-specific pre-training for Indian language NLP applications.

## 2. Related Work

### 2.1 Abusive Language Detection

Abusive language detection has been extensively studied for English (Davidson et al., 2017; Founta et al., 2018), with research focusing on distinguishing various forms of harmful content including hate speech, harassment, and offensive language. Recent work has expanded to multilingual settings using transformer-based models (Ranasinghe and Zampieri, 2021), though performance on low-resource languages often lags behind English benchmarks.

### 2.2 Tamil NLP

Tamil NLP has benefited substantially from the development of language-specific models in recent years. MuRIL (Khanuja et al., 2021) represents a significant advancement, having been pre-trained on 17 Indian languages using both native scripts and transliteration. Previous DravidianLangTech tasks have consistently demonstrated MuRIL's effectiveness for Tamil text classification (Chakravarthi et al., 2022), establishing it as a strong baseline for Indian language tasks.

## 3. Methodology

### 3.1 Dataset

The shared task provides a dataset of 3,286 training samples, 366 development samples, and 913 test samples. Notably, the class distribution is nearly balanced, with Non-Abusive comprising 51.6% and Abusive comprising 48.4% of samples. This balance simplifies the classification task compared to many real-world scenarios where abusive content is typically a minority class.

### 3.2 Model Selection

We evaluated multiple pre-trained models representing different pre-training strategies and architectural choices. MuRIL (google/muril-base-cased) with 236M parameters was pre-trained specifically on 17 Indian languages and served as our primary model. XLM-RoBERTa (xlm-roberta-base) with 278M parameters represents a general multilingual approach trained on 100+ languages. IndicBERT-v3 (ai4bharat/indic-bert) with 270M parameters provides another Indian language-focused alternative for comparison.

### 3.3 Training Configuration

Our training configuration employs several techniques to optimize performance. We use Focal Loss with gamma=2.0 to down-weight easy examples and focus training on harder cases. WeightedRandomSampler ensures balanced mini-batches during training despite the near-balanced dataset, providing additional regularization. We set max_length to 256 tokens, use a batch size of 16, and train with a learning rate of 1e-5. Early stopping monitors Macro F1 on the validation set with patience of 10 epochs to prevent overfitting while allowing sufficient training time.

### 3.4 Context Length Experiment

We hypothesized that Tamil social media text benefits from longer context windows compared to the standard 128 tokens typically used in text classification. This hypothesis is motivated by several observations: Tamil exhibits complex sentence structures that may require more context for accurate interpretation, code-mixed content benefits from surrounding context to disambiguate mixed-language phrases, and hashtags and mentions often appear at text boundaries where they provide important contextual signals for classification. To test this hypothesis, we compared max_length values of 128 and 256 tokens.

## 4. Results

### 4.1 Model Comparison

Our experiments reveal clear patterns in model performance. MuRIL v2 with 256 token context achieves the best performance at 82.76% Macro F1, reaching its best checkpoint at epoch 9. MuRIL v1 with 128 token context achieves 82.50% F1 but requires training until epoch 22 to reach its best checkpoint. Additional hyperparameter tuning on MuRIL with 128 tokens yields 82.45% F1. XLM-RoBERTa with 256 tokens achieves 81.95% F1, demonstrating that language-specific pre-training provides measurable benefits over general multilingual models. IndicBERT-v3 with 256 tokens achieves only 74.02% F1, suggesting that not all Indian language models perform equally well on this task.

### 4.2 Best Model Performance

Our best model, MuRIL v2, achieves balanced performance across both classes. For Non-Abusive content, the model achieves 82.81% precision, 84.13% recall, and 83.46% F1. For Abusive content, performance is similarly strong with 82.76% precision, 81.36% recall, and 82.05% F1. The difference between class F1 scores is less than 1.5 percentage points, indicating the model handles both classes equally well without systematic bias.

Several observations emerge from these results. First, performance is remarkably balanced across classes, likely due to the balanced training data. Second, longer context demonstrably helps, with 256 tokens outperforming 128 tokens by 0.26% F1. Third, and perhaps most interestingly, the model with longer context converges faster, reaching its best checkpoint at epoch 9 compared to epochs 17-22 for shorter context models.

### 4.3 Hyperparameter Tuning Results

We conducted additional experiments with tuned hyperparameters to assess whether performance could be further improved. Our default configuration with learning rate 1e-5 achieves 82.76% F1. Tuned settings with learning rate 2e-5, warmup ratio 0.1, and weight decay 0.01 achieve 82.45% F1. Adding gradient accumulation to the tuned configuration yields 82.23% F1. These results suggest that hyperparameter tuning does not improve over our default settings, indicating that MuRIL's pre-training is already well-suited for Tamil text classification tasks.

## 5. Analysis

### 5.1 Why MuRIL Outperforms

MuRIL's superior performance over XLM-RoBERTa can be attributed to three factors. First, language-specific pre-training on 17 Indian languages including Tamil provides the model with rich representations of Tamil linguistic patterns. Second, MuRIL's pre-training includes transliteration augmentation, enabling it to handle both native Tamil script and romanized Tamil effectively—a crucial capability for code-mixed social media content. Third, MuRIL employs script-aware tokenization that produces better subword segmentation for Tamil script compared to generic multilingual tokenizers.

### 5.2 Why Longer Context Helps

The improvement from 128 to 256 tokens, while modest (0.26% F1), is consistent and accompanied by faster convergence. This suggests that Tamil sentences may require more context for disambiguation than is typically assumed. Additionally, hashtags and mentions often appear at text boundaries where they provide classification signals; longer contexts capture these elements more reliably. Code-mixed content particularly benefits from surrounding context that helps disambiguate the intended meaning of mixed-language phrases.

### 5.3 Model Efficiency

An interesting finding is that MuRIL v2 with 256 tokens converges at epoch 9, while MuRIL v1 with 128 tokens requires until epoch 22 to reach its best performance. This counterintuitive result suggests that providing more context actually simplifies the learning task, allowing the model to learn discriminative patterns more quickly. The additional context may reduce ambiguity in training examples, enabling more efficient parameter updates.

## 6. Conclusion

We presented our system for detecting abusive Tamil text targeting women at DravidianLangTech@ACL 2026. Our key findings demonstrate the importance of model selection and configuration choices for this task. MuRIL achieves 82.76% Macro F1, outperforming XLM-RoBERTa by 0.81 percentage points, confirming the value of language-specific pre-training. Longer context windows of 256 tokens provide measurable benefits for Tamil social media text, both in terms of final performance and training efficiency. Language-specific pre-training proves crucial, with MuRIL's Indian language focus providing significant advantages over general multilingual alternatives. Finally, the balanced class distribution in the training data yields balanced model performance, with both classes achieving similar F1 scores.

Our code and trained models are available at: **[GitHub Link]**

## References

Chakravarthi, B. R., et al. (2022). Findings of the Shared Task on Abusive Language Detection for Dravidian Languages. In Proceedings of the DravidianLangTech Workshop at ACL.

Davidson, T., et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. In ICWSM.

Founta, A.-M., et al. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. In ICWSM.

Khanuja, S., et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.

Ranasinghe, T., and Zampieri, M. (2021). Multilingual Offensive Language Identification with Cross-lingual Embeddings. In EMNLP.

---

## Appendix A: Submission Runs

We submitted three runs for the evaluation phase. Run 1 uses MuRIL v2 with 256 token context achieving 82.76% F1 as our primary submission. Run 2 employs MuRIL v1 with 128 token context achieving 82.50% F1 to assess the impact of shorter context. Run 3 uses XLM-RoBERTa achieving 81.95% F1 to provide architectural diversity in our submissions.

## Appendix B: Hyperparameters

Our best model configuration uses google/muril-base-cased with max length 256, batch size 16, and learning rate 1e-5 with weight decay 0.01. Training ran for 50 epochs with early stopping triggered at epoch 19, with the best checkpoint occurring at epoch 9. We employed Focal Loss with gamma=2.0 to focus training on harder examples.
