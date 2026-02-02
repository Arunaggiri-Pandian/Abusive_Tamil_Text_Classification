"""
Model Analysis Tools for Abusive Tamil Text Detection
- Confusion matrix visualization
- Per-class metrics
- Error analysis
"""

import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_utils import load_data, AbusiveTextDataset, ID_TO_LABEL, NUM_LABELS


def analyze_model(
    model_dir: str,
    data_path: str,
    output_dir: str,
    split_name: str = "dev"
):
    """
    Comprehensive model analysis with confusion matrix and metrics.

    Args:
        model_dir: Path to saved model
        data_path: Path to labeled data CSV
        output_dir: Directory to save analysis
        split_name: Name of the split (train/dev/test)
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"Samples: {len(df)}")

    # Create dataset
    dataset = AbusiveTextDataset(
        texts=df["text"].tolist(),
        labels=df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Get predictions
    all_preds = []
    all_labels = []
    all_probs = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(output_dir, f"analysis_{split_name}_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    # Calculate metrics
    target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4
    )
    print(f"\nClassification Report:\n{report}")

    report_path = os.path.join(analysis_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Analysis: {split_name}\n")
        f.write(f"Model: {model_dir}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Samples: {len(df)}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f'Confusion Matrix - {split_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    cm_path = os.path.join(analysis_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f'Normalized Confusion Matrix - {split_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    cm_norm_path = os.path.join(analysis_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=150)
    plt.close()
    print(f"Normalized confusion matrix saved: {cm_norm_path}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    metrics_df = pd.DataFrame({
        "class": target_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })

    metrics_path = os.path.join(analysis_dir, "per_class_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Per-class metrics saved: {metrics_path}")

    # Overall metrics
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nOverall Metrics:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    # Error analysis - save misclassified samples
    errors = []
    for i, (true_label, pred_label, text) in enumerate(zip(all_labels, all_preds, df["text"])):
        if true_label != pred_label:
            errors.append({
                "text": text,
                "true_label": ID_TO_LABEL[true_label],
                "predicted_label": ID_TO_LABEL[pred_label],
                "confidence": max(all_probs[i])
            })

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df = errors_df.sort_values("confidence", ascending=False)

        errors_path = os.path.join(analysis_dir, "misclassified_samples.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Misclassified samples saved: {errors_path}")
        print(f"Total errors: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")

    print(f"\nAnalysis complete! Results saved to: {analysis_dir}")

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "analysis_dir": analysis_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model performance")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled data CSV")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--split", type=str, default="dev", help="Split name (train/dev/test)")

    args = parser.parse_args()
    analyze_model(args.model, args.data, args.output, args.split)
