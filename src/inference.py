"""
Inference and Submission Generation for Abusive Tamil Text Detection
"""

import os
import argparse
import zipfile
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from data_utils import load_data, AbusiveTextDataset, ID_TO_LABEL, NUM_LABELS


def predict(model_dir: str, test_path: str, output_dir: str, batch_size: int = 16):
    """
    Generate predictions for test data.

    Args:
        model_dir: Path to saved model directory
        test_path: Path to test CSV
        output_dir: Directory to save predictions
        batch_size: Batch size for inference
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load test data
    print(f"Loading test data from: {test_path}")
    test_df = load_data(test_path)
    print(f"Test samples: {len(test_df)}")

    # Create dataset (no labels)
    test_dataset = AbusiveTextDataset(
        texts=test_df["text"].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=128
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Predict
    all_preds = []
    all_probs = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Map predictions to labels
    pred_labels = [ID_TO_LABEL[p] for p in all_preds]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    submission_df = pd.DataFrame({
        "content": test_df["text"].tolist(),
        "labels": pred_labels
    })

    csv_path = os.path.join(output_dir, f"submission_{timestamp}.csv")
    submission_df.to_csv(csv_path, index=False)
    print(f"Submission CSV saved: {csv_path}")

    # Create zip file
    zip_path = os.path.join(output_dir, f"submission_{timestamp}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, os.path.basename(csv_path))
    print(f"Submission ZIP saved: {zip_path}")

    # Save detailed predictions with probabilities
    detailed_df = pd.DataFrame({
        "content": test_df["text"].tolist(),
        "predicted_label": pred_labels,
        "predicted_id": all_preds,
    })

    # Add probability columns for each class
    for i in range(NUM_LABELS):
        label_name = ID_TO_LABEL[i]
        detailed_df[f"prob_{label_name}"] = [p[i] for p in all_probs]

    detailed_path = os.path.join(output_dir, f"predictions_detailed_{timestamp}.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed predictions saved: {detailed_path}")

    # Print prediction distribution
    print("\nPrediction Distribution:")
    pred_counts = pd.Series(pred_labels).value_counts()
    for label, count in pred_counts.items():
        pct = count / len(pred_labels) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nSubmission files ready in: {output_dir}")
    return submission_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for abusive text detection")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    predict(args.model, args.test, args.output, args.batch_size)
