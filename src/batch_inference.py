"""
Batch Inference for Multi-Run Submission
Generates predictions from multiple models for DravidianLangTech@ACL 2026
"""

import os
import json
import argparse
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
from tqdm import tqdm

from data_utils import load_data, AbusiveTextDataset, ID_TO_LABEL, NUM_LABELS


# Model configurations: (model_dir, base_model_name, max_length, dev_f1)
MODELS = [
    ("outputs/models/muril_v2", "google/muril-base-cased", 256, 0.8276),
    ("outputs/models/muril_base", "google/muril-base-cased", 128, 0.8250),
    ("outputs/models/muril_tuned", "google/muril-base-cased", 128, 0.8245),
    ("outputs/models/muril_tuned_ga", "google/muril-base-cased", 128, 0.8223),
    ("outputs/models/xlm_roberta", "xlm-roberta-base", 256, 0.8195),
]


def load_model_safe(model_dir: str, base_model: str, device: torch.device):
    """
    Load model with proper handling of config.json issues.
    Some training scripts overwrite config.json with training config instead of model config.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Try loading directly first
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print(f"  Loaded model directly from {model_dir}")
    except Exception as e:
        print(f"  Direct load failed, using base model config: {e}")

        # Load base model config and then weights
        config = AutoConfig.from_pretrained(base_model, num_labels=NUM_LABELS)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, config=config, ignore_mismatched_sizes=True
        )

        # Load trained weights
        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)

            # Fix LayerNorm key naming (gamma/beta -> weight/bias)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict)
            print(f"  Loaded weights from {weights_path}")
        else:
            # Try pytorch_model.bin
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"  Loaded weights from {weights_path}")

    model.to(device)
    model.eval()
    return model, tokenizer


def predict_single_model(
    model_dir: str,
    base_model: str,
    max_length: int,
    test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 16
) -> tuple:
    """Generate predictions for a single model."""

    print(f"\nLoading model: {model_dir}")
    model, tokenizer = load_model_safe(model_dir, base_model, device)

    # Create dataset
    test_dataset = AbusiveTextDataset(
        texts=test_df["text"].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Predict
    all_preds = []
    all_probs = []

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

    return pred_labels, all_probs


def main(test_path: str, output_dir: str, team_name: str, batch_size: int = 16):
    """Generate predictions for all models and create submission files."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = load_data(test_path)
    print(f"Test samples: {len(test_df)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions for each model
    results = []

    for run_num, (model_dir, base_model, max_length, dev_f1) in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"Run {run_num}: {os.path.basename(model_dir)} (Dev F1: {dev_f1:.4f})")
        print(f"{'='*60}")

        pred_labels, all_probs = predict_single_model(
            model_dir, base_model, max_length, test_df, device, batch_size
        )

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            "Text": test_df["text"].tolist(),
            "Class": pred_labels
        })

        # Save CSV
        csv_name = f"{team_name}_run{run_num}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        submission_df.to_csv(csv_path, index=False)

        # Create ZIP
        zip_name = f"{team_name}_AbusiveTamil_run{run_num}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, csv_name)

        # Print distribution
        pred_counts = pd.Series(pred_labels).value_counts()
        print(f"\nPrediction Distribution (Run {run_num}):")
        for label, count in pred_counts.items():
            pct = count / len(pred_labels) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        results.append({
            "run": run_num,
            "model": os.path.basename(model_dir),
            "dev_f1": dev_f1,
            "csv": csv_path,
            "zip": zip_path,
            "distribution": pred_counts.to_dict()
        })

        print(f"✓ Saved: {zip_name}")

    # Summary
    print(f"\n{'='*60}")
    print("MULTI-RUN SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"Team: {team_name}")
    print(f"Output directory: {output_dir}")
    print(f"\nRuns generated:")
    for r in results:
        print(f"  Run {r['run']}: {r['model']} (Dev F1: {r['dev_f1']:.4f})")
        print(f"         → {os.path.basename(r['zip'])}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{team_name}_submission_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for multi-run submission")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--team", type=str, default="CHMOD_777", help="Team name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    main(args.test, args.output, args.team, args.batch_size)
