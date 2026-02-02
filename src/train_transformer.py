"""
Transformer Training for Abusive Tamil Text Detection

Features:
- Focal Loss for class imbalance
- WeightedRandomSampler for balanced batches
- Early stopping with patience
- Best model saved by Macro F1
- Classification report logging
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from data_utils import (
    load_data,
    get_sample_weights,
    AbusiveTextDataset,
    NUM_LABELS,
    ID_TO_LABEL,
    analyze_data_distribution
)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    gradient_accumulation_steps: int = 1
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss = loss / gradient_accumulation_steps  # Scale loss
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

    # Handle remaining gradients
    if (step + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate model and return loss, predictions, true labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def train(config_path: str, train_path: str, dev_path: str):
    """Main training function."""

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = load_data(train_path)
    dev_df = load_data(dev_path)

    analyze_data_distribution(train_df, "Train")
    analyze_data_distribution(dev_df, "Dev")

    # Load tokenizer and model
    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=NUM_LABELS
    )
    model.to(device)

    # Create datasets
    train_dataset = AbusiveTextDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )

    dev_dataset = AbusiveTextDataset(
        texts=dev_df["text"].tolist(),
        labels=dev_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )

    # WeightedRandomSampler for balanced batches
    sample_weights = get_sample_weights(train_df)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # Focal Loss
    criterion = FocalLoss(gamma=config["focal_gamma"])

    # Optimizer with weight decay
    weight_decay = config.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=weight_decay
    )

    # Gradient accumulation
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch={config['batch_size'] * gradient_accumulation_steps})")

    # Learning rate scheduler with warmup
    warmup_ratio = config.get("warmup_ratio", 0.0)
    # Adjust total steps for gradient accumulation
    total_steps = (len(train_loader) // gradient_accumulation_steps) * config["epochs"]
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = None
    if warmup_ratio > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Using linear scheduler with {warmup_steps} warmup steps (total: {total_steps})")

    # Create output directory
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Training loop with early stopping
    best_f1 = 0
    patience_counter = 0
    training_log = []

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Early stopping patience: {config['patience']}")

    for epoch in range(config["epochs"]):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*40}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler, gradient_accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        dev_loss, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)

        # Calculate metrics
        macro_f1 = f1_score(dev_labels, dev_preds, average="macro")

        print(f"Dev Loss: {dev_loss:.4f}")
        print(f"Dev Macro F1: {macro_f1:.4f}")

        # Log
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_macro_f1": macro_f1
        })

        # Check for improvement
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0

            # Save best model
            print(f"New best Macro F1: {macro_f1:.4f} - Saving model...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save classification report
            target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
            report = classification_report(
                dev_labels, dev_preds,
                target_names=target_names,
                digits=4
            )

            with open(os.path.join(output_dir, "best_classification_report.txt"), "w") as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Macro F1: {macro_f1:.4f}\n\n")
                f.write(report)

            print(f"\nClassification Report:\n{report}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")

            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    # Save training log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # Save config used
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer for abusive text detection")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev CSV")

    args = parser.parse_args()
    train(args.config, args.train, args.dev)
