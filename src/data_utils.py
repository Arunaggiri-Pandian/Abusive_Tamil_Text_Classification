"""
Data utilities for Abusive Tamil Text Detection
- Data loading from CSV
- Label mapping (to be updated with task-specific labels)
- PyTorch Dataset class
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


# ============================================================================
# LABEL MAPPING - Abusive Tamil Text Detection (DravidianLangTech@ACL 2026)
# ============================================================================
LABEL_TO_ID = {
    "Non-Abusive": 0,
    "Abusive": 1,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_LABELS = len(LABEL_TO_ID)


def get_label_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return label mappings."""
    return LABEL_TO_ID, ID_TO_LABEL


def load_data(file_path: str, text_col: str = "Text", label_col: str = "Class") -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        text_col: Name of text column
        label_col: Name of label column

    Returns:
        DataFrame with text and label columns
    """
    df = pd.read_csv(file_path)

    # Validate columns exist
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV. Available: {df.columns.tolist()}")

    if label_col in df.columns:
        # Training/dev data with labels
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]

        # Map labels to IDs
        df["label_id"] = df["label"].map(LABEL_TO_ID)

        # Check for unmapped labels
        unmapped = df[df["label_id"].isna()]["label"].unique()
        if len(unmapped) > 0:
            print(f"WARNING: Unmapped labels found: {unmapped}")
            print("Please update LABEL_TO_ID in data_utils.py")
    else:
        # Test data without labels
        df = df[[text_col]].copy()
        df.columns = ["text"]
        df["label"] = None
        df["label_id"] = None

    return df


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    Uses inverse frequency weighting.

    Args:
        df: DataFrame with label_id column

    Returns:
        Tensor of class weights
    """
    label_counts = df["label_id"].value_counts().sort_index()
    total = len(df)

    weights = []
    for i in range(NUM_LABELS):
        count = label_counts.get(i, 1)
        weight = total / (NUM_LABELS * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(df: pd.DataFrame) -> List[float]:
    """
    Get per-sample weights for WeightedRandomSampler.

    Args:
        df: DataFrame with label_id column

    Returns:
        List of weights for each sample
    """
    class_weights = get_class_weights(df)
    sample_weights = [class_weights[int(label_id)].item() for label_id in df["label_id"]]
    return sample_weights


class AbusiveTextDataset(Dataset):
    """PyTorch Dataset for abusive text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            labels: List of label IDs (None for test data)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def analyze_data_distribution(df: pd.DataFrame, split_name: str = "data"):
    """Print label distribution analysis."""
    print(f"\n{'='*50}")
    print(f"Data Distribution Analysis: {split_name}")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")

    if "label" in df.columns and df["label"].notna().any():
        print("\nLabel distribution:")
        label_counts = df["label"].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"{'='*50}\n")
