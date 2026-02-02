"""
Generate figures for Abusive Tamil Text Detection presentation.
No grid lines - values displayed above bars.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use seaborn-white style (no grid)
plt.style.use('seaborn-v0_8-white')
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'accent': '#FF9800',
    'danger': '#F44336',
    'purple': '#9C27B0',
    'teal': '#009688',
}


def plot_model_comparison():
    """Model comparison bar chart."""
    models = ['MuRIL v2\n(256)', 'MuRIL v1\n(128)', 'MuRIL\n(tuned)', 'XLM-RoBERTa', 'IndicBERT-v3']
    scores = [82.76, 82.50, 82.45, 81.95, 74.02]
    colors = [COLORS['primary'] if s == max(scores) else COLORS['secondary'] for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Macro F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Abusive Tamil Detection', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(70, 86)
    ax.grid(False)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: model_comparison.png")


def plot_per_class_f1():
    """Per-class F1 scores."""
    classes = ['Non-Abusive', 'Abusive']
    f1_scores = [83.46, 82.05]
    colors = [COLORS['secondary'], COLORS['danger']]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Scores (MuRIL v2)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(75, 88)
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_f1.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: per_class_f1.png")


def plot_context_length_comparison():
    """Context length impact."""
    configs = ['128 tokens', '256 tokens']
    scores = [82.50, 82.76]
    epochs = [22, 9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # F1 Score comparison
    colors = [COLORS['secondary'], COLORS['primary']]
    bars1 = ax1.bar(configs, scores, color=colors, edgecolor='white', linewidth=2)
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Macro F1 Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score by Context Length', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(81, 84)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Convergence comparison
    bars2 = ax2.bar(configs, epochs, color=colors, edgecolor='white', linewidth=2)
    for bar, epoch in zip(bars2, epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{epoch}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Best Epoch', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Speed', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 28)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'context_length_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: context_length_comparison.png")


def plot_dataset_distribution():
    """Dataset class distribution."""
    labels = ['Non-Abusive', 'Abusive']
    train = [1694, 1592]
    dev = [189, 177]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, train, width, label='Train', color=COLORS['primary'], edgecolor='white')
    bars2 = ax.bar(x + width/2, dev, width, label='Dev', color=COLORS['accent'], edgecolor='white')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: dataset_distribution.png")


def plot_precision_recall():
    """Precision and Recall comparison."""
    classes = ['Non-Abusive', 'Abusive']
    precision = [82.81, 82.76]
    recall = [84.13, 81.36]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', color=COLORS['primary'], edgecolor='white')
    bars2 = ax.bar(x + width/2, recall, width, label='Recall', color=COLORS['secondary'], edgecolor='white')

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall (MuRIL v2)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(78, 88)
    ax.legend()
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'precision_recall.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: precision_recall.png")


def plot_confusion_matrix():
    """Confusion matrix heatmap."""
    # Based on results: Non-Abusive (189 samples), Abusive (177 samples)
    # Non-Abusive: 84.13% recall -> 159 correct, 30 wrong
    # Abusive: 81.36% recall -> 144 correct, 33 wrong
    cm = np.array([
        [159, 30],   # Non-Abusive: 159 correct, 30 predicted as Abusive
        [33, 144],   # Abusive: 33 predicted as Non-Abusive, 144 correct
    ])

    labels = ['Non-Abusive', 'Abusive']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontweight='bold')

    # Add labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (MuRIL v2)', fontsize=14, fontweight='bold', pad=20)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="white" if cm[i, j] > 100 else "black",
                          fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: confusion_matrix.png")


if __name__ == "__main__":
    print("Generating figures for Abusive Tamil Text Detection...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    plot_model_comparison()
    plot_per_class_f1()
    plot_context_length_comparison()
    plot_dataset_distribution()
    plot_precision_recall()
    plot_confusion_matrix()

    print()
    print("All figures generated successfully!")
