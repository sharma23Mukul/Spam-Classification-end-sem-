"""
visualization/plots.py
======================
Visualization functions for the spam classification project.

All plots are saved to the 'outputs/' directory as PNG files.
Uses matplotlib and seaborn for charts, and wordcloud for word clouds.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _ensure_output_dir():
    """Create the outputs directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_word_frequency(spam_freq, ham_freq, top_n=20):
    """
    Plot the top-N most frequent words in spam vs ham messages.

    This visualization helps identify which words are most indicative
    of each class — the core of the Naïve Bayes model.

    Parameters
    ----------
    spam_freq : Counter
        Word frequency counter for spam messages.
    ham_freq : Counter
        Word frequency counter for ham messages.
    top_n : int
        Number of top words to display.
    """
    _ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top spam words
    spam_top = spam_freq.most_common(top_n)
    words_s, counts_s = zip(*spam_top) if spam_top else ([], [])
    axes[0].barh(range(len(words_s)), counts_s, color='#e74c3c', alpha=0.85)
    axes[0].set_yticks(range(len(words_s)))
    axes[0].set_yticklabels(words_s, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_title(f'Top {top_n} Words in SPAM Messages', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency', fontsize=12)

    # Top ham words
    ham_top = ham_freq.most_common(top_n)
    words_h, counts_h = zip(*ham_top) if ham_top else ([], [])
    axes[1].barh(range(len(words_h)), counts_h, color='#27ae60', alpha=0.85)
    axes[1].set_yticks(range(len(words_h)))
    axes[1].set_yticklabels(words_h, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_title(f'Top {top_n} Words in HAM Messages', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency', fontsize=12)

    plt.suptitle('Word Frequency Distribution: Spam vs Ham',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'word_frequency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    return path


def plot_word_clouds(spam_freq, ham_freq):
    """
    Generate word clouds for spam and ham messages.

    Word clouds visually represent word frequency — larger words
    appear more often by higher P(word | class).

    Parameters
    ----------
    spam_freq : Counter
        Word frequency counter for spam.
    ham_freq : Counter
        Word frequency counter for ham.
    """
    _ensure_output_dir()
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Spam word cloud
    if spam_freq:
        wc_spam = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds',
            max_words=100,
            random_state=42
        ).generate_from_frequencies(spam_freq)
        axes[0].imshow(wc_spam, interpolation='bilinear')
    axes[0].set_title('SPAM Word Cloud', fontsize=16, fontweight='bold', color='#e74c3c')
    axes[0].axis('off')

    # Ham word cloud
    if ham_freq:
        wc_ham = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            random_state=42
        ).generate_from_frequencies(ham_freq)
        axes[1].imshow(wc_ham, interpolation='bilinear')
    axes[1].set_title('HAM Word Cloud', fontsize=16, fontweight='bold', color='#27ae60')
    axes[1].axis('off')

    plt.suptitle('Word Clouds: Spam vs Ham', fontsize=18, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'word_clouds.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    paths.append(path)

    return paths


def plot_confusion_matrix(cm, title="Model", filename_suffix="model"):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : dict
        Confusion matrix dict with keys 'TP', 'TN', 'FP', 'FN'.
    title : str
        Model name for the plot title.
    filename_suffix : str
        Suffix for the output filename.
    """
    _ensure_output_dir()

    # Convert to 2D array: [[TN, FP], [FN, TP]]
    matrix = np.array([[cm['TN'], cm['FP']],
                        [cm['FN'], cm['TP']]])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                annot_kws={'size': 18},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'Confusion Matrix: {title}', fontsize=15, fontweight='bold')

    # Add annotations for TP/TN/FP/FN labels
    ax.text(0, -0.15, f"TN={cm['TN']}", ha='center', fontsize=10,
            color='green', transform=ax.transData)
    ax.text(1, -0.15, f"FP={cm['FP']}", ha='center', fontsize=10,
            color='red', transform=ax.transData)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{filename_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    return path


def plot_accuracy_comparison(metrics_multinomial, metrics_bernoulli):
    """
    Compare Multinomial and Bernoulli NB across all metrics.

    Creates a grouped bar chart showing accuracy, precision, recall,
    and F1 score for both models side by side.

    Parameters
    ----------
    metrics_multinomial : dict
        Metrics dict from classification_report for Multinomial NB.
    metrics_bernoulli : dict
        Metrics dict from classification_report for Bernoulli NB.
    """
    _ensure_output_dir()

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    keys = ['accuracy', 'precision', 'recall', 'f1_score']

    multi_values = [metrics_multinomial[k] for k in keys]
    bern_values = [metrics_bernoulli[k] for k in keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, multi_values, width,
                    label='Multinomial NB', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, bern_values, width,
                    label='Bernoulli NB', color='#e67e22', alpha=0.85)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Comparison: Multinomial vs Bernoulli Naïve Bayes',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    return path


def plot_error_distribution(y_true, y_pred_multi, y_pred_bern):
    """
    Plot error analysis: breakdown of error types for both models.

    Shows the count of:
    - True Positives, True Negatives (correct predictions)
    - False Positives (ham → spam), False Negatives (spam → ham)

    Parameters
    ----------
    y_true : list of str
        True labels.
    y_pred_multi : list of str
        Multinomial NB predictions.
    y_pred_bern : list of str
        Bernoulli NB predictions.
    """
    _ensure_output_dir()

    from evaluation.metrics import confusion_matrix

    cm_multi = confusion_matrix(y_true, y_pred_multi)
    cm_bern = confusion_matrix(y_true, y_pred_bern)

    categories = ['True Pos\n(Spam→Spam)', 'True Neg\n(Ham→Ham)',
                  'False Pos\n(Ham→Spam)', 'False Neg\n(Spam→Ham)']
    keys = ['TP', 'TN', 'FP', 'FN']
    colors = ['#27ae60', '#2ecc71', '#e74c3c', '#c0392b']

    multi_vals = [cm_multi[k] for k in keys]
    bern_vals = [cm_bern[k] for k in keys]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, multi_vals, width,
                    label='Multinomial NB', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, bern_vals, width,
                    label='Bernoulli NB', color='#e67e22', alpha=0.85)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Error Distribution: Prediction Breakdown',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    return path


def plot_cv_results(multi_accs, bern_accs):
    """
    Plot cross-validation accuracy across folds for both models.

    Parameters
    ----------
    multi_accs : list of float
        Fold accuracies for Multinomial NB.
    bern_accs : list of float
        Fold accuracies for Bernoulli NB.
    """
    _ensure_output_dir()

    folds = list(range(1, len(multi_accs) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(folds, multi_accs, 'o-', color='#3498db', linewidth=2,
            markersize=8, label=f'Multinomial (mean={np.mean(multi_accs):.4f})')
    ax.plot(folds, bern_accs, 's-', color='#e67e22', linewidth=2,
            markersize=8, label=f'Bernoulli (mean={np.mean(bern_accs):.4f})')

    ax.axhline(y=np.mean(multi_accs), color='#3498db', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(bern_accs), color='#e67e22', linestyle='--', alpha=0.5)

    ax.set_xlabel('Fold', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('5-Fold Cross Validation: Accuracy per Fold',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'cv_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
    return path
