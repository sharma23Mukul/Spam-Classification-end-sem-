"""
main.py
=======
Main entry point for the Probabilistic Spam Classification project.

Runs the COMPLETE analysis pipeline:
    1. Download dataset (if needed)
    2. Load and split data (Train / Val / Test)
    3. Preprocess text
    4. Train Naïve Bayes models (Multinomial, Bernoulli, Gaussian)
    5. Perform Hyperparameter Optimization (Grid Search)
    6. Evaluate on Test set
    7. Generate comprehensive comparison report and visualizations

Usage:
    python main.py
"""

import os
import sys

# Force UTF-8 output on Windows to avoid encoding errors with Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_dataset
from data.loader import load_all_data, train_val_test_split
from preprocessing.pipeline import (
    preprocess, preprocess_corpus, build_vocabulary,
    word_frequencies_by_class
)
from models.multinomial_nb import MultinomialNaiveBayes
from models.bernoulli_nb import BernoulliNaiveBayes
from models.gaussian_nb import GaussianNaiveBayes

from evaluation.metrics import classification_report
from evaluation.cross_validation import k_fold_cross_validation
from evaluation.hypothesis_testing import (
    confidence_interval_95, paired_t_test, mcnemar_test
)

# New Comparison Report module
import comparison_report

def print_header(text):
    """Print a formatted section header."""
    width = 60
    print(f"\n{'╔' + '═'*width + '╗'}")
    print(f"{'║'} {text:^{width-1}}{'║'}")
    print(f"{'╚' + '═'*width + '╝'}")


def main():
    """Run the complete spam classification analysis pipeline."""

    print_header("PROBABILISTIC SPAM CLASSIFICATION")
    print_header("Comparative Analysis of Naïve Bayes Variants")

    # ──────────────────────────────────────────────────────────────
    # STEP 1: Download and Load Data
    # ──────────────────────────────────────────────────────────────
    print_header("Step 1: Loading Datasets")
    download_dataset()
    data = load_all_data()

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Train/Val/Test Split (Stratified)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 2: Train/Validation/Test Split")
    train_data, val_data, test_data = train_val_test_split(data, val_ratio=0.15, test_ratio=0.15, seed=42)

    # ──────────────────────────────────────────────────────────────
    # STEP 3: Preprocess Text
    # ──────────────────────────────────────────────────────────────
    print_header("Step 3: Preprocessing & Vocabulary Building")
    train_processed = preprocess_corpus(train_data)
    val_processed = preprocess_corpus(val_data)
    test_processed = preprocess_corpus(test_data)
    
    # Filter vocab using min_df and max_df
    vocabulary = build_vocabulary(train_processed, min_df=2, max_df=0.95)
    print(f"  Vocabulary size: {len(vocabulary)} unique words (filtered with min_df=2, max_df=0.95)")

    freq = word_frequencies_by_class(train_processed)

    # ──────────────────────────────────────────────────────────────
    # STEP 4: Hyperparameter Optimization (Grid Search)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 4: Hyperparameter Optimization (Grid Search)")
    
    # Optimize Multinomial NB
    best_multi_model, best_multi_alpha = comparison_report.grid_search_multinomial(
        train_processed, val_processed, vocabulary, alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    # Optimize Bernoulli NB
    best_bern_model, best_bern_alpha = comparison_report.grid_search_bernoulli(
        train_processed, val_processed, vocabulary, alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    # Train Gaussian NB (No alpha to tune really, just smoothing)
    print("\n--- Training Gaussian NB ---")
    gaussian = GaussianNaiveBayes(alpha=1.0)
    gaussian.fit(train_processed, vocabulary)

    # ──────────────────────────────────────────────────────────────
    # STEP 5: Confidence Threshold Tuning
    # ──────────────────────────────────────────────────────────────
    print_header("Step 5: Threshold Tuning on Best Model")
    # Let's say Multinomial usually performs best on text
    best_threshold = comparison_report.threshold_tuning(best_multi_model, val_processed)
    print(f"Optimal Threshold selected: {best_threshold:.4f}")

    # ──────────────────────────────────────────────────────────────
    # STEP 6: Final Evaluation on Test Set
    # ──────────────────────────────────────────────────────────────
    print_header("Step 6: Final Evaluation on Test Set")
    
    y_true_test = [l for l, _ in test_data]
    
    models = {
        'Multinomial NB': best_multi_model,
        'Bernoulli NB': best_bern_model,
        'Gaussian NB': gaussian
    }
    
    test_results = {}
    models_probs = {}
    
    for name, model in models.items():
        preds = []
        probs = []
        for _, tokens in test_processed:
            pred, p = model.predict(tokens)
            preds.append(pred)
            probs.append(p['spam'])
            
        models_probs[name] = probs
        report, metrics = classification_report(y_true_test, preds, y_probs=probs, model_name=name, positive='spam')
        test_results[name] = metrics
        print(report)

    # ──────────────────────────────────────────────────────────────
    # STEP 7: Cross Validation (Train + Val combined)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 7: 5-Fold Stratified Cross-Validation")
    
    # Combine Train and Val for CV
    cv_data = train_data + val_data
    
    cv_multi = k_fold_cross_validation(cv_data, MultinomialNaiveBayes, k=5, alpha=best_multi_alpha)
    cv_bern = k_fold_cross_validation(cv_data, BernoulliNaiveBayes, k=5, alpha=best_bern_alpha)
    # Cross val for Gaussian
    cv_gauss = k_fold_cross_validation(cv_data, GaussianNaiveBayes, k=5, alpha=1.0)
    
    cv_results = {
        'Multinomial NB': cv_multi['fold_accuracies'],
        'Bernoulli NB': cv_bern['fold_accuracies'],
        'Gaussian NB': cv_gauss['fold_accuracies']
    }

    # ──────────────────────────────────────────────────────────────
    # STEP 8: Hypothesis Testing
    # ──────────────────────────────────────────────────────────────
    print_header("Step 8: Hypothesis Testing")

    # Paired t-test
    paired_t_test(cv_multi['fold_accuracies'], cv_bern['fold_accuracies'], "Multinomial NB", "Bernoulli NB")
    
    # Get predictions for McNemar
    preds_multi = []
    preds_bern = []
    preds_gauss = []
    for _, tokens in test_processed:
        p_m, _ = best_multi_model.predict(tokens)
        p_b, _ = best_bern_model.predict(tokens)
        p_g, _ = gaussian.predict(tokens)
        preds_multi.append(p_m)
        preds_bern.append(p_b)
        preds_gauss.append(p_g)
        
    mcnemar_test(y_true_test, preds_multi, preds_bern, "Multinomial NB", "Bernoulli NB")
    mcnemar_test(y_true_test, preds_multi, preds_gauss, "Multinomial NB", "Gaussian NB")

    # ──────────────────────────────────────────────────────────────
    # STEP 9: Generate Visualizations & Output
    # ──────────────────────────────────────────────────────────────
    print_header("Step 9: Generating Comparative Visualizations")

    comparison_report.generate_metrics_bar_chart(test_results)
    comparison_report.generate_roc_curves(y_true_test, models_probs)
    comparison_report.generate_confusion_matrices(test_results)
    comparison_report.generate_cv_boxplot(cv_results)
    comparison_report.generate_summary_csv(test_results, cv_results)

    # ──────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────
    print_header("ANALYSIS COMPLETE")
    print(f"""
  Visualizations saved to outputs/
  - metrics_comparison.png
  - roc_curves.png
  - confusion_matrices.png
  - cv_boxplot.png
  - pr_curve.png
  - results_summary.csv
""")

if __name__ == "__main__":
    main()
