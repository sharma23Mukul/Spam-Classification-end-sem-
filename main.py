"""
main.py
=======
Main entry point for the Probabilistic Spam Classification project.

Runs the COMPLETE analysis pipeline:
    1. Download dataset (if needed)
    2. Load and split data
    3. Preprocess text
    4. Train both Naïve Bayes models
    5. Evaluate with metrics and cross-validation
    6. Perform hypothesis testing
    7. Generate all visualizations
    8. Print comprehensive report with sample predictions

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
from data.loader import load_all_data, train_test_split
from preprocessing.pipeline import (
    preprocess, preprocess_corpus, build_vocabulary,
    word_frequencies_by_class
)
from models.multinomial_nb import MultinomialNaiveBayes
from models.bernoulli_nb import BernoulliNaiveBayes
from evaluation.metrics import classification_report, confusion_matrix
from evaluation.cross_validation import k_fold_cross_validation
from evaluation.hypothesis_testing import (
    confidence_interval_95, paired_t_test, mcnemar_test
)
from visualization.plots import (
    plot_word_frequency, plot_word_clouds,
    plot_confusion_matrix, plot_accuracy_comparison,
    plot_error_distribution, plot_cv_results
)


def print_header(text):
    """Print a formatted section header."""
    width = 60
    print(f"\n{'╔' + '═'*width + '╗'}")
    print(f"{'║'} {text:^{width-1}}{'║'}")
    print(f"{'╚' + '═'*width + '╝'}")


def print_sample_predictions(model, model_name, n=5):
    """
    Show sample predictions with probability breakdowns.

    This demonstrates Bayes' Theorem in action:
        P(Spam | message) vs P(Ham | message)
    """
    test_messages = [
        "Congratulations! You've won a free iPhone! Call now to claim your prize!",
        "Hey, are you coming to class tomorrow?",
        "URGENT! Your account has been compromised. Click here immediately!",
        "Can you pick up some milk on your way home?",
        "Win cash prizes! Text WIN to 80085 now! Free entry!",
        "I'll be there in 10 minutes, see you soon",
        "You have been selected for a special offer! Reply YES to claim",
        "Thanks for dinner last night, it was really fun!",
    ]

    print(f"\n  Sample Predictions ({model_name}):")
    print(f"  {'─'*55}")

    for msg in test_messages[:n]:
        tokens = preprocess(msg)
        pred, probs = model.predict(tokens)

        # Color indicator
        indicator = "🔴 SPAM" if pred == 'spam' else "🟢 HAM "

        print(f"\n  Message: \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print(f"  Prediction: {indicator}")
        print(f"  P(spam | msg) = {probs['spam']:.6f}")
        print(f"  P(ham  | msg) = {probs['ham']:.6f}")


def print_top_words(spam_freq, ham_freq, top_n=15):
    """
    Print the top discriminative words for each class.

    These are the words with the highest P(word | class), which
    are the most informative features for the Naïve Bayes model.
    """
    print(f"\n  Top {top_n} Spam Words:")
    print(f"  {'─'*35}")
    for word, count in spam_freq.most_common(top_n):
        print(f"    {word:<20} {count:>6} occurrences")

    print(f"\n  Top {top_n} Ham Words:")
    print(f"  {'─'*35}")
    for word, count in ham_freq.most_common(top_n):
        print(f"    {word:<20} {count:>6} occurrences")


def print_comparison_table(metrics_multi, metrics_bern):
    """Print a formatted comparison table of both models."""
    print(f"\n  {'─'*55}")
    print(f"  {'Metric':<20} {'Multinomial':>15} {'Bernoulli':>15}")
    print(f"  {'─'*55}")

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        display_name = metric.replace('_', ' ').title()
        print(f"  {display_name:<20} {metrics_multi[metric]:>15.4f} {metrics_bern[metric]:>15.4f}")

    # Determine winner for each metric
    print(f"  {'─'*55}")
    print(f"  {'Winner':<20}", end="")
    multi_wins = sum(1 for k in ['accuracy', 'precision', 'recall', 'f1_score']
                     if metrics_multi[k] > metrics_bern[k])
    bern_wins = 4 - multi_wins
    if multi_wins > bern_wins:
        print(f" {'← Multinomial wins':>31}")
    elif bern_wins > multi_wins:
        print(f" {'Bernoulli wins →':>31}")
    else:
        print(f" {'Tie':>31}")
    print(f"  {'─'*55}")


def main():
    """Run the complete spam classification analysis pipeline."""

    print_header("PROBABILISTIC SPAM CLASSIFICATION")
    print_header("Using Naïve Bayes — From Scratch")

    # ──────────────────────────────────────────────────────────────
    # STEP 1: Download and Load Data (Multiple Datasets)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 1: Loading Datasets")
    download_dataset()
    data = load_all_data()

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Train/Test Split (Stratified)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 2: Train/Test Split")
    train_data, test_data = train_test_split(data, test_ratio=0.2, seed=42)

    # ──────────────────────────────────────────────────────────────
    # STEP 3: Preprocess Text
    # ──────────────────────────────────────────────────────────────
    print_header("Step 3: Preprocessing")
    train_processed = preprocess_corpus(train_data)
    test_processed = preprocess_corpus(test_data)
    vocabulary = build_vocabulary(train_processed)
    print(f"  Vocabulary size: {len(vocabulary)} unique words")

    # Word frequency analysis
    freq = word_frequencies_by_class(train_processed)

    # ──────────────────────────────────────────────────────────────
    # STEP 4: Train Both Models
    # ──────────────────────────────────────────────────────────────
    print_header("Step 4: Training Models")

    # Model 1: Multinomial Naïve Bayes
    print("\n  [Model 1: Multinomial Naïve Bayes]")
    multinomial = MultinomialNaiveBayes(alpha=1.0)
    multinomial.fit(train_processed, vocabulary)

    # Model 2: Bernoulli Naïve Bayes
    print("\n  [Model 2: Bernoulli Naïve Bayes]")
    bernoulli = BernoulliNaiveBayes(alpha=1.0)
    bernoulli.fit(train_processed, vocabulary)

    # ──────────────────────────────────────────────────────────────
    # STEP 5: Evaluate on Test Set
    # ──────────────────────────────────────────────────────────────
    print_header("Step 5: Evaluation on Test Set")

    # Multinomial predictions
    y_true_m, y_pred_m, _ = multinomial.predict_batch(test_processed)
    report_m, metrics_m = classification_report(y_true_m, y_pred_m, "Multinomial NB")
    print(report_m)

    # Bernoulli predictions
    y_true_b, y_pred_b, _ = bernoulli.predict_batch(test_processed)
    report_b, metrics_b = classification_report(y_true_b, y_pred_b, "Bernoulli NB")
    print(report_b)

    # ──────────────────────────────────────────────────────────────
    # STEP 6: Model Comparison Table
    # ──────────────────────────────────────────────────────────────
    print_header("Step 6: Model Comparison")
    print_comparison_table(metrics_m, metrics_b)

    # ──────────────────────────────────────────────────────────────
    # STEP 7: Cross Validation
    # ──────────────────────────────────────────────────────────────
    print_header("Step 7: 5-Fold Cross Validation")

    cv_multi = k_fold_cross_validation(data, MultinomialNaiveBayes, k=5)
    cv_bern = k_fold_cross_validation(data, BernoulliNaiveBayes, k=5)

    # ──────────────────────────────────────────────────────────────
    # STEP 8: Confidence Intervals
    # ──────────────────────────────────────────────────────────────
    print_header("Step 8: 95% Confidence Intervals")
    print("\n  Multinomial NB:")
    ci_multi = confidence_interval_95(cv_multi['fold_accuracies'])
    print("\n  Bernoulli NB:")
    ci_bern = confidence_interval_95(cv_bern['fold_accuracies'])

    # ──────────────────────────────────────────────────────────────
    # STEP 9: Hypothesis Testing
    # ──────────────────────────────────────────────────────────────
    print_header("Step 9: Hypothesis Testing")

    # Paired t-test on CV scores
    print("\n  [Test 1: Paired t-test on CV accuracies]")
    t_result = paired_t_test(
        cv_multi['fold_accuracies'],
        cv_bern['fold_accuracies']
    )

    # McNemar's test on test set predictions
    print("\n  [Test 2: McNemar's test on test set predictions]")
    mc_result = mcnemar_test(y_true_m, y_pred_m, y_pred_b)

    # ──────────────────────────────────────────────────────────────
    # STEP 10: Top Words Analysis (Bonus)
    # ──────────────────────────────────────────────────────────────
    print_header("Step 10: Top Discriminative Words")
    print_top_words(freq['spam'], freq['ham'])

    # ──────────────────────────────────────────────────────────────
    # STEP 11: Sample Predictions
    # ──────────────────────────────────────────────────────────────
    print_header("Step 11: Sample Predictions")
    print_sample_predictions(multinomial, "Multinomial NB", n=8)
    print_sample_predictions(bernoulli, "Bernoulli NB", n=8)

    # ──────────────────────────────────────────────────────────────
    # STEP 12: Label Noise / Ambiguity Analysis
    # ──────────────────────────────────────────────────────────────
    print_header("Step 12: Label Noise & Ambiguity Analysis")
    print("\n  Analyzing how the model handles AMBIGUOUS messages")
    print("  (messages that different people might classify differently)")
    print("  Using confidence threshold = 0.70\n")

    for model, model_name in [(multinomial, "Multinomial NB"),
                               (bernoulli, "Bernoulli NB")]:
        noise = model.analyze_label_noise(test_processed, confidence_threshold=0.70)
        print(f"\n  {model_name} Label Noise Analysis:")
        print(f"  {'─'*50}")
        print(f"    Confident & Correct:  {noise['confident_correct']:>5} ({noise['confident_correct_pct']:.1f}%)")
        print(f"    Confident & WRONG:    {noise['confident_wrong']:>5} ({noise['confident_wrong_pct']:.1f}%)")
        print(f"    UNCERTAIN:            {noise['uncertain']:>5} ({noise['uncertain_pct']:.1f}%)")

        if noise['uncertain_examples']:
            print(f"\n    Example UNCERTAIN messages (borderline cases):")
            for i, ex in enumerate(noise['uncertain_examples'][:3], 1):
                print(f"      {i}. Label='{ex['label']}' | "
                      f"P(spam)={ex['spam_prob']:.4f}, P(ham)={ex['ham_prob']:.4f}")
                print(f"         Words: {ex['tokens_preview']}")

        if noise['mislabel_candidates']:
            print(f"\n    Possible MISLABELED data (model confidently disagrees):")
            for i, ex in enumerate(noise['mislabel_candidates'][:3], 1):
                print(f"      {i}. Labeled '{ex['label']}' but model says "
                      f"'{ex['predicted']}' (confidence={ex['confidence']:.4f})")
                print(f"         Words: {ex['tokens_preview']}")

    # ──────────────────────────────────────────────────────────────
    # STEP 13: Visualizations
    # ──────────────────────────────────────────────────────────────
    print_header("Step 13: Generating Visualizations")

    plot_word_frequency(freq['spam'], freq['ham'])
    plot_word_clouds(freq['spam'], freq['ham'])
    plot_confusion_matrix(metrics_m['confusion_matrix'], "Multinomial NB", "multinomial")
    plot_confusion_matrix(metrics_b['confusion_matrix'], "Bernoulli NB", "bernoulli")
    plot_accuracy_comparison(metrics_m, metrics_b)
    plot_error_distribution(y_true_m, y_pred_m, y_pred_b)
    plot_cv_results(cv_multi['fold_accuracies'], cv_bern['fold_accuracies'])

    # ──────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────
    print_header("ANALYSIS COMPLETE")
    print(f"""
  Summary:
    Dataset: {len(data)} messages (SMS + Email combined)
    Models trained: Multinomial NB + Bernoulli NB (from scratch)
    Multinomial Accuracy: {metrics_m['accuracy']:.4f}
    Bernoulli Accuracy:   {metrics_b['accuracy']:.4f}
    5-Fold CV Mean (Multi):  {cv_multi['mean_accuracy']:.4f} +/- {cv_multi['std_accuracy']:.4f}
    5-Fold CV Mean (Bern):   {cv_bern['mean_accuracy']:.4f} +/- {cv_bern['std_accuracy']:.4f}
    95% CI (Multi):  [{ci_multi['ci_lower']:.4f}, {ci_multi['ci_upper']:.4f}]
    95% CI (Bern):   [{ci_bern['ci_lower']:.4f}, {ci_bern['ci_upper']:.4f}]
    Hypothesis test:  {t_result['conclusion']}
    All plots saved to: outputs/

  To launch the interactive UI, run:
    streamlit run app.py
""")


if __name__ == "__main__":
    main()
