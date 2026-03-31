"""
evaluation/cross_validation.py
==============================
K-Fold Cross Validation implemented from scratch.

Cross Validation Purpose:
    A single train/test split may give optimistic or pessimistic results
    depending on which data points end up in which set. K-Fold CV
    addresses this by:

    1. Dividing the dataset into K equal-sized "folds"
    2. Training on K-1 folds, testing on the remaining 1 fold
    3. Repeating K times (each fold serves as the test set once)
    4. Averaging results across all K runs

    This gives a more ROBUST estimate of model performance and allows
    us to compute standard deviation and confidence intervals.

    With K=5:
        Fold 1: [Test] [Train] [Train] [Train] [Train]
        Fold 2: [Train] [Test] [Train] [Train] [Train]
        Fold 3: [Train] [Train] [Test] [Train] [Train]
        Fold 4: [Train] [Train] [Train] [Test] [Train]
        Fold 5: [Train] [Train] [Train] [Train] [Test]
"""

import random
from preprocessing.pipeline import preprocess_corpus, build_vocabulary
from evaluation.metrics import confusion_matrix, accuracy


def k_fold_split(data, k=5, seed=42):
    """
    Split data into K stratified folds.

    Stratification ensures each fold has approximately the same
    proportion of spam and ham as the full dataset.

    Parameters
    ----------
    data : list of tuple
        List of (label, message) tuples.
    k : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of tuple
        List of K (train_data, test_data) pairs.
    """
    random.seed(seed)

    # Separate by class for stratified splitting
    spam = [(l, m) for l, m in data if l == 'spam']
    ham = [(l, m) for l, m in data if l == 'ham']

    random.shuffle(spam)
    random.shuffle(ham)

    # Create K folds for each class
    spam_folds = _split_into_k(spam, k)
    ham_folds = _split_into_k(ham, k)

    # Combine folds and create train/test pairs
    fold_pairs = []
    for i in range(k):
        test_data = spam_folds[i] + ham_folds[i]
        train_data = []
        for j in range(k):
            if j != i:
                train_data.extend(spam_folds[j])
                train_data.extend(ham_folds[j])

        random.shuffle(train_data)
        random.shuffle(test_data)
        fold_pairs.append((train_data, test_data))

    return fold_pairs


def _split_into_k(items, k):
    """Split a list into K approximately equal parts."""
    fold_size = len(items) // k
    remainder = len(items) % k

    folds = []
    start = 0
    for i in range(k):
        # Distribute remainder items across the first few folds
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(items[start:end])
        start = end

    return folds


def k_fold_cross_validation(data, model_class, k=5, alpha=1.0, seed=42):
    """
    Perform K-Fold Cross Validation.

    For each fold:
        1. Split into train/test
        2. Preprocess training data
        3. Build vocabulary from training data ONLY (no data leakage!)
        4. Train model
        5. Evaluate on test fold

    Parameters
    ----------
    data : list of tuple
        List of (label, message) tuples (raw, unprocessed).
    model_class : class
        The NaiveBayes class to instantiate (MultinomialNaiveBayes or
        BernoulliNaiveBayes).
    k : int
        Number of folds (default: 5).
    alpha : float
        Smoothing parameter for the model.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {
            'fold_accuracies': [float, ...],
            'fold_metrics': [dict, ...],
            'mean_accuracy': float,
            'std_accuracy': float,
            'fold_predictions': [(y_true, y_pred), ...]
        }
    """
    fold_pairs = k_fold_split(data, k=k, seed=seed)

    fold_accuracies = []
    fold_metrics = []
    fold_predictions = []

    print(f"\n{'='*55}")
    print(f"  {k}-Fold Cross Validation: {model_class.__name__}")
    print(f"{'='*55}")

    for i, (train_data, test_data) in enumerate(fold_pairs):
        print(f"\n  --- Fold {i+1}/{k} ---")
        print(f"      Train: {len(train_data)}, Test: {len(test_data)}")

        # Preprocess training data
        train_processed = preprocess_corpus(train_data)
        test_processed = preprocess_corpus(test_data)

        # Build vocabulary from TRAINING data only (avoid data leakage)
        vocabulary = build_vocabulary(train_processed)

        # Train model
        model = model_class(alpha=alpha)
        model.fit(train_processed, vocabulary)

        # Predict on test data
        y_true, y_pred, _ = model.predict_batch(test_processed)

        # Compute metrics
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy(cm)

        fold_accuracies.append(acc)
        fold_metrics.append(cm)
        fold_predictions.append((y_true, y_pred))

        print(f"      Accuracy: {acc:.4f}")

    # Compute summary statistics
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    variance = sum((a - mean_acc) ** 2 for a in fold_accuracies) / (k - 1)
    std_acc = variance ** 0.5

    print(f"\n  {'─'*40}")
    print(f"  Cross-Validation Results:")
    print(f"    Fold Accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
    print(f"    Mean Accuracy:   {mean_acc:.4f}")
    print(f"    Std Deviation:   {std_acc:.4f}")
    print(f"  {'─'*40}")

    return {
        'fold_accuracies': fold_accuracies,
        'fold_metrics': fold_metrics,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_predictions': fold_predictions
    }
