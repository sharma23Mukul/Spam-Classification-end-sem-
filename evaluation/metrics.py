"""
evaluation/metrics.py
=====================
Classification metrics computed from scratch.

All metrics are derived from the CONFUSION MATRIX, which is the
fundamental tool for evaluating binary classifiers:

                    Predicted
                  Ham    Spam
    Actual  Ham [ TN  |  FP ]
           Spam [ FN  |  TP ]

Where:
    TP (True Positive)  = correctly predicted spam
    TN (True Negative)  = correctly predicted ham
    FP (False Positive) = ham incorrectly predicted as spam (Type I error)
    FN (False Negative) = spam incorrectly predicted as ham (Type II error)

Metrics:
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)     — "Of all predicted spam, how many are actually spam?"
    Recall    = TP / (TP + FN)     — "Of all actual spam, how many did we catch?"
    F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)  — harmonic mean
"""


def confusion_matrix(y_true, y_pred, positive='spam'):
    """
    Compute a 2×2 confusion matrix for binary classification.

    Parameters
    ----------
    y_true : list of str
        True labels.
    y_pred : list of str
        Predicted labels.
    positive : str
        The positive class label (default: 'spam').

    Returns
    -------
    dict
        {'TP': int, 'TN': int, 'FP': int, 'FN': int}

    Example
    -------
    >>> cm = confusion_matrix(['spam','ham','spam','ham'], ['spam','ham','ham','ham'])
    >>> print(cm)
    {'TP': 1, 'TN': 2, 'FP': 0, 'FN': 1}
    """
    tp = tn = fp = fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == positive and pred == positive:
            tp += 1  # Correctly identified spam
        elif true != positive and pred != positive:
            tn += 1  # Correctly identified ham
        elif true != positive and pred == positive:
            fp += 1  # Ham wrongly classified as spam (Type I error)
        else:  # true == positive and pred != positive
            fn += 1  # Spam missed — classified as ham (Type II error)

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def accuracy(cm):
    """
    Accuracy = (TP + TN) / Total

    The proportion of ALL predictions that are correct.
    Can be misleading for imbalanced datasets (e.g., if 90% is ham,
    a model that always predicts 'ham' gets 90% accuracy).

    Parameters
    ----------
    cm : dict
        Confusion matrix from confusion_matrix().

    Returns
    -------
    float
        Accuracy score between 0 and 1.
    """
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if total == 0:
        return 0.0
    return (cm['TP'] + cm['TN']) / total


def precision(cm):
    """
    Precision = TP / (TP + FP)

    "Of all messages we PREDICTED as spam, what fraction is actually spam?"

    High precision means few false positives (few ham messages
    incorrectly sent to the spam folder).

    Returns
    -------
    float
        Precision score between 0 and 1.
    """
    denominator = cm['TP'] + cm['FP']
    if denominator == 0:
        return 0.0
    return cm['TP'] / denominator


def recall(cm):
    """
    Recall (Sensitivity) = TP / (TP + FN)

    "Of all messages that ARE actually spam, what fraction did we catch?"

    High recall means few false negatives (few spam messages
    slipping into the inbox).

    Returns
    -------
    float
        Recall score between 0 and 1.
    """
    denominator = cm['TP'] + cm['FN']
    if denominator == 0:
        return 0.0
    return cm['TP'] / denominator


def f1_score(cm):
    """
    F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

    The HARMONIC MEAN of precision and recall. It balances both metrics.
    - If either precision or recall is low, F1 is low.
    - F1 = 1 only when both precision and recall are perfect.

    Why harmonic mean instead of arithmetic mean?
        Harmonic mean penalizes extreme imbalances more. For example:
        - Arithmetic mean of P=1.0, R=0.0 = 0.50 (misleadingly high)
        - Harmonic mean of P=1.0, R=0.0 = 0.00 (correctly shows failure)

    Returns
    -------
    float
        F1 score between 0 and 1.
    """
    p = precision(cm)
    r = recall(cm)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def classification_report(y_true, y_pred, model_name="Model"):
    """
    Generate a formatted classification report.

    Parameters
    ----------
    y_true : list of str
        True labels.
    y_pred : list of str
        Predicted labels.
    model_name : str
        Name to display in the report header.

    Returns
    -------
    tuple of (str, dict)
        (formatted_report_string, metrics_dict)
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy(cm)
    prec = precision(cm)
    rec = recall(cm)
    f1 = f1_score(cm)

    report = f"""
{'='*55}
  Classification Report: {model_name}
{'='*55}
  Confusion Matrix:
                    Predicted
                  Ham    Spam
    Actual  Ham [{cm['TN']:>5} | {cm['FP']:>5} ]
           Spam [{cm['FN']:>5} | {cm['TP']:>5} ]

  Metrics:
    Accuracy  = {acc:.4f}  ({cm['TP']+cm['TN']}/{cm['TP']+cm['TN']+cm['FP']+cm['FN']})
    Precision = {prec:.4f}  (TP/(TP+FP) = {cm['TP']}/({cm['TP']}+{cm['FP']}))
    Recall    = {rec:.4f}  (TP/(TP+FN) = {cm['TP']}/({cm['TP']}+{cm['FN']}))
    F1 Score  = {f1:.4f}  (2×P×R/(P+R))
{'='*55}
"""

    metrics = {
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    return report, metrics
