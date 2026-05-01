"""
evaluation/metrics.py
=====================
Classification metrics for evaluating spam classifiers.

Includes basic metrics from scratch (for pedagogical purposes) and
advanced metrics (ROC-AUC, PR-AUC) using scikit-learn.
"""

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def confusion_matrix(y_true, y_pred, positive='spam'):
    """Compute a 2×2 confusion matrix for binary classification."""
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == positive and pred == positive:
            tp += 1
        elif true != positive and pred != positive:
            tn += 1
        elif true != positive and pred == positive:
            fp += 1
        else:
            fn += 1
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def get_binary_labels(y, positive='spam'):
    return [1 if label == positive else 0 for label in y]


def classification_report(y_true, y_pred, y_probs=None, model_name="Model", positive='spam'):
    """
    Generate a formatted classification report including advanced metrics.

    Parameters
    ----------
    y_true : list of str
        True labels.
    y_pred : list of str
        Predicted labels.
    y_probs : list of float, optional
        Probability of the positive class (used for AUC).
    model_name : str
        Name to display in the report header.

    Returns
    -------
    tuple of (str, dict)
        (formatted_report_string, metrics_dict)
    """
    cm = confusion_matrix(y_true, y_pred, positive)
    
    y_true_bin = get_binary_labels(y_true, positive)
    y_pred_bin = get_binary_labels(y_pred, positive)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    
    # Binary metrics (focusing on positive class)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # Macro and Weighted Averages
    macro_prec = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_rec = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    
    weighted_prec = precision_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    weighted_rec = recall_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)

    # Area Under Curve metrics
    roc_auc = None
    pr_auc = None
    fpr = None
    tpr = None
    precisions = None
    recalls = None
    
    if y_probs is not None:
        try:
            roc_auc = roc_auc_score(y_true_bin, y_probs)
            pr_auc = average_precision_score(y_true_bin, y_probs)
            fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
            precisions, recalls, _ = precision_recall_curve(y_true_bin, y_probs)
        except ValueError:
            pass # Handle cases where only one class is present in true labels

    report = f"""
{'='*55}
  Classification Report: {model_name}
{'='*55}
  Confusion Matrix:
                    Predicted
                  Ham    Spam
    Actual  Ham [{cm['TN']:>5} | {cm['FP']:>5} ]
           Spam [{cm['FN']:>5} | {cm['TP']:>5} ]

  Binary Metrics (Spam as positive):
    Accuracy  = {acc:.4f}
    Precision = {prec:.4f}
    Recall    = {rec:.4f}
    F1 Score  = {f1:.4f}
    
  Averages:
    Macro F1    = {macro_f1:.4f}
    Weighted F1 = {weighted_f1:.4f}
"""

    if roc_auc is not None and pr_auc is not None:
        report += f"""
  Curve Metrics:
    ROC-AUC   = {roc_auc:.4f}
    PR-AUC    = {pr_auc:.4f}
"""
    report += f"{'='*55}\n"

    metrics = {
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precisions': precisions,
        'recalls': recalls
    }

    return report, metrics
