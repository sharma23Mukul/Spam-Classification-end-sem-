"""
comparison_report.py
====================
Generates comparative analysis visualizations and performs
hyperparameter optimization for Naïve Bayes variants.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

from evaluation.metrics import classification_report, get_binary_labels
from evaluation.hypothesis_testing import mcnemar_test, paired_t_test

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_metrics_bar_chart(results):
    """Generate a grouped bar chart comparing key metrics."""
    data = []
    for model_name, res in results.items():
        data.append({
            'Model': model_name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score']
        })
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted, palette='viridis')
    plt.title('Model Comparison across Key Metrics')
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=300)
    plt.close()

def generate_roc_curves(y_true, models_probs):
    """Generate ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    y_true_bin = get_binary_labels(y_true, positive='spam')

    for model_name, probs in models_probs.items():
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, probs)
            plt.plot(fpr, tpr, label=f'{model_name}')
        except Exception as e:
            print(f"Could not plot ROC for {model_name}: {e}")

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300)
    plt.close()

def generate_confusion_matrices(results):
    """Generate confusion matrices heatmaps."""
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
    
    if num_models == 1:
        axes = [axes]
        
    for ax, (model_name, res) in zip(axes, results.items()):
        cm = res['confusion_matrix']
        cm_matrix = [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]]
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()

def generate_cv_boxplot(cv_results):
    """Generate boxplot for cross-validation results."""
    data = []
    for model_name, scores in cv_results.items():
        for score in scores:
            data.append({'Model': model_name, 'Accuracy': score})
            
    if not data:
        return
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='Accuracy', data=df, palette='Set2')
    sns.stripplot(x='Model', y='Accuracy', data=df, color='black', alpha=0.6)
    plt.title('5-Fold Cross-Validation Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cv_boxplot.png'), dpi=300)
    plt.close()

def grid_search_multinomial(train_data, val_data, vocab, alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]):
    """Optimize alpha for Multinomial NB."""
    print("\n--- Grid Search: Multinomial NB ---")
    from models.multinomial_nb import MultinomialNaiveBayes
    
    best_alpha = None
    best_f1 = -1
    best_model = None
    
    y_val = [l for l, _ in val_data]
    
    for alpha in alphas:
        model = MultinomialNaiveBayes(alpha=alpha)
        model.fit(train_data, vocab)
        
        preds = []
        for _, tokens in val_data:
            pred, _ = model.predict(tokens)
            preds.append(pred)
            
        _, metrics = classification_report(y_val, preds, positive='spam')
        f1 = metrics['f1_score']
        print(f"Alpha: {alpha} -> F1 Score: {f1:.4f} (Accuracy: {metrics['accuracy']:.4f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model = model
            
    print(f"Best Alpha: {best_alpha} with F1 = {best_f1:.4f}")
    return best_model, best_alpha

def grid_search_bernoulli(train_data, val_data, vocab, alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]):
    """Optimize alpha for Bernoulli NB."""
    print("\n--- Grid Search: Bernoulli NB ---")
    from models.bernoulli_nb import BernoulliNaiveBayes
    
    best_alpha = None
    best_f1 = -1
    best_model = None
    
    y_val = [l for l, _ in val_data]
    
    for alpha in alphas:
        model = BernoulliNaiveBayes(alpha=alpha)
        model.fit(train_data, vocab)
        
        preds = []
        for _, tokens in val_data:
            pred, _ = model.predict(tokens)
            preds.append(pred)
            
        _, metrics = classification_report(y_val, preds, positive='spam')
        f1 = metrics['f1_score']
        print(f"Alpha: {alpha} -> F1 Score: {f1:.4f} (Accuracy: {metrics['accuracy']:.4f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model = model
            
    print(f"Best Alpha: {best_alpha} with F1 = {best_f1:.4f}")
    return best_model, best_alpha

def threshold_tuning(model, val_data):
    """Tune confidence threshold using precision-recall curve analysis."""
    print(f"\n--- Threshold Tuning ---")
    y_val = [l for l, _ in val_data]
    y_val_bin = get_binary_labels(y_val, positive='spam')
    
    probs = []
    for _, tokens in val_data:
        _, p = model.predict(tokens)
        probs.append(p['spam'])
        
    precisions, recalls, thresholds = precision_recall_curve(y_val_bin, probs)
    
    # We want Spam Precision >= 0.95 and Spam Recall >= 0.90
    best_threshold = 0.5
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= 0.95 and r >= 0.90:
            best_threshold = t
            print(f"Found threshold {t:.4f} with Precision {p:.4f} and Recall {r:.4f}")
            break
            
    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='PR Curve')
    plt.axvline(x=0.90, color='r', linestyle='--', label='Recall = 0.90 Target')
    plt.axhline(y=0.95, color='g', linestyle='--', label='Precision = 0.95 Target')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Best Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'), dpi=300)
    plt.close()
    
    return best_threshold

def generate_summary_csv(results, cv_results, filename="results_summary.csv"):
    """Export results to CSV."""
    data = []
    for model, res in results.items():
        row = {
            'Model': model,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score'],
            'Macro F1': res['macro_f1'],
            'Weighted F1': res['weighted_f1'],
            'ROC-AUC': res.get('roc_auc', None),
            'PR-AUC': res.get('pr_auc', None)
        }
        if model in cv_results:
            cv_scores = cv_results[model]
            row['CV Mean Acc'] = sum(cv_scores) / len(cv_scores)
            row['CV Std Acc'] = math.sqrt(sum((s - row['CV Mean Acc'])**2 for s in cv_scores) / max(1, len(cv_scores)-1))
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"\n[OK] Saved results summary to {os.path.join(OUTPUT_DIR, filename)}")
