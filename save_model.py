"""
save_model.py
=============
Trains the final Naïve Bayes models on ALL available data and 
saves them to disk using `pickle`. 

Includes Laplace Smoothing Optimization:
    We search over multiple α values using the validation set
    and pick the one that maximizes accuracy. This prevents
    overfitting (α too small) and underfitting (α too large).

Why save the model?
-------------------
When running an API (or even the Streamlit app), rebuilding the 
vocabulary and computing the word probabilities from scratch every 
time the server starts is slow. 
By serializing (pickling) the fitted model instance, the API can 
start and load the model into memory in milliseconds, enabling 
lightning-fast, hassle-free predictions.
"""

import os
import sys
import pickle

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_dataset
from data.loader import load_all_data, train_val_test_split
from preprocessing.pipeline import (
    preprocess_corpus, build_vocabulary, word_frequencies_by_class
)
from models.multinomial_nb import MultinomialNaiveBayes
from models.bernoulli_nb import BernoulliNaiveBayes
from models.gaussian_nb import GaussianNaiveBayes
from evaluation.metrics import classification_report

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pkl")
APP_BUNDLE_PATH = os.path.join(MODEL_DIR, "app_bundle.pkl")

# ─── Laplace Smoothing Search Space ─────────────────────────────
# α = 0.01 : very aggressive, almost no smoothing (risky for unseen words)
# α = 0.1  : light Lidstone smoothing
# α = 0.5  : moderate smoothing
# α = 1.0  : classic Laplace smoothing
# α = 2.0  : heavy smoothing (conservative)
ALPHA_CANDIDATES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def find_best_alpha(train_processed, val_processed, vocabulary, model_class, model_name):
    """
    Search for the optimal Laplace smoothing parameter α using the validation set.
    
    For each candidate α, we:
        1. Train the model on the training set
        2. Evaluate accuracy on the validation set
        3. Pick the α with the highest validation accuracy
    
    This is a form of hyperparameter tuning that prevents us from
    blindly using α=1.0 when a different value might work better.
    
    Parameters
    ----------
    train_processed : list of tuple
        Preprocessed training data.
    val_processed : list of tuple
        Preprocessed validation data.
    vocabulary : list of str
        The vocabulary built from training data.
    model_class : class
        The NB model class to instantiate (MultinomialNaiveBayes or BernoulliNaiveBayes).
    model_name : str
        Human-readable name for logging.
        
    Returns
    -------
    float
        The best α value.
    """
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Laplace Smoothing Optimization: {model_name:<11s} │")
    print(f"  ├─────────┬───────────┬───────────────────────┤")
    print(f"  │    α     │ Val Acc   │ Status                │")
    print(f"  ├─────────┼───────────┼───────────────────────┤")
    
    best_alpha = 1.0
    best_acc = 0.0
    
    for alpha in ALPHA_CANDIDATES:
        model = model_class(alpha=alpha)
        # Train silently (suppress print output)
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            model.fit(train_processed, vocabulary)
        finally:
            sys.stdout = old_stdout
        
        # Evaluate on validation set
        y_true, y_pred, _ = model.predict_batch(val_processed)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = correct / len(y_true)
        
        marker = " ◀ BEST" if acc > best_acc else ""
        print(f"  │  {alpha:<6.2f} │  {acc:.4f}  │{marker:>23s}│")
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    print(f"  └─────────┴───────────┴───────────────────────┘")
    print(f"  ✓ Best α = {best_alpha} (Val Accuracy = {best_acc:.4f})")
    
    return best_alpha


def save_model():
    print("\n" + "="*50)
    print("  Model Persistence (Serialization)")
    print("="*50)

    # 0. Ensure data is downloaded
    download_dataset()

    # 1. Load ALL Data (SMS + SpamAssassin + Enron + Marketing + HF Email Classifier)
    print("[1/7] Loading ALL datasets...")
    data = load_all_data()

    if not data:
        print("[!] No data available. Run 'download_data.py' first.")
        return

    # 2. Split into train/val/test
    print("[2/7] Splitting data into train/val/test...")
    train_data, val_data, test_data = train_val_test_split(data, val_ratio=0.15, test_ratio=0.15, seed=42)

    # 3. Preprocessing
    print("[3/7] Preprocessing and tokenizing corpus...")
    train_processed = preprocess_corpus(train_data)
    val_processed = preprocess_corpus(val_data)
    test_processed = preprocess_corpus(test_data)
    vocabulary = build_vocabulary(train_processed, min_df=2, max_df=0.95)
    freq = word_frequencies_by_class(train_processed)

    # 4. Laplace Smoothing Optimization
    print("[4/7] Optimizing Laplace smoothing parameter α...")
    best_alpha_multi = find_best_alpha(
        train_processed, val_processed, vocabulary,
        MultinomialNaiveBayes, "Multinomial"
    )
    best_alpha_bern = find_best_alpha(
        train_processed, val_processed, vocabulary,
        BernoulliNaiveBayes, "Bernoulli"
    )

    # 5. Train final models with optimized α
    print(f"\n[5/7] Training Multinomial NB (α={best_alpha_multi})...")
    multinomial = MultinomialNaiveBayes(alpha=best_alpha_multi)
    multinomial.fit(train_processed, vocabulary)

    print(f"[6/7] Training Bernoulli NB (α={best_alpha_bern})...")
    bernoulli = BernoulliNaiveBayes(alpha=best_alpha_bern)
    bernoulli.fit(train_processed, vocabulary)
    
    print("[*] Training Gaussian Naïve Bayes...")
    gaussian = GaussianNaiveBayes()
    gaussian.fit(train_processed, vocabulary)

    # 6. Evaluate all models on the test set
    print("[7/7] Evaluating models and saving everything...")
    y_true_m, y_pred_m, probs_m = multinomial.predict_batch(test_processed)
    prob_spam_m = [p['spam'] for p in probs_m]
    _, metrics_m = classification_report(y_true_m, y_pred_m, y_probs=prob_spam_m, model_name="Multinomial")

    y_true_b, y_pred_b, probs_b = bernoulli.predict_batch(test_processed)
    prob_spam_b = [p['spam'] for p in probs_b]
    _, metrics_b = classification_report(y_true_b, y_pred_b, y_probs=prob_spam_b, model_name="Bernoulli")
    
    y_true_g, y_pred_g, probs_g = gaussian.predict_batch(test_processed)
    prob_spam_g = [p['spam'] for p in probs_g]
    _, metrics_g = classification_report(y_true_g, y_pred_g, y_probs=prob_spam_g, model_name="Gaussian")

    # --- Save the API model (backward compatible) ---
    api_bundle = {
        "model": multinomial,
        "vocabulary_size": len(vocabulary),
        "total_messages_trained": len(data)
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(api_bundle, f)
    print(f"\n[✓] API model saved! Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")

    # --- Save the full app bundle (everything Streamlit needs) ---
    app_bundle = {
        "multinomial": multinomial,
        "bernoulli": bernoulli,
        "gaussian": gaussian,
        "vocabulary": vocabulary,
        "freq": freq,
        "metrics_multi": metrics_m,
        "metrics_bern": metrics_b,
        "metrics_gauss": metrics_g,
        "total_data": len(data),
        "total_train": len(train_data),
        "total_test": len(test_data),
        "best_alpha_multi": best_alpha_multi,
        "best_alpha_bern": best_alpha_bern,
    }
    with open(APP_BUNDLE_PATH, "wb") as f:
        pickle.dump(app_bundle, f)
    print(f"[✓] App bundle saved!  Size: {os.path.getsize(APP_BUNDLE_PATH) / 1024 / 1024:.2f} MB")
    print(f"    Multinomial α = {best_alpha_multi}, Bernoulli α = {best_alpha_bern}")
    print("    You can now boot the Streamlit app or FastAPI server instantly.")

if __name__ == "__main__":
    save_model()
