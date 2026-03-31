"""
save_model.py
=============
Trains the final Naïve Bayes models on all available data and 
saves them to disk using `pickle`. 

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

from data.loader import load_all_data
from preprocessing.pipeline import preprocess_corpus, build_vocabulary
from models.multinomial_nb import MultinomialNaiveBayes

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_model.pkl")

def save_model():
    print("\n" + "="*50)
    print("  Model Persistence (Serialization)")
    print("="*50)

    # 1. Load All Data (Train on 100% of the dataset for the production model)
    print("[1/4] Loading all datasets...")
    data = load_all_data()

    if not data:
        print("[!] No data available. Run 'download_data.py' first.")
        return

    # 2. Preprocessing
    print("[2/4] Preprocessing and tokenizing corpus...")
    processed_data = preprocess_corpus(data)
    vocabulary = build_vocabulary(processed_data)

    # 3. Training the best model (Multinomial NB was proven statistically better)
    print("[3/4] Training Production Multinomial Naïve Bayes model...")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(processed_data, vocabulary)

    # 4. Save to Disk
    print(f"[4/4] Saving trained model to {MODEL_PATH}...")
    
    # We package the model class instance and the base vocabulary just in case
    # though the model instance stores the vocabulary internally.
    export_bundle = {
        "model": model,
        "vocabulary_size": len(vocabulary),
        "total_messages_trained": len(data)
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(export_bundle, f)

    print(f"\n[✓] SUCCESS: Model saved! Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
    print("    You can now boot the FastAPI server using `api.py`.")

if __name__ == "__main__":
    save_model()
