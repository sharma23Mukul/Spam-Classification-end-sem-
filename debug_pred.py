import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import preprocess

def debug_prediction():
    bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "app_bundle.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
        
    model = bundle['multinomial']
    
    # Example obvious spam message
    spam_msg = "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
    
    tokens = preprocess(spam_msg)
    print("Tokens:", tokens)
    
    # Compute manually
    log_posteriors = {}
    print(f"\nPrior P(spam): {model.log_priors['spam']:.4f}")
    print(f"Prior P(ham): {model.log_priors['ham']:.4f}")
    
    for cls in model.classes:
        ll = model._compute_log_likelihood(tokens, cls)
        print(f"\nClass: {cls}")
        print(f"  Log Likelihood: {ll:.4f}")
        for token in tokens:
            if token in model.log_likelihoods[cls]:
                print(f"    {token}: {model.log_likelihoods[cls][token]:.4f}")
            else:
                vocab_size = len(model.vocabulary)
                n_class = model.class_word_counts[cls]
                import math
                unseen_lp = math.log(model.alpha / (n_class + model.alpha * vocab_size))
                print(f"    {token} (UNSEEN): {unseen_lp:.4f}")
                
        log_posteriors[cls] = model.log_priors[cls] + ll
        print(f"  Total Log Posterior ({cls}): {log_posteriors[cls]:.4f}")
        
    result = model.predict_with_confidence(tokens)
    print("\nFinal Result:")
    print(result)

if __name__ == "__main__":
    debug_prediction()
