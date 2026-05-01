"""Debug Bernoulli NB to understand why it predicts ham for everything."""
import os, sys, pickle, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.pipeline import preprocess

bundle_path = os.path.join("models", "app_bundle.pkl")
with open(bundle_path, "rb") as f:
    bundle = pickle.load(f)

bern = bundle['bernoulli']

msg = "Congratulations! You won a free iPhone! Click here to claim your prize!"
tokens = preprocess(msg)
print(f"Message: {msg}")
print(f"Tokens: {tokens}")
print(f"Num tokens: {len(tokens)}")
print(f"Vocab size: {len(bern.vocabulary)}")
print()

# Compute manually
for cls in ['spam', 'ham']:
    log_prior = bern.log_priors[cls]
    baseline = bern._absent_baseline[cls]
    
    token_set = set(tokens) & bern.vocab_set
    
    delta = 0.0
    for word in token_set:
        d = bern.log_prob_present[cls][word] - bern.log_prob_absent[cls][word]
        delta += d
        print(f"  {cls}: word='{word}' log_P_present={bern.log_prob_present[cls][word]:.4f} log_P_absent={bern.log_prob_absent[cls][word]:.4f} delta={d:.4f}")
    
    log_likelihood = baseline + delta
    log_posterior = log_prior + log_likelihood
    
    print(f"\n  {cls}: log_prior={log_prior:.4f} baseline={baseline:.2f} delta={delta:.4f} log_likelihood={log_likelihood:.2f} log_posterior={log_posterior:.2f}")
    print()

# Check: what's the baseline ratio?  
print(f"Baseline ham:  {bern._absent_baseline['ham']:.2f}")
print(f"Baseline spam: {bern._absent_baseline['spam']:.2f}")
print(f"Baseline diff (ham - spam): {bern._absent_baseline['ham'] - bern._absent_baseline['spam']:.2f}")
print()

# The issue: with 41K vocab words and P(spam)=0.21, 
# the baseline for ham is much less negative because most words are absent
# from spam docs more than from ham docs

# Check doc counts
print(f"Spam docs: {bern.class_doc_counts['spam']}")
print(f"Ham docs: {bern.class_doc_counts['ham']}")
print()

# Sample some absent probabilities
import random
random.seed(42)
sample_words = random.sample(list(bern.vocabulary), 10)
print("Sample word absent probabilities:")
for w in sample_words:
    spam_absent = math.exp(bern.log_prob_absent['spam'][w])
    ham_absent = math.exp(bern.log_prob_absent['ham'][w])
    print(f"  '{w}': P(absent|spam)={spam_absent:.6f} P(absent|ham)={ham_absent:.6f} ratio={spam_absent/ham_absent:.4f}")
