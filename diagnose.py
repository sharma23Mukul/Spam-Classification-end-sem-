"""Quick diagnostic script to test model predictions."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import preprocess

bundle_path = os.path.join("models", "app_bundle.pkl")
if os.path.exists(bundle_path):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    print("=== Bundle Info ===")
    print(f"Total data: {bundle['total_data']}")
    print(f"Total train: {bundle['total_train']}")
    print(f"Total test: {bundle['total_test']}")
    print(f"Vocabulary size: {len(bundle['vocabulary'])}")
    print()
    print("=== Multinomial Metrics ===")
    for k, v in bundle['metrics_multi'].items():
        print(f"  {k}: {v}")
    print()
    print("=== Bernoulli Metrics ===")
    for k, v in bundle['metrics_bern'].items():
        print(f"  {k}: {v}")
    
    # Test some messages
    model = bundle['multinomial']
    bern = bundle['bernoulli']
    
    test_msgs = [
        ("Congratulations! You won a free iPhone! Click here to claim your prize!", "spam"),
        ("Hey, are you free for lunch tomorrow?", "ham"),
        ("URGENT: Your account has been compromised. Click here to verify.", "spam"),
        ("Meeting at 3pm today in conference room B", "ham"),
        ("You have won a lottery worth 1 million dollars", "spam"),
        ("Can you pick up the kids from school today?", "ham"),
        ("Dear customer, your subscription renewal is pending. Click to renew.", "spam"),
        ("Free entry to win a trip to Bahamas! Text WIN to 12345", "spam"),
        ("Hi mom, I will be home by 7pm", "ham"),
        ("Limited time offer! 50% off on all products!", "spam"),
    ]
    
    print()
    print("=== Prediction Tests (Multinomial) ===")
    wrong_multi = 0
    for msg, expected in test_msgs:
        tokens = preprocess(msg)
        result = model.predict_with_confidence(tokens, confidence_threshold=0.70)
        pred = result['prediction'] if result['prediction'] != 'uncertain' else max(result['probabilities'], key=result['probabilities'].get)
        correct = "OK" if pred == expected else "WRONG"
        if pred != expected:
            wrong_multi += 1
        print(f"  [{correct:>5}] pred={result['prediction']:>9} expected={expected:>4} P(spam)={result['probabilities']['spam']:.4f} P(ham)={result['probabilities']['ham']:.4f} | {msg[:55]}")
    print(f"  Wrong: {wrong_multi}/{len(test_msgs)}")
    
    print()
    print("=== Prediction Tests (Bernoulli) ===")
    wrong_bern = 0
    for msg, expected in test_msgs:
        tokens = preprocess(msg)
        result = bern.predict_with_confidence(tokens, confidence_threshold=0.70)
        pred = result['prediction'] if result['prediction'] != 'uncertain' else max(result['probabilities'], key=result['probabilities'].get)
        correct = "OK" if pred == expected else "WRONG"
        if pred != expected:
            wrong_bern += 1
        print(f"  [{correct:>5}] pred={result['prediction']:>9} expected={expected:>4} P(spam)={result['probabilities']['spam']:.4f} P(ham)={result['probabilities']['ham']:.4f} | {msg[:55]}")
    print(f"  Wrong: {wrong_bern}/{len(test_msgs)}")

    # Check class balance / priors
    print()
    print("=== Model Priors ===")
    import math
    for cls in model.classes:
        print(f"  P({cls}) = {math.exp(model.log_priors[cls]):.4f}")
    print()
    print("=== Vocab Sample (spam-indicative words) ===")
    spam_words = ['free', 'win', 'winner', 'prize', 'click', 'call', 'urgent', 'congratulations', 'offer', 'cash', 'claim', 'lottery']
    for w in spam_words:
        in_vocab = w in set(bundle['vocabulary'])
        print(f"  '{w}': in_vocab={in_vocab}")
else:
    print("No bundle found, would need to train from scratch")
