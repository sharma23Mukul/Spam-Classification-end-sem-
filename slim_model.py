import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join("models", "saved_model.pkl")
MAX_VOCAB = 50000

print("Loading full model...")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
spam_ll = model.log_likelihoods.get("spam", {})
ham_ll = model.log_likelihoods.get("ham", {})

print(f"Original vocab: {len(spam_ll)} words")

word_scores = {}
for word in spam_ll:
    if word in ham_ll:
        word_scores[word] = abs(spam_ll[word] - ham_ll[word])

top_words = sorted(word_scores, key=word_scores.get, reverse=True)[:MAX_VOCAB]
top_set = set(top_words)

model.log_likelihoods["spam"] = {w: v for w, v in spam_ll.items() if w in top_set}
model.log_likelihoods["ham"] = {w: v for w, v in ham_ll.items() if w in top_set}

print(f"Trimmed vocab: {len(model.log_likelihoods['spam'])} words")

slim_bundle = {
    "model": model,
    "vocabulary_size": MAX_VOCAB,
    "total_messages_trained": bundle["total_messages_trained"]
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(slim_bundle, f)

print(f"Slim model saved! Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
