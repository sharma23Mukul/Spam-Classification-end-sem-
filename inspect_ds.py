from datasets import load_dataset
from collections import Counter

TOKEN = "your_huggingface_token_here"
ds = load_dataset("jason23322/high-accuracy-email-classifier", split="train", token=TOKEN)

cats = Counter(ds["category"])
print("Train split categories:")
for k, v in cats.most_common():
    print(f"  {k}: {v}")

ds_test = load_dataset("jason23322/high-accuracy-email-classifier", split="test", token=TOKEN)
cats2 = Counter(ds_test["category"])
print("\nTest split categories:")
for k, v in cats2.most_common():
    print(f"  {k}: {v}")

print("\nCategory ID mapping (first 5 per category):")
seen = {}
for row in ds:
    cat = row["category"]
    if cat not in seen:
        seen[cat] = row["category_id"]
for cat, cid in seen.items():
    print(f"  {cat} -> {cid}")
