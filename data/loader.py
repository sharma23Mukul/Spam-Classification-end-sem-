"""
data/loader.py
==============
Functions for loading MULTIPLE spam classification datasets and
merging them into a unified training corpus.

Supported datasets:
    1. SMS Spam Collection (UCI) — SMS messages
    2. SpamAssassin Public Corpus — Email messages

Why multiple datasets?
    Training on a single dataset can lead to OVERFITTING to that
    dataset's specific patterns. By combining SMS and email data:
    - The model learns both SMS spam patterns ("Win free prize!") and
      email spam patterns ("Dear beneficiary, I am writing to...")
    - The vocabulary is more diverse, improving probability estimates
    - The model generalizes better to new, unseen messages

Statistical Note:
    We perform STRATIFIED splitting to maintain the same spam/ham ratio
    in both train and test sets. This is important because the combined
    dataset may have a different class balance than either individual
    dataset.

Label Noise & Ambiguity:
    Some messages are genuinely borderline — one person's "promotional
    email" is another person's spam. We address this by:
    1. Using a CONFIDENCE THRESHOLD in predictions
    2. Flagging uncertain predictions (P close to 0.5) as "uncertain"
    3. Tracking label agreement across model predictions
"""

import os
import random


def load_sms_data(filepath=None):
    """
    Load the SMS Spam Collection dataset from a TSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the TSV file. Defaults to data/SMSSpamCollection.tsv

    Returns
    -------
    list of tuple
        Each tuple is (label, message) where label is 'spam' or 'ham'.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "SMSSpamCollection.tsv"
        )

    if not os.path.exists(filepath):
        print(f"[!!] SMS dataset not found at: {filepath}")
        return []

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', maxsplit=1)
            if len(parts) != 2:
                continue

            label, message = parts
            label = label.strip().lower()

            if label not in ('spam', 'ham'):
                continue

            data.append((label, message.strip()))

    return data


def load_csv_directory(directory_path=None, limit_per_file=1000000):
    """
    Load all CSV files from a specific directory and automatically 
    parse their text and label columns.
    """
    import csv
    import glob
    import sys
    
    # Increase CSV field size limit to handle abnormally massive emails
    csv.field_size_limit(sys.maxsize)

    if directory_path is None:
        directory_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "custom_datasets"
        )

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        return []

    data = []
    print(f"\n  [>>] Found {len(csv_files)} custom CSV datasets in: {os.path.basename(directory_path)}/, parsing...")

    spam_classes = {'spam', 'promotion', 'promotional', 'phishing'}
    ham_classes = {'ham', 'update', 'updates', 'primary', 'personal', 'social', 'notification'}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"      -> Parsing {filename}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                text_col = next((c for c in reader.fieldnames if c and c.lower() in ('text', 'email', 'message', 'body', 'content', 'sms')), None)
                label_col = next((c for c in reader.fieldnames if c and c.lower() in ('label', 'category', 'class', 'type', 'target')), None)
                
                if not text_col or not label_col:
                    print(f"        [!!] Skipping {filename}: Missing clear 'text' or 'label' columns. Found: {reader.fieldnames}")
                    continue
                    
                count = 0
                for row in reader:
                    if count >= limit_per_file:
                        break
                        
                    raw_label = str(row.get(label_col, '')).strip().lower()
                    text = str(row.get(text_col, '')).strip()
                    
                    if not text:
                        continue
                        
                    binary_label = None
                    if raw_label in spam_classes:
                        binary_label = 'spam'
                    elif raw_label in ham_classes:
                        binary_label = 'ham'
                        
                    if binary_label == None:
                        if raw_label == '1': binary_label = 'spam'
                        elif raw_label == '0': binary_label = 'ham'
                        else: continue
                    
                    data.append((binary_label, text[:1500]))
                    count += 1
                    
        except Exception as e:
            print(f"        [!!] Failed to parse {filename}: {e}")

    return data

def load_all_data():
    """
    Load and MERGE all available datasets into a single corpus.

    This function:
        1. Loads the SMS Spam Collection
        2. Loads the SpamAssassin email corpus (if downloaded)
        3. Merges them with source tracking
        4. Reports dataset statistics

    Combining multiple datasets is a form of DATA AUGMENTATION that:
        - Increases the effective training size
        - Diversifies the word distribution
        - Reduces the variance of our probability estimates

    Returns
    -------
    list of tuple
        Each tuple is (label, message).
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # ─── Load SMS Data ──────────────────────────────────────────
    sms_data = load_sms_data()
    sms_spam = sum(1 for l, _ in sms_data if l == 'spam')
    sms_ham = len(sms_data) - sms_spam

    print(f"\n  Dataset 1: SMS Spam Collection")
    print(f"    Total: {len(sms_data)} messages")
    print(f"    Spam: {sms_spam}, Ham: {sms_ham}")

    # ─── Load SpamAssassin Data ─────────────────────────────────
    sa_data = []
    sa_dir = os.path.join(data_dir, "spamassassin")

    if os.path.exists(sa_dir) and os.path.exists(os.path.join(sa_dir, ".download_complete")):
        from download_data import load_spamassassin_data
        sa_data = load_spamassassin_data(sa_dir)
        sa_spam = sum(1 for l, _ in sa_data if l == 'spam')
        sa_ham = len(sa_data) - sa_spam

        print(f"\n  Dataset 2: SpamAssassin Email Corpus")
        print(f"    Total: {len(sa_data)} emails")
        print(f"    Spam: {sa_spam}, Ham: {sa_ham}")
    else:
        print(f"\n  Dataset 2: SpamAssassin (not downloaded, skipping)")
        print(f"    Run 'python download_data.py' to download additional datasets")

    # ─── Load Custom CSV Datasets ───────────────────────────────
    custom_data = load_csv_directory()
    if custom_data:
        c_spam = sum(1 for l, _ in custom_data if l == 'spam')
        c_ham = len(custom_data) - c_spam
        print(f"\n  Dataset 3: Custom CSV Datasets")
        print(f"    Total: {len(custom_data)} emails")
        print(f"    Spam/Promo: {c_spam}, Ham/Updt: {c_ham}")
    else:
        print(f"\n  Dataset 3: Custom CSV Datasets (not found, skipping)")
        print(f"    To use, drop ANY .csv files into: data/custom_datasets/")

    # ─── Merge All Datasets ─────────────────────────────────────
    all_data = sms_data + sa_data + custom_data

    total_spam = sum(1 for l, _ in all_data if l == 'spam')
    total_ham = len(all_data) - total_spam
    spam_ratio = total_spam / len(all_data) * 100 if all_data else 0

    print(f"\n  {'='*45}")
    print(f"  Combined Dataset Summary")
    print(f"  {'='*45}")
    print(f"    Total messages: {len(all_data)}")
    print(f"    Spam: {total_spam} ({spam_ratio:.1f}%)")
    print(f"    Ham:  {total_ham} ({100-spam_ratio:.1f}%)")
    print(f"    Sources: {len([1 for d in [sms_data, sa_data, custom_data] if d])} datasets")
    print(f"  {'='*45}")

    if not all_data:
        raise RuntimeError(
            "No data loaded! Run 'python download_data.py' first."
        )

    return all_data


def train_test_split(data, test_ratio=0.2, seed=42):
    """
    Split data into training and testing sets using STRATIFIED sampling.

    Statistical Rationale:
        Stratified sampling ensures that the proportion of spam and ham
        messages is preserved in both the training and test sets. This is
        critical for:
        1. Unbiased estimation of prior probabilities P(Spam) and P(Ham)
        2. Fair evaluation of classifier performance
        3. Avoiding sampling bias in imbalanced datasets

    Parameters
    ----------
    data : list of tuple
        List of (label, message) tuples.
    test_ratio : float
        Fraction of data to use for testing (default: 0.2 = 20%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (list, list)
        (train_data, test_data) each containing (label, message) tuples.
    """
    random.seed(seed)

    # Separate by class for stratified splitting
    spam_msgs = [(l, m) for l, m in data if l == 'spam']
    ham_msgs = [(l, m) for l, m in data if l == 'ham']

    # Shuffle within each class
    random.shuffle(spam_msgs)
    random.shuffle(ham_msgs)

    # Calculate split indices
    spam_test_size = int(len(spam_msgs) * test_ratio)
    ham_test_size = int(len(ham_msgs) * test_ratio)

    # Split each class
    spam_test = spam_msgs[:spam_test_size]
    spam_train = spam_msgs[spam_test_size:]

    ham_test = ham_msgs[:ham_test_size]
    ham_train = ham_msgs[ham_test_size:]

    # Combine and shuffle
    train_data = spam_train + ham_train
    test_data = spam_test + ham_test

    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"\n  [OK] Split: {len(train_data)} train, {len(test_data)} test")
    print(f"    Train - Spam: {len(spam_train)}, Ham: {len(ham_train)}")
    print(f"    Test  - Spam: {len(spam_test)}, Ham: {len(ham_test)}")

    return train_data, test_data
