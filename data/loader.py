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
from sklearn.model_selection import train_test_split as sk_train_test_split


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

    spam_classes = {'spam', 'promotion', 'promotional', 'promotions', 'phishing'}
    ham_classes = {'ham', 'update', 'updates', 'primary', 'personal', 'social', 'notification', 'forum', 'verify_code', 'social_media'}

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

def load_all_data(exclusive_file=None):
    """
    Load and MERGE all available datasets into a single corpus.

    Parameters
    ----------
    exclusive_file : str, optional
        If provided, ONLY load this specific CSV file from custom_datasets/
        and skip all other data sources (SMS, SpamAssassin, etc.).
        This is useful for hit-and-trial retraining on a single dataset.

    Returns
    -------
    list of tuple
        Each tuple is (label, message).
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))

    if exclusive_file:
        # ─── Exclusive Mode: Train ONLY on the specified file ────
        print(f"\n  [!] EXCLUSIVE MODE: Training ONLY on '{exclusive_file}'")
        custom_dir = os.path.join(data_dir, "custom_datasets")
        exclusive_path = os.path.join(custom_dir, exclusive_file)
        if not os.path.exists(exclusive_path):
            raise RuntimeError(f"Exclusive file not found: {exclusive_path}")
        
        # Load only that one file
        import csv
        import sys
        csv.field_size_limit(sys.maxsize)
        data = []
        spam_classes = {'spam', 'promotion', 'promotional', 'promotions', 'phishing'}
        ham_classes = {'ham', 'update', 'updates', 'primary', 'personal', 'social', 'notification', 'forum', 'verify_code', 'social_media'}
        with open(exclusive_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            text_col = next((c for c in reader.fieldnames if c and c.lower() in ('text', 'email', 'message', 'body', 'content', 'sms')), None)
            label_col = next((c for c in reader.fieldnames if c and c.lower() in ('label', 'category', 'class', 'type', 'target')), None)
            if not text_col or not label_col:
                raise RuntimeError(f"Could not find text/label columns in {exclusive_file}. Found: {reader.fieldnames}")
            for row in reader:
                raw_label = str(row.get(label_col, '')).strip().lower()
                text = str(row.get(text_col, '')).strip()
                if not text:
                    continue
                if raw_label in spam_classes:
                    data.append(('spam', text[:1500]))
                elif raw_label in ham_classes:
                    data.append(('ham', text[:1500]))

        total_spam = sum(1 for l, _ in data if l == 'spam')
        total_ham = len(data) - total_spam
        spam_ratio = total_spam / len(data) * 100 if data else 0
        print(f"    Total: {len(data)} messages")
        print(f"    Spam: {total_spam} ({spam_ratio:.1f}%), Ham: {total_ham} ({100-spam_ratio:.1f}%)")
        return data

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


def train_val_test_split(data, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into training, validation, and testing sets using STRATIFIED sampling via scikit-learn.

    Statistical Rationale:
        Stratified sampling ensures that the proportion of spam and ham
        messages is preserved in all sets. This is critical for unbiased evaluation.

    Parameters
    ----------
    data : list of tuple
        List of (label, message) tuples.
    val_ratio : float
        Fraction of data to use for validation (default: 0.15 = 15%).
    test_ratio : float
        Fraction of data to use for testing (default: 0.15 = 15%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (list, list, list)
        (train_data, val_data, test_data) each containing (label, message) tuples.
    """
    labels = [l for l, m in data]
    
    # First split: separate out the test set
    train_val_data, test_data, train_val_labels, _ = sk_train_test_split(
        data, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )
    
    # Second split: separate train and validation sets
    # The validation size needs to be adjusted relative to the train_val dataset
    val_adj_ratio = val_ratio / (1.0 - test_ratio)
    
    train_data, val_data = sk_train_test_split(
        train_val_data, test_size=val_adj_ratio, stratify=train_val_labels, random_state=seed
    )

    print(f"\n  [OK] Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    train_spam = sum(1 for l, m in train_data if l == 'spam')
    val_spam = sum(1 for l, m in val_data if l == 'spam')
    test_spam = sum(1 for l, m in test_data if l == 'spam')
    
    print(f"    Train - Spam: {train_spam}, Ham: {len(train_data) - train_spam}")
    print(f"    Val   - Spam: {val_spam}, Ham: {len(val_data) - val_spam}")
    print(f"    Test  - Spam: {test_spam}, Ham: {len(test_data) - test_spam}")

    return train_data, val_data, test_data
