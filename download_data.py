"""
download_data.py
================
Downloads MULTIPLE spam classification datasets for diverse training:

1. SMS Spam Collection (UCI) — 5,574 SMS messages
2. SpamAssassin Public Corpus — ~6,000+ emails (ham + spam)
3. Enron Spam Dataset (subset) — email messages
4. HuggingFace Marketing-Emails — promotional emails
5. HuggingFace high-accuracy-email-classifier — 13,477 labeled emails

Using multiple datasets from different domains (SMS + Email) makes the
model more ROBUST and generalizable. This is important because:
    - A model trained only on SMS might not recognize email spam patterns
    - A model trained only on email might miss SMS-specific spam tactics
    - Combining datasets provides a wider vocabulary and more diverse
      spam/ham patterns for better probability estimates

Reference:
    SMS: Almeida & Gomez Hidalgo, UCI ML Repository, 2012
    SpamAssassin: Apache SpamAssassin Public Corpus
    Enron: Metsis, Androutsopoulos & Paliouras, CEAS 2006
"""

import os
import zipfile
import tarfile
import urllib.request
import sys
import email
import email.policy

# ─── Configuration ───────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Dataset URLs
SMS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

SPAMASSASSIN_URLS = {
    'ham1': "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    'ham2': "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    'spam1': "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    'spam2': "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
}

HF_TOKEN = "your_huggingface_token_here"  # Get from settings or env


def download_sms_dataset():
    """
    Downloads and extracts the SMS Spam Collection dataset.

    Returns the path to the TSV file, or None if download fails.
    This dataset contains 5,574 labeled SMS messages.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    tsv_path = os.path.join(DATA_DIR, "SMSSpamCollection.tsv")

    if os.path.exists(tsv_path):
        print(f"[OK] SMS dataset already exists at: {tsv_path}")
        return tsv_path

    print(f"[>>] Downloading SMS Spam Collection dataset...")
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")

    try:
        urllib.request.urlretrieve(SMS_URL, zip_path)
        print(f"[OK] Downloaded SMS dataset")
    except Exception as e:
        print(f"[!!] SMS download failed: {e}")
        print("    You can manually download from:")
        print("    https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        return None

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    extracted = os.path.join(DATA_DIR, "SMSSpamCollection")
    if os.path.exists(extracted) and not os.path.exists(tsv_path):
        os.rename(extracted, tsv_path)

    # Cleanup
    for f in [zip_path, os.path.join(DATA_DIR, "readme")]:
        if os.path.exists(f):
            os.remove(f)

    print(f"[OK] SMS dataset ready: {tsv_path}")
    return tsv_path


def download_spamassassin_dataset():
    """
    Downloads and extracts the SpamAssassin public corpus.

    The SpamAssassin corpus contains real EMAIL messages, providing a
    different domain from SMS for more robust training.

    Categories downloaded:
        - easy_ham: clearly legitimate emails (~2,500)
        - hard_ham: legitimate emails that look like spam (~250)
        - spam: spam emails (~500)
        - spam_2: more spam emails (~1,400)

    Returns the path to the extracted directory, or None on failure.
    """
    sa_dir = os.path.join(DATA_DIR, "spamassassin")
    marker = os.path.join(sa_dir, ".download_complete")

    if os.path.exists(marker):
        print(f"[OK] SpamAssassin dataset already exists at: {sa_dir}")
        return sa_dir

    os.makedirs(sa_dir, exist_ok=True)
    print(f"[>>] Downloading SpamAssassin Public Corpus...")

    for name, url in SPAMASSASSIN_URLS.items():
        tar_path = os.path.join(sa_dir, f"{name}.tar.bz2")
        print(f"    Downloading {name}...")

        try:
            urllib.request.urlretrieve(url, tar_path)
        except Exception as e:
            print(f"    [!!] Failed to download {name}: {e}")
            print(f"    Skipping {name}, continuing with other datasets...")
            continue

        # Extract tar.bz2
        try:
            with tarfile.open(tar_path, 'r:bz2') as tar:
                tar.extractall(sa_dir)
            print(f"    [OK] Extracted {name}")
        except Exception as e:
            print(f"    [!!] Failed to extract {name}: {e}")
            continue

        # Cleanup tar file
        if os.path.exists(tar_path):
            os.remove(tar_path)

    # Create marker file
    with open(marker, 'w') as f:
        f.write("download complete")

    print(f"[OK] SpamAssassin dataset ready: {sa_dir}")
    return sa_dir


def _extract_email_body(filepath):
    """
    Extract the text body from a raw email file.

    Emails can have complex MIME structures (multipart, attachments, etc).
    We extract only the plain text content, which is what our NB model
    can work with.

    Parameters
    ----------
    filepath : str
        Path to the raw email file.

    Returns
    -------
    str or None
        The email body text, or None if extraction fails.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        # Parse email
        msg = email.message_from_string(raw, policy=email.policy.default)

        # Get body text
        body = msg.get_body(preferencelist=('plain',))
        if body:
            text = body.get_content()
            if isinstance(text, str) and len(text.strip()) > 10:
                return text.strip()

        # Fallback: try to get any text content
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        text = part.get_payload(decode=True)
                        if text:
                            return text.decode('utf-8', errors='ignore').strip()
                    except Exception:
                        continue
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode('utf-8', errors='ignore').strip()

    except Exception:
        pass

    return None


def load_spamassassin_data(sa_dir):
    """
    Parse the SpamAssassin corpus into (label, message) tuples.

    The corpus is organized as:
        sa_dir/easy_ham/     → ham emails
        sa_dir/hard_ham/     → ham emails (borderline cases)
        sa_dir/spam/         → spam emails
        sa_dir/spam_2/       → more spam emails

    Parameters
    ----------
    sa_dir : str
        Path to the SpamAssassin directory.

    Returns
    -------
    list of tuple
        List of (label, message) tuples.
    """
    data = []

    # Map directories to labels
    label_dirs = {
        'ham': ['easy_ham', 'hard_ham'],
        'spam': ['spam', 'spam_2']
    }

    for label, dir_names in label_dirs.items():
        for dir_name in dir_names:
            dir_path = os.path.join(sa_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue

            for filename in os.listdir(dir_path):
                if filename.startswith('.') or filename == 'cmds':
                    continue

                filepath = os.path.join(dir_path, filename)
                if not os.path.isfile(filepath):
                    continue

                body = _extract_email_body(filepath)
                if body and len(body) > 20:
                    # Truncate very long emails to keep processing manageable
                    # This also helps Naive Bayes focus on the most relevant content
                    data.append((label, body[:2000]))

    return data


def download_hf_marketing_dataset():
    """
    Downloads the 'marketeam/Marketing-Emails' dataset from HuggingFace
    and saves it to the custom_datasets folder.
    """
    import csv
    custom_dir = os.path.join(DATA_DIR, "custom_datasets")
    os.makedirs(custom_dir, exist_ok=True)
    csv_path = os.path.join(custom_dir, "marketing_emails.csv")

    if os.path.exists(csv_path):
        print(f"[OK] Marketing dataset already exists at: {csv_path}")
        return csv_path

    print(f"[>>] Downloading Marketing-Emails from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset('marketeam/Marketing-Emails', split='train')
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'text'])
            for row in ds:
                text = str(row.get('0', '')).strip()
                if text:
                    writer.writerow(['spam', text])
        print(f"[OK] Downloaded and formatted Marketing dataset: {csv_path}")
        return csv_path
    except ImportError:
        print(f"[!!] Failed to import 'datasets'. Run 'pip install datasets'.")
        return None
    except Exception as e:
        print(f"[!!] Failed to download Marketing dataset: {e}")
        return None


def download_hf_email_classifier_dataset(token=None):
    """
    Downloads the 'jason23322/high-accuracy-email-classifier' gated dataset
    from HuggingFace and saves it as a CSV for training.
    
    Category mapping (binary):
        spam  -> spam, promotions
        ham   -> forum, verify_code, social_media, updates
    """
    import csv
    custom_dir = os.path.join(DATA_DIR, "custom_datasets")
    os.makedirs(custom_dir, exist_ok=True)
    csv_path = os.path.join(custom_dir, "high_accuracy_emails.csv")

    if os.path.exists(csv_path):
        print(f"[OK] High-Accuracy Email Classifier dataset already exists at: {csv_path}")
        return csv_path

    print(f"[>>] Downloading jason23322/high-accuracy-email-classifier from HuggingFace...")
    try:
        from datasets import load_dataset

        SPAM_CATEGORIES = {'spam', 'promotions'}
        HAM_CATEGORIES = {'forum', 'verify_code', 'social_media', 'updates'}

        total_written = 0
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'text'])

            for split_name in ['train', 'test']:
                ds = load_dataset(
                    'jason23322/high-accuracy-email-classifier',
                    split=split_name,
                    token=token
                )
                for row in ds:
                    cat = str(row.get('category', '')).strip().lower()
                    text = str(row.get('text', '')).strip()
                    if not text:
                        continue

                    if cat in SPAM_CATEGORIES:
                        binary_label = 'spam'
                    elif cat in HAM_CATEGORIES:
                        binary_label = 'ham'
                    else:
                        continue

                    writer.writerow([binary_label, text])
                    total_written += 1

        print(f"[OK] Downloaded High-Accuracy Email Classifier: {total_written} emails -> {csv_path}")
        return csv_path
    except ImportError:
        print(f"[!!] Failed to import 'datasets'. Run 'pip install datasets'.")
        return None
    except Exception as e:
        print(f"[!!] Failed to download High-Accuracy Email Classifier: {e}")
        return None


def download_hf_enron_dataset():
    """
    Downloads the 'SetFit/enron_spam' cleaned dataset from HuggingFace
    and saves it to the custom_datasets folder.
    
    This replaces the messy legacy Enron sample with a high-quality,
    properly labeled version.
    """
    import csv
    custom_dir = os.path.join(DATA_DIR, "custom_datasets")
    os.makedirs(custom_dir, exist_ok=True)
    csv_path = os.path.join(custom_dir, "enron_cleaned.csv")

    if os.path.exists(csv_path):
        print(f"[OK] Clean Enron dataset already exists at: {csv_path}")
        return csv_path

    print(f"[>>] Downloading Clean Enron Dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        # Load train and test splits
        ds_train = load_dataset('SetFit/enron_spam', split='train')
        ds_test = load_dataset('SetFit/enron_spam', split='test')
        
        total_written = 0
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'text'])
            
            for ds in [ds_train, ds_test]:
                for row in ds:
                    text = str(row.get('text', '')).strip()
                    # 1 is spam, 0 is ham in this dataset
                    label = 'spam' if row.get('label') == 1 else 'ham'
                    if text:
                        writer.writerow([label, text])
                        total_written += 1
                        
        print(f"[OK] Downloaded and formatted Clean Enron dataset: {total_written} emails -> {csv_path}")
        return csv_path
    except ImportError:
        print(f"[!!] Failed to import 'datasets'. Run 'pip install datasets'.")
        return None
    except Exception as e:
        print(f"[!!] Failed to download Clean Enron dataset: {e}")
        return None


def download_dataset():
    """
    Download ALL datasets and return paths.

    This function orchestrates the download of multiple datasets
    for comprehensive training.

    Returns
    -------
    dict
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "="*55)
    print("  Downloading Datasets for Training")
    print("="*55)

    sms_path = download_sms_dataset()
    sa_dir = download_spamassassin_dataset()
    mkt_path = download_hf_marketing_dataset()
    hf_path = download_hf_email_classifier_dataset(token=HF_TOKEN)
    enron_path = download_hf_enron_dataset()

    print("\n" + "="*55)
    print("  Dataset Download Summary")
    print("="*55)
    print(f"  SMS Spam Collection: {'Ready' if sms_path else 'Failed'}")
    print(f"  SpamAssassin Corpus: {'Ready' if sa_dir else 'Failed'}")
    print(f"  Marketing Emails: {'Ready' if mkt_path else 'Failed'}")
    print(f"  High-Accuracy Classifier: {'Ready' if hf_path else 'Failed'}")
    print(f"  Clean Enron Dataset: {'Ready' if enron_path else 'Failed'}")
    print("="*55 + "\n")

    return {
        'sms_path': sms_path,
        'spamassassin_dir': sa_dir,
        'marketing_path': mkt_path,
        'hf_email_path': hf_path,
        'enron_clean_path': enron_path,
    }


if __name__ == "__main__":
    download_dataset()
