"""Check the data distribution in custom CSV datasets."""
import csv, sys, os
csv.field_size_limit(sys.maxsize)

spam_classes = {'spam', 'promotion', 'promotional', 'phishing'}
ham_classes = {'ham', 'update', 'updates', 'primary', 'personal', 'social', 'notification'}

for fname in ['processed_data.csv', 'kaggle_emails.csv']:
    filepath = os.path.join('data', 'custom_datasets', fname)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    print(f"\n{'='*60}")
    print(f"File: {fname} ({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")
    print(f"{'='*60}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        print(f"Columns: {reader.fieldnames}")
        
        text_col = next((c for c in reader.fieldnames if c and c.lower() in ('text', 'email', 'message', 'body', 'content', 'sms')), None)
        label_col = next((c for c in reader.fieldnames if c and c.lower() in ('label', 'category', 'class', 'type', 'target')), None)
        
        print(f"Text column: {text_col}")
        print(f"Label column: {label_col}")
        
        if not text_col or not label_col:
            print("Could not find text/label columns!")
            continue
        
        raw_labels = {}
        binary_labels = {'spam': 0, 'ham': 0, 'skipped': 0}
        total = 0
        samples = []
        
        for row in reader:
            total += 1
            raw_label = str(row.get(label_col, '')).strip().lower()
            raw_labels[raw_label] = raw_labels.get(raw_label, 0) + 1
            
            if raw_label in spam_classes:
                binary_labels['spam'] += 1
            elif raw_label in ham_classes:
                binary_labels['ham'] += 1
            elif raw_label == '1':
                binary_labels['spam'] += 1
            elif raw_label == '0':
                binary_labels['ham'] += 1
            else:
                binary_labels['skipped'] += 1
            
            if total <= 3:
                text = str(row.get(text_col, ''))[:120]
                samples.append(f"  Row {total}: label='{raw_label}', text='{text}'")
        
        print(f"\nTotal rows: {total}")
        print(f"\nRaw label distribution:")
        for lbl, cnt in sorted(raw_labels.items(), key=lambda x: -x[1]):
            print(f"  '{lbl}': {cnt} ({cnt/total*100:.1f}%)")
        
        print(f"\nBinary label mapping:")
        for lbl, cnt in binary_labels.items():
            print(f"  {lbl}: {cnt} ({cnt/total*100:.1f}%)")
        
        print(f"\nFirst 3 samples:")
        for s in samples:
            print(s)

# Also check SMS data
print(f"\n{'='*60}")
print(f"SMS Spam Collection")
print(f"{'='*60}")
sms_path = os.path.join('data', 'SMSSpamCollection.tsv')
spam_count = 0
ham_count = 0
with open(sms_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', maxsplit=1)
        if len(parts) == 2:
            label = parts[0].strip().lower()
            if label == 'spam':
                spam_count += 1
            elif label == 'ham':
                ham_count += 1
print(f"Spam: {spam_count}, Ham: {ham_count}, Total: {spam_count + ham_count}")
print(f"Spam ratio: {spam_count/(spam_count+ham_count)*100:.1f}%")
