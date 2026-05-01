import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_all_data

def analyze_length_pattern():
    print("[>>] Loading all data for length analysis...")
    data = load_all_data()
    
    df = pd.DataFrame(data, columns=['label', 'text'])
    df['length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\n[>>] Statistical Summary of Length by Class:")
    summary = df.groupby('label')[['length', 'word_count']].agg(['mean', 'median', 'min', 'max', 'std'])
    print(summary)
    
    # Analyze the pattern: Is longer more likely to be Ham?
    avg_ham_len = df[df['label'] == 'ham']['length'].mean()
    avg_spam_len = df[df['label'] == 'spam']['length'].mean()
    
    print(f"\nAverage Ham Length: {avg_ham_len:.2f} characters")
    print(f"Average Spam Length: {avg_spam_len:.2f} characters")
    
    if avg_ham_len > avg_spam_len:
        print("\nPATTERN IDENTIFIED: In this dataset, Ham messages are generally LONGER than Spam messages.")
    else:
        print("\nPATTERN IDENTIFIED: In this dataset, Spam messages are generally LONGER than Ham messages.")
        
    # Percentile analysis
    print("\n[>>] Percentile Analysis (Length):")
    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"  {int(q*100)}th percentile - Ham: {df[df['label']=='ham']['length'].quantile(q):.0f}, Spam: {df[df['label']=='spam']['length'].quantile(q):.0f}")

if __name__ == "__main__":
    analyze_length_pattern()
