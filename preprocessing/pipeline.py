"""
preprocessing/pipeline.py
=========================
Text preprocessing pipeline for spam classification.

This module handles the transformation of raw text messages into clean,
tokenized representations suitable for Naïve Bayes classification.

Preprocessing Steps:
    1. Convert to lowercase (normalize case)
    2. Remove punctuation (remove non-alphabetic characters)
    3. Remove numbers (not useful for spam detection in most cases)
    4. Tokenize (split into individual words)
    5. Remove stopwords (common words like 'the', 'is', 'at' that
       don't carry discriminative information)

Why these steps matter for Naïve Bayes:
    - Reducing vocabulary size improves probability estimates (less sparsity)
    - Removing noise (punctuation, numbers) focuses the model on
      meaningful word features
    - Stopword removal prevents common words from dominating the
      posterior probability calculations
"""

import re
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only on first run)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Cache the stopwords set for efficiency
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    """
    Apply the full preprocessing pipeline to a single text message.

    Pipeline:
        text → lowercase → remove punctuation & numbers → tokenize → 
        lemmatize (noun+verb) → remove stopwords → generate bigrams

    Parameters
    ----------
    text : str
        Raw text message.

    Returns
    -------
    list of str
        List of cleaned, lowercase tokens (unigrams + bigrams).

    Example
    -------
    >>> preprocess("FREE entry in 2 a weekly comp!")
    ['free', 'entry', 'weekly', 'comp', 'free_entry', 'entry_weekly', 'weekly_comp']

    >>> preprocess("Hey, are you coming to class?")
    ['hey', 'coming', 'class', 'hey_coming', 'coming_class']
    """
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Handle URLs and Emails (replace with special tokens)
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)

    # Step 3: Remove punctuation and numbers
    # Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Step 4: Tokenize — split text into individual words
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords + lemmatize (both noun AND verb forms)
    tokens = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos='v'), pos='n')
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    # Step 6: Generate bigrams — captures phrases like "free_offer", "click_here"
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    
    return tokens + bigrams


def preprocess_corpus(data):
    """
    Preprocess an entire corpus of labeled messages.

    Parameters
    ----------
    data : list of tuple
        List of (label, message) tuples.

    Returns
    -------
    list of tuple
        List of (label, tokens) where tokens is a list of cleaned words.

    Example
    -------
    >>> corpus = [('spam', 'Win FREE cash now!'), ('ham', 'Hey how are you?')]
    >>> processed = preprocess_corpus(corpus)
    >>> print(processed[0])
    ('spam', ['win', 'free', 'cash'])
    """
    processed = []
    for label, message in data:
        tokens = preprocess(message)
        processed.append((label, tokens))
    return processed


def build_vocabulary(processed_data, min_df=2, max_df=0.95):
    """
    Build a sorted vocabulary (set of unique words) from the preprocessed corpus.

    Filters vocabulary based on Document Frequency (DF):
        - min_df: minimum number of documents a word must appear in
        - max_df: maximum proportion of documents a word can appear in

    The vocabulary size |V| is used in Laplace smoothing:
        P(word | class) = (count(word, class) + α) / (N_class + α × |V|)

    Parameters
    ----------
    processed_data : list of tuple
        List of (label, tokens) tuples.
    min_df : int
        Minimum document frequency (remove rare words).
    max_df : float
        Maximum document frequency proportion (remove overly common words).

    Returns
    -------
    list of str
        Sorted list of unique words in the corpus.
    """
    doc_freq = Counter()
    total_docs = len(processed_data)
    
    for _, tokens in processed_data:
        doc_freq.update(set(tokens))
        
    vocab = set()
    for word, count in doc_freq.items():
        freq_prop = count / total_docs
        if count >= min_df and freq_prop <= max_df:
            vocab.add(word)
            
    return sorted(vocab)


def word_frequencies_by_class(processed_data):
    """
    Count word frequencies separately for each class (spam/ham).

    These frequency counts are the foundation of the likelihood calculation:
        P(word | class) = count(word, class) / total_words_in_class

    Parameters
    ----------
    processed_data : list of tuple
        List of (label, tokens) tuples.

    Returns
    -------
    dict
        {
            'spam': Counter({'free': 50, 'win': 30, ...}),
            'ham':  Counter({'hey': 100, 'going': 80, ...})
        }

    Example
    -------
    >>> freq = word_frequencies_by_class(processed_data)
    >>> print(freq['spam'].most_common(5))
    [('free', 50), ('call', 45), ('win', 30), ...]
    """
    freq = {
        'spam': Counter(),
        'ham': Counter()
    }

    for label, tokens in processed_data:
        freq[label].update(tokens)

    print(f"[✓] Word frequencies computed")
    print(f"    Spam vocabulary: {len(freq['spam'])} unique words, "
          f"{sum(freq['spam'].values())} total occurrences")
    print(f"    Ham vocabulary:  {len(freq['ham'])} unique words, "
          f"{sum(freq['ham'].values())} total occurrences")

    return freq


def document_frequency_by_class(processed_data):
    """
    Count DOCUMENT frequency (presence/absence) for each word per class.

    Used by Bernoulli Naïve Bayes, which cares about whether a word
    APPEARS in a document, not how many times it appears.

    Parameters
    ----------
    processed_data : list of tuple
        List of (label, tokens) tuples.

    Returns
    -------
    dict
        {
            'spam': Counter({'free': 45, 'win': 28, ...}),  # doc counts
            'ham':  Counter({'hey': 90, 'going': 75, ...})   # doc counts
        }
    """
    doc_freq = {
        'spam': Counter(),
        'ham': Counter()
    }

    for label, tokens in processed_data:
        # Use set to count each word only once per document
        unique_tokens = set(tokens)
        doc_freq[label].update(unique_tokens)

    return doc_freq
