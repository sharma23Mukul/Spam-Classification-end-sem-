"""
models/multinomial_nb.py
========================
Multinomial Naïve Bayes classifier implemented from scratch.

Model Overview:
    The Multinomial NB models text as a BAG OF WORDS where each word's
    FREQUENCY (count) matters. A message like "free free call" would
    count 'free' twice and 'call' once.

    This is suitable when word repetition carries information —
    e.g., a message saying "free" 5 times is more likely spam than
    one saying "free" once.

Likelihood Calculation:
    P(word | class) = (count(word, class) + α) / (N_class + α × |V|)

    where:
        count(word, class) = number of times 'word' appears in all
                             documents of 'class'
        N_class = total number of word occurrences in 'class'
        α = Laplace smoothing parameter (typically 1)
        |V| = vocabulary size (number of unique words)

    The denominator (N_class + α × |V|) ensures all probabilities
    sum to 1 across the vocabulary.
"""

import math
from collections import Counter
from models.base import NaiveBayesBase


class MultinomialNaiveBayes(NaiveBayesBase):
    """
    Multinomial Naïve Bayes classifier.

    Uses word FREQUENCY counts for likelihood estimation.
    Best suited for text where word count matters.
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        # log P(word | class) for each word and class
        self.log_likelihoods = {'spam': {}, 'ham': {}}
        # Total word count per class
        self.class_word_counts = {}

    def _compute_likelihoods(self, processed_data):
        """
        Compute log P(word | class) for every word in the vocabulary.

        Uses the Multinomial model with Laplace smoothing:

            P(w | c) = (count(w, c) + α) / (Σ count(w', c) + α × |V|)
                                             w'

        We store log P(w | c) to avoid underflow when multiplying
        many small probabilities.

        Parameters
        ----------
        processed_data : list of tuple
            List of (label, tokens) tuples.
        """
        vocab_size = len(self.vocabulary)

        # Count word frequencies for each class
        word_counts = {'spam': Counter(), 'ham': Counter()}
        for label, tokens in processed_data:
            word_counts[label].update(tokens)

        # Total word occurrences per class
        for cls in self.classes:
            self.class_word_counts[cls] = sum(word_counts[cls].values())

        # Compute log-likelihood for every word in vocabulary
        for cls in self.classes:
            n_class = self.class_word_counts[cls]
            denominator = n_class + self.alpha * vocab_size

            for word in self.vocabulary:
                # count(word, class) + α  (Laplace smoothing)
                numerator = word_counts[cls].get(word, 0) + self.alpha

                # log P(word | class)
                self.log_likelihoods[cls][word] = math.log(numerator / denominator)

        print(f"  Likelihood computation (Multinomial):")
        print(f"    Total words in spam: {self.class_word_counts['spam']}")
        print(f"    Total words in ham:  {self.class_word_counts['ham']}")

        # Show example calculation for a common spam word
        example_word = 'free'
        if example_word in word_counts['spam']:
            count = word_counts['spam'][example_word]
            n = self.class_word_counts['spam']
            p = (count + self.alpha) / (n + self.alpha * vocab_size)
            print(f"\n    Example: P('{example_word}' | spam)")
            print(f"    = (count + α) / (N_spam + α × |V|)")
            print(f"    = ({count} + {self.alpha}) / ({n} + {self.alpha} × {vocab_size})")
            print(f"    = {count + self.alpha} / {n + self.alpha * vocab_size}")
            print(f"    = {p:.6f}")

    def _compute_log_likelihood(self, tokens, cls):
        """
        Compute log P(message | class) for the Multinomial model.

        For Multinomial NB, we sum log P(word | class) for EVERY token
        occurrence in the message (including duplicates).

            log P(msg | class) = Σ  count(w, msg) × log P(w | class)
                                 w ∈ msg

        For words not in the vocabulary, we use Laplace smoothing:
            P(unknown | class) = α / (N_class + α × |V|)

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens of the message.
        cls : str
            Class label ('spam' or 'ham').

        Returns
        -------
        float
            log P(message | class)
        """
        log_likelihood = 0.0
        vocab_size = len(self.vocabulary)

        for token in tokens:
            if token in self.log_likelihoods[cls]:
                log_likelihood += self.log_likelihoods[cls][token]
            else:
                # Handle unseen words with Laplace smoothing
                # P(unseen | class) = α / (N_class + α × |V|)
                n_class = self.class_word_counts[cls]
                log_likelihood += math.log(
                    self.alpha / (n_class + self.alpha * vocab_size)
                )

        return log_likelihood
