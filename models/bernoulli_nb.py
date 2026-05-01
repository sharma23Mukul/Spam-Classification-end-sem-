"""
models/bernoulli_nb.py
======================
Bernoulli Naïve Bayes classifier implemented from scratch.

Model Overview:
    The Bernoulli NB models text as a BINARY feature vector where each
    word is either PRESENT (1) or ABSENT (0) in the document.
    Word frequency is ignored — only presence matters.

    Key difference from Multinomial NB:
    Bernoulli NB explicitly PENALIZES absent words by including the
    term (1 - P(word | class)) for words NOT in the document.

Likelihood Calculation:
    P(word | class) = (doc_count(word, class) + α) / (N_docs_class + 2α)

    where:
        doc_count(word, class) = number of DOCUMENTS in 'class' that
                                  contain 'word'
        N_docs_class = total number of documents in 'class'
        α = Laplace smoothing parameter
        2α in denominator because the word is binary: present or absent

    For a document d, the full likelihood is:

        P(d | class) = ∏ [P(w|c)^x_w × (1 - P(w|c))^(1 - x_w)]
                       w ∈ V

    where x_w = 1 if word w is in d, else 0.

    In log space:
        log P(d | class) = Σ  x_w × log P(w|c) + (1 - x_w) × log(1 - P(w|c))
                           w ∈ V
"""

import math
from collections import Counter
from models.base import NaiveBayesBase


class BernoulliNaiveBayes(NaiveBayesBase):
    """
    Bernoulli Naïve Bayes classifier.

    Uses binary word PRESENCE for likelihood estimation.
    Explicitly models both presence and absence of words.
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        # log P(word | class) and log(1 - P(word | class))
        self.log_prob_present = {'spam': {}, 'ham': {}}
        self.log_prob_absent = {'spam': {}, 'ham': {}}
        self.class_doc_counts = {}

    def _compute_likelihoods(self, processed_data):
        """
        Compute log P(word | class) for every word in the vocabulary.

        Uses the Bernoulli model with Laplace smoothing:

            P(w | c) = (doc_count(w, c) + α) / (N_docs_c + 2α)

        We also precompute log(1 - P(w | c)) for the absent-word penalty.

        Parameters
        ----------
        processed_data : list of tuple
            List of (label, tokens) tuples.
        """
        # Count documents per class
        self.class_doc_counts = {'spam': 0, 'ham': 0}
        for label, _ in processed_data:
            self.class_doc_counts[label] += 1

        # Count DOCUMENT frequency: how many documents of each class contain word
        doc_freq = {'spam': Counter(), 'ham': Counter()}
        for label, tokens in processed_data:
            unique_tokens = set(tokens)  # Binary: present or not
            doc_freq[label].update(unique_tokens)

        # Compute log-likelihoods for each word
        # Also precompute the "all-absent baseline" per class for fast prediction:
        #   baseline[cls] = Σ log(1 - P(w|c))  for all w ∈ V
        # Then at prediction time we start from that baseline and only adjust
        # for words that ARE present, replacing log(1-P) with log(P).
        # This turns O(|V|) per prediction into O(|tokens|).
        self._absent_baseline = {}

        for cls in self.classes:
            n_docs = self.class_doc_counts[cls]
            baseline = 0.0

            for word in self.vocabulary:
                # P(word present | class) with Laplace smoothing
                # Denominator has 2α because it's a binary (Bernoulli) variable
                numerator = doc_freq[cls].get(word, 0) + self.alpha
                denominator = n_docs + 2 * self.alpha

                prob_present = numerator / denominator
                prob_absent = 1.0 - prob_present

                log_p = math.log(prob_present)
                log_1mp = math.log(prob_absent)

                self.log_prob_present[cls][word] = log_p
                self.log_prob_absent[cls][word] = log_1mp
                baseline += log_1mp

            self._absent_baseline[cls] = baseline

        print(f"  Likelihood computation (Bernoulli):")
        print(f"    Spam documents: {self.class_doc_counts['spam']}")
        print(f"    Ham documents:  {self.class_doc_counts['ham']}")

        # Show example calculation
        example_word = 'free'
        if example_word in doc_freq['spam']:
            count = doc_freq['spam'][example_word]
            n = self.class_doc_counts['spam']
            p = (count + self.alpha) / (n + 2 * self.alpha)
            print(f"\n    Example: P('{example_word}' present | spam)")
            print(f"    = (doc_count + α) / (N_docs_spam + 2α)")
            print(f"    = ({count} + {self.alpha}) / ({n} + 2×{self.alpha})")
            print(f"    = {count + self.alpha} / {n + 2 * self.alpha}")
            print(f"    = {p:.6f}")
            print(f"    P('{example_word}' absent | spam) = {1-p:.6f}")

    def _compute_log_likelihood(self, tokens, cls):
        """
        Compute log P(message | class) for the Bernoulli model.

        Uses a precomputed "all-absent baseline" for speed.
        The baseline is: Σ log(1 - P(w|c)) over all words w ∈ V
        (i.e., assume ALL words are absent).

        Then for each word that IS present, we adjust by replacing
        the absent contribution with the present contribution:
            delta = log P(w|c) - log(1 - P(w|c))

        This gives the same result as the full O(|V|) loop but runs
        in O(|tokens|) time — ~1000x faster for large vocabularies.

            log P(msg | class) = baseline[c] + Σ [log P(w|c) - log(1-P(w|c))]
                                               w ∈ tokens ∩ V

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
        # Start from the "all words absent" baseline
        log_likelihood = self._absent_baseline[cls]

        # For each word PRESENT in the message, swap absent → present
        token_set = set(tokens) & self.vocab_set  # Only words in vocabulary

        log_present = self.log_prob_present[cls]
        log_absent = self.log_prob_absent[cls]

        for word in token_set:
            # Replace log(1-P(w|c)) with log(P(w|c))
            log_likelihood += log_present[word] - log_absent[word]

        return log_likelihood
