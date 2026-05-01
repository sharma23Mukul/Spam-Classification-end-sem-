"""
models/gaussian_nb.py
=====================
Gaussian Naïve Bayes classifier implemented from scratch.

Model Overview:
    Unlike Multinomial or Bernoulli NB which deal with discrete token counts
    or presence, Gaussian NB is used when features are CONTINUOUS.
    It assumes that the continuous features associated with each class
    are distributed according to a Gaussian (Normal) distribution.

    We engineer continuous features from the text:
        1. Email length (total words)
        2. Unique word count
        3. Average word length
        4. Count of URLs
        5. Count of Emails

Likelihood Calculation:
    P(x_i | class) = (1 / sqrt(2 * pi * var_i)) * exp( - (x_i - mean_i)^2 / (2 * var_i) )

    In log space:
    log P(x_i | class) = -0.5 * log(2 * pi * var_i) - (x_i - mean_i)^2 / (2 * var_i)

    Alpha is used as variance smoothing (epsilon) to prevent division by zero
    for features that might have zero variance.
"""

import math
from models.base import NaiveBayesBase


class GaussianNaiveBayes(NaiveBayesBase):
    """
    Gaussian Naïve Bayes classifier.
    
    Uses engineered continuous features and assumes normal distribution.
    """

    def __init__(self, alpha=1.0):
        # We use alpha for variance smoothing
        super().__init__(alpha=alpha)
        self.feature_means = {'spam': [], 'ham': []}
        self.feature_vars = {'spam': [], 'ham': []}

    def _extract_features(self, tokens):
        """Convert a list of tokens into a continuous feature vector."""
        length = len(tokens)
        unique_count = len(set(tokens))
        avg_len = sum(len(t) for t in tokens) / max(1, length)
        url_count = tokens.count('urltoken')
        email_count = tokens.count('emailtoken')
        
        return [length, unique_count, avg_len, url_count, email_count]

    def _compute_likelihoods(self, processed_data):
        """
        Compute mean and variance for each feature per class.
        """
        features_by_class = {'spam': [], 'ham': []}
        
        for label, tokens in processed_data:
            feats = self._extract_features(tokens)
            features_by_class[label].append(feats)
            
        for cls in self.classes:
            data = features_by_class[cls]
            num_samples = len(data)
            num_features = len(data[0]) if num_samples > 0 else 5
            
            means = [0.0] * num_features
            variances = [0.0] * num_features
            
            if num_samples > 0:
                for i in range(num_features):
                    # Mean
                    mean = sum(row[i] for row in data) / num_samples
                    means[i] = mean
                    
                    # Variance (population variance)
                    var = sum((row[i] - mean) ** 2 for row in data) / num_samples
                    
                    # Add variance smoothing based on alpha (epsilon)
                    # to prevent zero variance which causes division by zero
                    # We scale it by the maximum variance of the feature across all data if we were doing it globally,
                    # but a simple 1e-4 * alpha is sufficient here.
                    variances[i] = var + (self.alpha * 1e-4)
                    
            self.feature_means[cls] = means
            self.feature_vars[cls] = variances
            
        print(f"  Likelihood computation (Gaussian):")
        print(f"    Features extracted: length, unique_count, avg_len, url_count, email_count")
        print(f"    Spam means: {[round(m, 2) for m in self.feature_means['spam']]}")
        print(f"    Ham means:  {[round(m, 2) for m in self.feature_means['ham']]}")

    def _compute_log_likelihood(self, tokens, cls):
        """
        Compute log P(features | class) using Gaussian PDF.
        """
        feats = self._extract_features(tokens)
        means = self.feature_means[cls]
        vars_ = self.feature_vars[cls]
        
        log_likelihood = 0.0
        
        for i in range(len(feats)):
            x = feats[i]
            mean = means[i]
            var = vars_[i]
            
            # Gaussian PDF log
            # log( P(x|y) ) = -0.5 * log(2 * pi * var) - (x - mean)^2 / (2 * var)
            log_p = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
            log_likelihood += log_p
            
        return log_likelihood
