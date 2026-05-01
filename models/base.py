"""
models/base.py
==============
Abstract base class for Naïve Bayes classifiers.

Mathematical Foundation:
========================
Naïve Bayes applies Bayes' Theorem to classify documents:

    P(class | document) = P(document | class) × P(class) / P(document)

Since P(document) is the same for all classes, we only need:

    P(class | document) ∝ P(class) × P(document | class)

The "naïve" assumption is CONDITIONAL INDEPENDENCE of words:

    P(document | class) = ∏ P(word_i | class)
                          i

In log space (to avoid numerical underflow with many small probabilities):

    log P(class | document) = log P(class) + Σ log P(word_i | class)
                                              i

We use LAPLACE SMOOTHING (add-α) to handle unseen words:
    Without smoothing, P(unseen_word | class) = 0, which would make the
    entire product zero. With α = 1:

    P(word | class) = (count(word, class) + 1) / (total + |V|)

    where |V| is the vocabulary size.
"""

import math
from abc import ABC, abstractmethod


class NaiveBayesBase(ABC):
    """
    Abstract base class for Naïve Bayes classifiers.

    Subclasses must implement:
        - _compute_likelihoods(): how word probabilities are calculated
        - _compute_log_likelihood(): log P(document | class) for a message

    Attributes
    ----------
    alpha : float
        Laplace smoothing parameter (default: 1.0).
    classes : list
        List of class labels ['ham', 'spam'].
    log_priors : dict
        Log prior probabilities: {class: log P(class)}.
    vocabulary : list
        Sorted list of all unique words in the training corpus.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the Naïve Bayes classifier.

        Parameters
        ----------
        alpha : float
            Laplace smoothing parameter.
            α = 0: no smoothing (dangerous — unseen words get P = 0)
            α = 1: Laplace smoothing (standard choice)
            α < 1: Lidstone smoothing (less aggressive)
        """
        self.alpha = alpha
        self.classes = ['ham', 'spam']
        self.log_priors = {}
        self.vocabulary = []
        self._is_fitted = False

    def fit(self, processed_data, vocabulary):
        """
        Train the Naïve Bayes model on preprocessed data.

        This computes:
            1. Prior probabilities: P(class) = N_class / N_total
            2. Likelihoods: P(word | class) [delegated to subclass]

        Parameters
        ----------
        processed_data : list of tuple
            List of (label, tokens) after preprocessing.
        vocabulary : list of str
            Complete vocabulary of the corpus.

        Mathematical Detail:
            Prior probability represents our initial belief about class
            distribution BEFORE observing any message content.

            P(Spam) = # spam messages / # total messages
            P(Ham)  = # ham messages  / # total messages

            These are estimated using Maximum Likelihood Estimation (MLE).
        """
        self.vocabulary = vocabulary
        self.vocab_set = set(vocabulary)

        # Step 1: Calculate Prior Probabilities P(class)
        # -----------------------------------------------
        # Count messages per class
        class_counts = {cls: 0 for cls in self.classes}
        for label, _ in processed_data:
            class_counts[label] += 1

        total = len(processed_data)

        # Compute log priors: log P(class)
        # Using log to avoid underflow in multiplication
        for cls in self.classes:
            self.log_priors[cls] = math.log(class_counts[cls] / total)

        print(f"\n{'='*50}")
        print(f"  {self.__class__.__name__} — Training")
        print(f"{'='*50}")
        print(f"  Prior Probabilities:")
        for cls in self.classes:
            prob = class_counts[cls] / total
            print(f"    P({cls:>4}) = {class_counts[cls]}/{total} = {prob:.4f}")
            print(f"    log P({cls:>4}) = {self.log_priors[cls]:.4f}")
        print(f"  Vocabulary size |V| = {len(self.vocabulary)}")
        print(f"  Smoothing α = {self.alpha}")

        # Step 2: Compute Likelihoods P(word | class)
        # Delegated to subclass (Multinomial vs Bernoulli)
        self._compute_likelihoods(processed_data)

        self._is_fitted = True
        print(f"  [✓] Model trained successfully\n")

    @abstractmethod
    def _compute_likelihoods(self, processed_data):
        """Compute P(word | class) for all words. Implemented by subclasses."""
        pass

    @abstractmethod
    def _compute_log_likelihood(self, tokens, cls):
        """
        Compute log P(message | class) for a tokenized message.
        Implemented by subclasses.
        """
        pass

    def predict(self, tokens, decision_threshold=0.5):
        """
        Classify a preprocessed message using Bayes' Theorem.

        Computes the POSTERIOR probability for each class:

            log P(class | message) = log P(class) + Σ log P(word_i | class)

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens of the message.
        decision_threshold : float
            The threshold for the 'ham' class.
            If P(ham) > decision_threshold, predict 'ham'.
            Otherwise, predict 'spam'.
            Increasing this makes the model more conservative about labeling ham.

        Returns
        -------
        tuple
            (predicted_class, probabilities_dict)
            where probabilities_dict = {'spam': P(spam|msg), 'ham': P(ham|msg)}

        Mathematical Detail:
            We compute log posteriors and then convert to probabilities
            using the log-sum-exp trick for numerical stability:

            P(class | msg) = exp(log_posterior_class) / Σ exp(log_posterior_c)
                                                        c
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Compute log posterior for each class
        log_posteriors = {}
        for cls in self.classes:
            # log P(class | message) ∝ log P(class) + log P(message | class)
            log_posteriors[cls] = (
                self.log_priors[cls] +          # log P(class)
                self._compute_log_likelihood(tokens, cls)  # log P(message | class)
            )

        # Convert log posteriors to probabilities using log-sum-exp trick
        # TEMPERATURE SCALING FIX FOR NAIVE BAYES LENGTH BIAS OVERCONFIDENCE
        # For long documents, Naive Bayes creates massive differences in log-posteriors,
        # leading to 1.0 / 0.0 probabilities. We divide the log-posteriors by a temperature
        # factor proportional to the sequence length to scale them back to a realistic range.
        num_tokens = max(1, len(tokens))
        T = max(1.0, num_tokens / 10.0)
        
        scaled_posteriors = {cls: lp / T for cls, lp in log_posteriors.items()}

        # This avoids numerical overflow when exponentiating
        max_log = max(scaled_posteriors.values())
        log_sum = max_log + math.log(
            sum(math.exp(lp - max_log) for lp in scaled_posteriors.values())
        )

        probabilities = {}
        for cls in self.classes:
            probabilities[cls] = math.exp(scaled_posteriors[cls] - log_sum)

        # Apply custom decision threshold for 'ham'
        if probabilities['ham'] > decision_threshold:
            predicted = 'ham'
        else:
            predicted = 'spam'

        return predicted, probabilities

    def predict_batch(self, processed_data):
        """
        Predict labels for a batch of preprocessed messages.

        Parameters
        ----------
        processed_data : list of tuple
            List of (label, tokens) tuples.

        Returns
        -------
        tuple of (list, list, list)
            (y_true, y_pred, probabilities_list)
        """
        y_true = []
        y_pred = []
        probs_list = []

        for label, tokens in processed_data:
            pred, probs = self.predict(tokens)
            y_true.append(label)
            y_pred.append(pred)
            probs_list.append(probs)

        return y_true, y_pred, probs_list

    def predict_with_confidence(self, tokens, confidence_threshold=0.70, decision_threshold=0.5):
        """
        Classify a message WITH confidence-based uncertainty detection.

        WHY THIS MATTERS (Label Ambiguity Problem):
            In real-world datasets, some messages are genuinely BORDERLINE.
            For example, "50% off sale at your favorite store" could be:
            - HAM if the user signed up for the store's newsletter
            - SPAM if it's an unsolicited advertisement

            Different annotators may label the SAME message differently.
            Rather than forcing a binary decision on these ambiguous cases,
            we use a CONFIDENCE THRESHOLD:

            - If P(predicted_class) >= threshold → confident prediction
            - If P(predicted_class) < threshold  → mark as "uncertain"

            This is statistically more honest and prevents the model from
            making overconfident predictions on genuinely ambiguous content.

        Statistical Justification:
            When P(spam | msg) ≈ P(ham | msg) ≈ 0.5, the posterior
            provides very weak evidence for either class. The confidence
            threshold prevents acting on weak evidence.

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens of the message.
        confidence_threshold : float
            Minimum probability to consider a prediction confident.
            Default 0.70 — the predicted class must have at least 70%
            posterior probability.

        Returns
        -------
        dict
            {
                'prediction': str ('spam', 'ham', or 'uncertain'),
                'confidence': float (probability of predicted class),
                'probabilities': dict {'spam': float, 'ham': float},
                'is_confident': bool,
                'explanation': str
            }
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        pred, probs = self.predict(tokens, decision_threshold=decision_threshold)
        confidence = probs[pred]
        is_confident = confidence >= confidence_threshold

        if is_confident:
            explanation = (
                f"Confident {pred.upper()} (P={confidence:.4f} >= "
                f"threshold={confidence_threshold})"
            )
            final_pred = pred
        else:
            explanation = (
                f"UNCERTAIN - P(spam)={probs['spam']:.4f}, "
                f"P(ham)={probs['ham']:.4f}. "
                f"Neither class exceeds threshold={confidence_threshold}. "
                f"This message is ambiguous — different people might "
                f"classify it differently."
            )
            final_pred = 'uncertain'

        return {
            'prediction': final_pred,
            'confidence': confidence,
            'probabilities': probs,
            'is_confident': is_confident,
            'explanation': explanation
        }

    def analyze_label_noise(self, processed_data, confidence_threshold=0.70):
        """
        Analyze label noise / ambiguity in the dataset.

        This function identifies messages where the model's prediction
        DISAGREES with the given label AND the model is confident, as
        well as messages where the model is UNCERTAIN.

        Why this matters:
            - Disagreements where model is confident may indicate
              MISLABELED data (label noise)
            - Uncertain predictions indicate genuinely AMBIGUOUS messages
            - This analysis helps identify problematic samples that
              different human annotators might label differently

        Categories:
            1. CONFIDENT CORRECT: Model agrees with label, high confidence
            2. CONFIDENT WRONG: Model disagrees with label, high confidence
               → Possible mislabeled data or hard cases
            3. UNCERTAIN: Model is not confident about either class
               → Genuinely ambiguous messages

        Parameters
        ----------
        processed_data : list of tuple
            List of (label, tokens) tuples.
        confidence_threshold : float
            Confidence threshold for uncertainty detection.

        Returns
        -------
        dict with analysis results and example messages
        """
        confident_correct = 0
        confident_wrong = 0
        uncertain = 0

        uncertain_examples = []
        mislabel_candidates = []

        for label, tokens in processed_data:
            result = self.predict_with_confidence(tokens, confidence_threshold)

            if result['is_confident']:
                raw_pred = max(result['probabilities'],
                              key=result['probabilities'].get)
                if raw_pred == label:
                    confident_correct += 1
                else:
                    confident_wrong += 1
                    if len(mislabel_candidates) < 5:
                        mislabel_candidates.append({
                            'label': label,
                            'predicted': raw_pred,
                            'confidence': result['confidence'],
                            'tokens_preview': ' '.join(tokens[:10])
                        })
            else:
                uncertain += 1
                if len(uncertain_examples) < 5:
                    uncertain_examples.append({
                        'label': label,
                        'spam_prob': result['probabilities']['spam'],
                        'ham_prob': result['probabilities']['ham'],
                        'tokens_preview': ' '.join(tokens[:10])
                    })

        total = len(processed_data)
        return {
            'total': total,
            'confident_correct': confident_correct,
            'confident_wrong': confident_wrong,
            'uncertain': uncertain,
            'confident_correct_pct': confident_correct / total * 100,
            'confident_wrong_pct': confident_wrong / total * 100,
            'uncertain_pct': uncertain / total * 100,
            'uncertain_examples': uncertain_examples,
            'mislabel_candidates': mislabel_candidates,
        }

