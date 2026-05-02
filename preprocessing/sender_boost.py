"""
preprocessing/sender_boost.py
=============================
Domain-based spam probability boost.

When the input text contains tokens from known spam-sender domains
(e.g. reddit, alibaba, unstop/jia, linkedin, unacademy), we apply a
significant additive boost to the final spam probability BEFORE
the threshold decision is made.

This acts as a lightweight prior override: the user has explicitly
told us these senders are spam for them, so we encode that knowledge
directly into the inference path rather than relying purely on the
Naive Bayes word likelihoods (which may not have enough signal from
these domains in the training corpus).
"""

import re

# ─── Configurable Spam Sender Rules ────────────────────────────────
# Each entry: (pattern, boost_amount)
#   pattern:      regex applied against the ORIGINAL (raw, lowercased) message text
#   boost_amount: how much to ADD to P(spam) [clamped to [0, 1] afterward]

SPAM_SENDER_RULES = [
    # Reddit notifications / digests
    (r'\breddit\b', 0.45),
    (r'\breddit\.com\b', 0.50),
    (r'\bnoreply@reddit\.com\b', 0.55),

    # Alibaba promotional emails
    (r'\balibaba\b', 0.50),
    (r'\balibaba\.com\b', 0.55),
    (r'\baliexpress\b', 0.45),

    # Unstop (formerly D2C) / Jia from Unstop
    (r'\bunstop\b', 0.45),
    (r'\bunstop\.com\b', 0.50),
    (r'\bjia\b.*\bunstop\b', 0.55),
    (r'\bjia rom\b', 0.50),

    # LinkedIn notifications
    (r'\blinkedin\b', 0.30),
    (r'\blinkedin\.com\b', 0.35),
    (r'\bjobs-listings@linkedin\b', 0.40),

    # Unacademy
    (r'\bunacademy\b', 0.35),
    (r'\bunacademy\.com\b', 0.40),
]


def compute_sender_boost(raw_message: str) -> float:
    """
    Scan the raw message text for known spam-sender patterns and
    return the MAXIMUM boost that matches.

    Parameters
    ----------
    raw_message : str
        The original, unprocessed message text.

    Returns
    -------
    float
        The spam probability boost to apply (0.0 if no patterns match).
    """
    text_lower = raw_message.lower()
    max_boost = 0.0

    for pattern, boost in SPAM_SENDER_RULES:
        if re.search(pattern, text_lower):
            max_boost = max(max_boost, boost)

    return max_boost


def apply_sender_boost(spam_prob: float, raw_message: str) -> float:
    """
    Apply the sender-based boost to a spam probability value.

    Parameters
    ----------
    spam_prob : float
        The raw P(spam) from the model (between 0 and 1).
    raw_message : str
        The original message text to scan for sender patterns.

    Returns
    -------
    float
        The boosted P(spam), clamped to [0.0, 1.0].
    """
    boost = compute_sender_boost(raw_message)
    if boost > 0:
        boosted = min(1.0, spam_prob + boost)
        return boosted
    return spam_prob
