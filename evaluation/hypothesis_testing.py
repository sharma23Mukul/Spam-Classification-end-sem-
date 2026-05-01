"""
evaluation/hypothesis_testing.py
================================
Statistical hypothesis testing and confidence interval computation.

This module provides:
    1. 95% Confidence Interval for model accuracy
    2. Paired t-test for comparing two models' cross-validation scores
    3. McNemar's test for comparing two models' predictions on the SAME test set

These are essential for making RIGOROUS statistical claims about model
performance rather than just reporting point estimates.

Why we need hypothesis testing:
    If Model A gets 96.5% accuracy and Model B gets 95.8%, can we
    conclude that A is truly better? Or could this difference be due
    to random variation? Hypothesis testing gives us a principled
    framework to answer this question.
"""

import math
from scipy import stats


def confidence_interval_95(accuracies):
    """
    Compute a 95% confidence interval for the mean accuracy.

    Uses the t-distribution (not z-distribution) because we typically
    have a small number of folds (e.g., 5), so the Central Limit Theorem
    approximation to normal is not reliable.

    Formula:
        CI = x̄ ± t_(α/2, n-1) × (s / √n)

    where:
        x̄ = sample mean
        s = sample standard deviation
        n = number of folds
        t_(α/2, n-1) = critical value from t-distribution
                       with (n-1) degrees of freedom
        α = 0.05 for 95% CI

    Interpretation:
        "We are 95% confident that the true accuracy of the model
         lies within this interval."

    Parameters
    ----------
    accuracies : list of float
        Accuracy scores from K-fold cross validation.

    Returns
    -------
    dict
        {
            'mean': float,
            'std': float,
            'ci_lower': float,
            'ci_upper': float,
            'margin_of_error': float
        }
    """
    n = len(accuracies)
    mean = sum(accuracies) / n
    variance = sum((a - mean) ** 2 for a in accuracies) / (n - 1)
    std = math.sqrt(variance)

    # Standard error of the mean
    se = std / math.sqrt(n)

    # t-critical value for 95% CI with (n-1) degrees of freedom
    # For 5 folds: t_(0.025, 4) ≈ 2.776
    t_critical = stats.t.ppf(0.975, df=n - 1)

    margin_of_error = t_critical * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    result = {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin_of_error': margin_of_error
    }

    print(f"\n  95% Confidence Interval:")
    print(f"    Mean accuracy: {mean:.4f}")
    print(f"    Std deviation: {std:.4f}")
    print(f"    Standard error: {se:.4f}")
    print(f"    t-critical (α=0.05, df={n-1}): {t_critical:.4f}")
    print(f"    Margin of error: {margin_of_error:.4f}")
    print(f"    CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    Interpretation: We are 95% confident the true accuracy")
    print(f"                    lies between {ci_lower:.4f} and {ci_upper:.4f}")

    return result


def paired_t_test(scores_a, scores_b, model_a_name="Model A", model_b_name="Model B", alpha=0.05):
    """
    Paired t-test to compare two models' cross-validation scores.

    Hypotheses:
        H₀ (Null):        μ_A = μ_B  (no difference in performance)
        H₁ (Alternative): μ_A ≠ μ_B  (there IS a difference)

    This is a TWO-SIDED test because we don't assume which model is better.

    The paired t-test is appropriate here because:
        1. We have PAIRED observations (same folds for both models)
        2. The differences between pairs should be approximately normal

    Test statistic:
        t = d̄ / (s_d / √n)

    where:
        d̄ = mean of differences (score_A_i - score_B_i)
        s_d = standard deviation of differences
        n = number of folds

    Decision rule:
        If p-value < α (0.05), REJECT H₀ → models are significantly different
        If p-value ≥ α, FAIL TO REJECT H₀ → no significant difference

    Parameters
    ----------
    scores_a : list of float
        K-fold accuracy scores for Model A.
    scores_b : list of float
        K-fold accuracy scores for Model B.
    model_a_name: str
        Name of Model A.
    model_b_name: str
        Name of Model B.
    alpha : float
        Significance level (default: 0.05).

    Returns
    -------
    dict
        {
            't_statistic': float,
            'p_value': float,
            'reject_null': bool,
            'conclusion': str
        }
    """
    # Compute paired differences
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(differences)

    # Mean and std of differences
    d_mean = sum(differences) / n
    d_var = sum((d - d_mean) ** 2 for d in differences) / (n - 1)
    d_std = math.sqrt(d_var) if d_var > 0 else 1e-10

    # t-statistic
    t_stat = d_mean / (d_std / math.sqrt(n))

    # p-value (two-sided)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    reject = p_value < alpha

    if reject:
        if d_mean > 0:
            conclusion = f"{model_a_name} is SIGNIFICANTLY BETTER than {model_b_name}"
        else:
            conclusion = f"{model_b_name} is SIGNIFICANTLY BETTER than {model_a_name}"
    else:
        conclusion = "No significant difference between the two models"

    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': reject,
        'conclusion': conclusion
    }

    print(f"\n  Paired t-test ({model_a_name} vs {model_b_name}):")
    print(f"    H₀: Both models have equal mean accuracy")
    print(f"    H₁: Models have different mean accuracies")
    print(f"    Differences: {[f'{d:.4f}' for d in differences]}")
    print(f"    Mean difference: {d_mean:.4f}")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Significance level α = {alpha}")
    print(f"    Decision: {'REJECT H₀' if reject else 'FAIL TO REJECT H₀'}")
    print(f"    → {conclusion}")

    return result


def mcnemar_test(y_true, preds_a, preds_b, model_a_name="Model A", model_b_name="Model B", alpha=0.05):
    """
    McNemar's test to compare two classifiers on the SAME dataset.

    McNemar's test looks at DISAGREEMENTS between models:
        - b = cases where A is CORRECT but B is WRONG
        - c = cases where A is WRONG but B is CORRECT

    McNemar's is more appropriate than paired t-test when we want to
    compare predictions on the SAME test set (not across folds).

    Hypotheses:
        H₀: Both models have the same error rate (b = c)
        H₁: Models have different error rates

    Test statistic (with continuity correction):
        χ² = (|b - c| - 1)² / (b + c)

    This follows a chi-squared distribution with 1 degree of freedom.

    Parameters
    ----------
    y_true : list of str
        True labels.
    preds_a : list of str
        Predictions from Model A.
    preds_b : list of str
        Predictions from Model B.
    model_a_name: str
        Name of Model A.
    model_b_name: str
        Name of Model B.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        {
            'b': int,  (A correct, B wrong)
            'c': int,  (A wrong, B correct)
            'chi2_statistic': float,
            'p_value': float,
            'reject_null': bool,
            'conclusion': str
        }
    """
    # Count disagreements
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct

    for true, pa, pb in zip(y_true, preds_a, preds_b):
        a_correct = (pa == true)
        b_correct = (pb == true)

        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # McNemar's test statistic with continuity correction
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    reject = p_value < alpha

    if reject:
        if b > c:
            conclusion = f"{model_a_name} is SIGNIFICANTLY BETTER"
        else:
            conclusion = f"{model_b_name} is SIGNIFICANTLY BETTER"
    else:
        conclusion = "No significant difference between the two models"

    result = {
        'b': b,
        'c': c,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'reject_null': reject,
        'conclusion': conclusion
    }

    print(f"\n  McNemar's Test ({model_a_name} vs {model_b_name}):")
    print(f"    H₀: Both models have the same error rate")
    print(f"    H₁: Models have different error rates")
    print(f"    Contingency:")
    print(f"      b (A correct, B wrong) = {b}")
    print(f"      c (A wrong, B correct) = {c}")
    print(f"    χ² statistic: {chi2:.4f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Significance level α = {alpha}")
    print(f"    Decision: {'REJECT H₀' if reject else 'FAIL TO REJECT H₀'}")
    print(f"    → {conclusion}")

    return result
