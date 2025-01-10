"""
Metrics and validation functions for analyzing LLM coordination results.
"""

def convergence_metric(p_values, tolerance=1e-10):
    """
    Calculate Convergence Metric 1.

    Parameters:
    p_values (list or tuple): Probabilities for the four options, e.g., [0.2, 0.3, 0.4, 0.1].
    tolerance (float): A small value to account for floating-point inaccuracies.

    Returns:
    float: Convergence Metric 1 score.
    """
    if len(p_values) != 4:
        raise ValueError("p_values must contain exactly four probabilities.")

    # Calculate the probability for unanswered
    p_unanswered = 1 - sum(p_values)

    # Ensure probabilities are valid with a tolerance for floating-point inaccuracies
    if p_unanswered < -tolerance or any(p < -tolerance or p > 1 + tolerance for p in p_values):
        print(p_unanswered)
        print(p_values)
        raise ValueError("Probabilities must be non-negative and sum to 1 or less (within tolerance).")

    # Include unanswered as the fifth option
    all_p_values = list(p_values) + [max(0, p_unanswered)]  # Ensure p_unanswered is at least 0

    # Calculate the convergence metric
    convergence_score = sum(p**2 for p in all_p_values)

    return convergence_score