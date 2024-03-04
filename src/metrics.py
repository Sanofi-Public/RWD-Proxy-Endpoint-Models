"""Metrics for ML."""

from scipy.stats import spearmanr


def spearman_score(y_true, y_pred):
    """Spearman correlation between `y_true` and `y_pred`."""

    return spearmanr(y_true, y_pred).correlation
