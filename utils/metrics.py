"""Evaluation metrics for few-shot classification.

The manuscript reports the average Top-1 accuracy over 100 test episodes
repeated under five fixed random seeds, with the mean and 95% confidence
interval computed over the pooled episodes (Eq.26-27).
"""

import numpy as np


def top1_accuracy(correct: int, total: int) -> float:
    """Top-1 accuracy over the pooled query samples (Eq.26-27)."""
    return correct / max(1, total)


def mean_confidence_interval(episode_accs):
    """Return (mean, std, ci95) over a pooled list of episode accuracies."""
    arr = np.asarray(episode_accs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    ci95 = 1.96 * std / np.sqrt(arr.size)
    return mean, std, ci95
