import numpy as np
from collections import Counter


def accuracy(y_true, probs, classes):
    return np.mean(classes[probs.argmax(axis=1)] == y_true)


def log_loss(y_true, probs, classes):
    # Assumes classes array is sorted
    idx = np.searchsorted(classes, y_true)
    return -np.mean(np.log(probs[np.arange(len(y_true)), idx] + 1e-15))


def baseline_accuracy(y):
    # Majority class classifier
    return Counter(y).most_common(1)[0][1] / len(y)


def baseline_log_loss(y, classes):
    # Uniform classifier
    return np.log(len(classes))


def confidence_interval(samples):
    # Returns mean and confidence bounds
    return samples.mean(), *np.percentile(samples, [2.5, 97.5])
