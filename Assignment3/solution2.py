import numpy as np
from scipy.optimize import (
    fmin_l_bfgs_b,
)  # L-BFGS-B optimizer: handles gradients and parameter updates for us

# HELPERS


def softmax(logits):
    # Subtract row-wise max before exp for numerical stability
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def sigmoid(z):
    # Two-branch formula avoids overflow for both large positive and negative z
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


# MULTINOMIAL LOGISTIC REGRESSION
# Model: P(y=k | x) = softmax(X @ W + b)[k]
# Loss:  NLL = -mean_i log P(y=y_i | x_i)
class MultinomialLogReg:

    def __init__(self, lr=None, n_steps=None):
        pass  # lr and n_steps ignored: scipy optimizer handles convergence internally

    def build(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self.classes = classes

        class_indices = np.searchsorted(classes, y)

        def nll(params):
            # Unpack flat parameter vector into W (n_features, n_classes) and b (n_classes,)
            W = params[: n_features * n_classes].reshape(n_features, n_classes)
            b = params[n_features * n_classes :]

            log_probs = np.log(softmax(X @ W + b) + 1e-15)
            return -np.mean(log_probs[np.arange(n_samples), class_indices])

        # Start from zeros; fmin_l_bfgs_b uses numerical gradients when grad is not supplied
        params0 = np.zeros(n_features * n_classes + n_classes)
        result, _, _ = fmin_l_bfgs_b(nll, params0, approx_grad=True)

        self.W = result[: n_features * n_classes].reshape(n_features, n_classes)
        self.b = result[n_features * n_classes :]
        return self

    def predict(self, X):
        return softmax(X @ self.W + self.b)


# ORDINAL LOGISTIC REGRESSION (proportional-odds model)
# Model: P(y <= k | x) = sigmoid(threshold_k - x @ beta)
# Class probabilities: P(y=k|x) = P(y<=k|x) - P(y<=k-1|x)
# Thresholds must be ordered: parameterised as t0, t0+exp(g0), t0+exp(g0)+exp(g1), ...
# so that raw_gaps are unconstrained and ordering is always satisfied via exp.
class OrdinalLogReg:

    def __init__(self, lr=None, n_steps=None):
        pass  # lr and n_steps ignored: scipy optimizer handles convergence internally

    def build(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self.classes = classes

        class_indices = np.searchsorted(classes, y)

        def get_thresholds(t0, raw_gaps):
            # Thresholds: [t0, t0+exp(g0), t0+exp(g0)+exp(g1), ...]
            if n_classes > 2:
                return np.concatenate([[t0], t0 + np.cumsum(np.exp(raw_gaps))])
            return np.array([t0])

        def nll(params):
            # Unpack flat parameter vector into beta, t0, and raw_gaps
            beta = params[:n_features]
            t0 = params[n_features]
            raw_gaps = params[n_features + 1 :]

            thresholds = get_thresholds(t0, raw_gaps)
            linear_predictor = X @ beta

            cum_probs = sigmoid(
                thresholds[np.newaxis, :] - linear_predictor[:, np.newaxis]
            )
            cum_with_bounds = np.concatenate(
                [np.zeros((n_samples, 1)), cum_probs, np.ones((n_samples, 1))], axis=1
            )
            class_probs = np.clip(np.diff(cum_with_bounds), 1e-15, 1)

            return -np.mean(np.log(class_probs[np.arange(n_samples), class_indices]))

        # Flat parameter vector: [beta, t0, raw_gaps]
        params0 = np.zeros(n_features + 1 + (n_classes - 2))
        result, _, _ = fmin_l_bfgs_b(nll, params0, approx_grad=True)

        beta = result[:n_features]
        t0 = result[n_features]
        raw_gaps = result[n_features + 1 :]

        self.beta = beta
        self.thresholds = get_thresholds(t0, raw_gaps)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        linear_predictor = X @ self.beta
        cum_probs = sigmoid(
            self.thresholds[np.newaxis, :] - linear_predictor[:, np.newaxis]
        )
        cum_with_bounds = np.concatenate(
            [np.zeros((n_samples, 1)), cum_probs, np.ones((n_samples, 1))], axis=1
        )
        return np.clip(np.diff(cum_with_bounds), 0, 1)


if __name__ == "__main__":
    pass
