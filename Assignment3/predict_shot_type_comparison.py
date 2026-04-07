import time
import numpy as np
import pandas as pd
from collections import Counter

from solution1 import MultinomialLogReg as MultinomialLogRegGD
from solution2 import MultinomialLogReg as MultinomialLogRegLBFGS


# Data loading and feature engineering
def load_and_prepare(path):
    df = pd.read_csv(path, sep=";")
    y = df["ShotType"].values

    X = df.drop(columns=["ShotType"])
    X = pd.get_dummies(
        X, columns=["Competition", "PlayerType", "Movement"], drop_first=True
    )
    feature_names = list(X.columns)
    X = X.astype(float).values

    # Normalize so coefficients are comparable across features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    return X, y, feature_names


# Metrics
def accuracy(y_true, probs, classes):
    return np.mean(classes[probs.argmax(axis=1)] == y_true)


def log_loss(y_true, probs, classes):
    idx = np.searchsorted(classes, y_true)
    return -np.mean(np.log(probs[np.arange(len(y_true)), idx] + 1e-15))


def baseline_accuracy(y):
    # Majority class classifier
    return Counter(y).most_common(1)[0][1] / len(y)


def baseline_log_loss(y, classes):
    # Uniform classifier: 1/K probability per class
    return np.log(len(classes))


# Train/test split
def split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Convergence tracking
# Runs gradient descent and records training NLL every log_every steps.
# Used to find how many steps are needed to match L-BFGS-B's converged loss.
def track_convergence(X_train, y_train, classes, lr=1.0, n_steps=1000, log_every=50):
    from solution1 import Node, softmax, neg_mean, matmul, add

    n_samples, n_features = X_train.shape
    n_classes = len(classes)
    class_indices = np.searchsorted(classes, y_train)
    Y_onehot = np.eye(n_classes)[class_indices]

    W = Node(np.zeros((n_features, n_classes)))
    b = Node(np.zeros((1, n_classes)))
    loss_history = []

    for step in range(n_steps + 1):
        for param in [W, b]:
            param.zero_grad()

        logits = add(matmul(Node(X_train), W), b)
        row_max = logits.data.max(axis=1, keepdims=True)
        lse_val = (
            np.log(np.exp(logits.data - row_max).sum(axis=1, keepdims=True) + 1e-15)
            + row_max
        )
        log_sum_exp = Node(lse_val, parents=(logits,))

        def lse_grad(logits=logits, log_sum_exp=log_sum_exp):
            logits.grad += log_sum_exp.grad * softmax(logits.data)

        log_sum_exp.grad_fn = lse_grad

        log_probs = Node(logits.data - log_sum_exp.data, parents=(logits, log_sum_exp))

        def lp_grad(logits=logits, log_sum_exp=log_sum_exp, log_probs=log_probs):
            logits.grad += log_probs.grad
            log_sum_exp.grad -= log_probs.grad.sum(axis=1, keepdims=True)

        log_probs.grad_fn = lp_grad

        selected = Node(Y_onehot * log_probs.data, parents=(log_probs,))

        def sel_grad(log_probs=log_probs, selected=selected):
            log_probs.grad += Y_onehot * selected.grad

        selected.grad_fn = sel_grad

        loss = neg_mean(selected)
        nll = -log_probs.data[np.arange(n_samples), class_indices].mean()

        if step % log_every == 0:
            loss_history.append((step, float(nll)))

        if step < n_steps:
            loss.backward()
            for param in [W, b]:
                param.data -= lr * param.grad

    return loss_history


# Printing results
def print_performance(
    y_train, y_test, probs_train, probs_test, classes, label, fit_time
):
    train_acc = accuracy(y_train, probs_train, classes)
    test_acc = accuracy(y_test, probs_test, classes)
    train_loss = log_loss(y_train, probs_train, classes)
    test_loss = log_loss(y_test, probs_test, classes)

    print(f"\n  {label}  (fit: {fit_time:.2f}s)")
    print(f"  {'':30} {'Train':>8}  {'Test':>8}")
    print(f"  {'Accuracy':<30} {train_acc:>8.3f}  {test_acc:>8.3f}")
    print(f"  {'Log-loss':<30} {train_loss:>8.3f}  {test_loss:>8.3f}")


def print_baseline(y_train, y_test, classes):
    print(f"\n  Baseline  (fit: 0.00s)")
    print(f"  {'':30} {'Train':>8}  {'Test':>8}")
    print(
        f"  {'Accuracy (majority class)':<30} {baseline_accuracy(y_train):>8.3f}  {baseline_accuracy(y_test):>8.3f}"
    )
    print(
        f"  {'Log-loss (uniform)':<30} {baseline_log_loss(y_train, classes):>8.3f}  {baseline_log_loss(y_test, classes):>8.3f}"
    )


def print_convergence(loss_history, lbfgs_loss):
    # Show GD loss at each checkpoint alongside L-BFGS-B's converged loss
    print(f"\n  Convergence")
    print(f"  {'Step':>6}  {'GD loss':>10}  {'L-BFGS-B loss':>14}")
    for step, loss in loss_history:
        print(f"  {step:>6}  {loss:>10.4f}  {lbfgs_loss:>14.4f}")


def print_similarity(probs_gd, probs_lbfgs, classes):
    pred_gd = classes[probs_gd.argmax(axis=1)]
    pred_lbfgs = classes[probs_lbfgs.argmax(axis=1)]

    print(f"\n  Prediction similarity on test set (gradient descent vs L-BFGS-B)")
    print(f"  Class prediction agreement:      {np.mean(pred_gd == pred_lbfgs):.3f}")
    print(
        f"  Mean absolute prob difference:   {np.abs(probs_gd - probs_lbfgs).mean():.4f}"
    )


def print_stability(probs_gd, probs_lbfgs):
    print(f"\n  Numerical stability on test set")
    print(f"  {'':30} {'GD':>8}  {'L-BFGS-B':>10}")
    print(
        f"  {'NaN in probs':<30} {str(np.isnan(probs_gd).any()):>8}  {str(np.isnan(probs_lbfgs).any()):>10}"
    )
    print(
        f"  {'Probs near zero (< 1e-6)':<30} {(probs_gd < 1e-6).sum():>8}  {(probs_lbfgs < 1e-6).sum():>10}"
    )
    print(
        f"  {'Probs near one (> 1-1e-6)':<30} {(probs_gd > 1-1e-6).sum():>8}  {(probs_lbfgs > 1-1e-6).sum():>10}"
    )


if __name__ == "__main__":
    np.random.seed(42)

    X, y, feature_names = load_and_prepare("dataset.csv")
    classes = np.unique(y)

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    # Solution 1: gradient descent (200 steps, not fully converged — see convergence section)
    t = time.time()
    model_gd = MultinomialLogRegGD(lr=1.0, n_steps=200).build(X_train, y_train)
    time_gd = time.time() - t
    probs_gd_train = model_gd.predict(X_train)
    probs_gd_test = model_gd.predict(X_test)

    # Solution 2: L-BFGS-B (fully converged)
    t = time.time()
    model_lbfgs = MultinomialLogRegLBFGS().build(X_train, y_train)
    time_lbfgs = time.time() - t
    probs_lbfgs_train = model_lbfgs.predict(X_train)
    probs_lbfgs_test = model_lbfgs.predict(X_test)

    print("\nPERFORMANCE")
    print_baseline(y_train, y_test, classes)
    print_performance(
        y_train,
        y_test,
        probs_gd_train,
        probs_gd_test,
        classes,
        "Solution 1: gradient descent  (lr=1.0, steps=200)",
        fit_time=time_gd,
    )
    print_performance(
        y_train,
        y_test,
        probs_lbfgs_train,
        probs_lbfgs_test,
        classes,
        "Solution 2: L-BFGS-B",
        fit_time=time_lbfgs,
    )

    print("\nCONVERGENCE")
    lbfgs_train_loss = log_loss(y_train, probs_lbfgs_train, classes)
    loss_history = track_convergence(
        X_train, y_train, classes, lr=1.0, n_steps=1000, log_every=50
    )
    print_convergence(loss_history, lbfgs_train_loss)

    print("\nSIMILARITY")
    print_similarity(probs_gd_test, probs_lbfgs_test, classes)

    print("\nNUMERICAL STABILITY")
    print_stability(probs_gd_test, probs_lbfgs_test)
