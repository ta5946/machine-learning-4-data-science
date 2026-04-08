import numpy as np
import pandas as pd

from metrics import (
    accuracy,
    log_loss,
    baseline_accuracy,
    baseline_log_loss,
    confidence_interval,
)
from solution1 import MultinomialLogReg


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


# Bootstrap: model and baseline on the same resamples, OOB evaluation.
def bootstrap(X, y, classes, n_boot=100, lr=1.0, n_steps=200):
    n_samples, n_classes = X.shape[0], len(classes)

    coef_samples = []
    acc_samples, loss_samples = [], []
    bacc_samples, bloss_samples = [], []

    for i in range(n_boot):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        # Skip resamples missing a class
        if len(np.unique(y_boot)) < n_classes:
            continue

        model = MultinomialLogReg(lr=lr, n_steps=n_steps).build(X_boot, y_boot)

        # ~37% of samples are not drawn and serve as held-out OOB
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[idx] = False
        X_oob, y_oob = X[oob_mask], y[oob_mask]

        probs_oob = model.predict(X_oob)

        coef_samples.append(model.W.copy())
        acc_samples.append(accuracy(y_oob, probs_oob, classes))
        loss_samples.append(log_loss(y_oob, probs_oob, classes))
        bacc_samples.append(baseline_accuracy(y_oob))
        bloss_samples.append(baseline_log_loss(y_oob, classes))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_boot} done")

    return {
        "model": {
            "acc": np.array(acc_samples),
            "loss": np.array(loss_samples),
            "coefs": np.array(coef_samples),
        },
        "base": {
            "acc": np.array(bacc_samples),
            "loss": np.array(bloss_samples),
        },
    }


# Output formatting
def print_metrics(results):
    acc_mean, acc_low, acc_high = confidence_interval(results["model"]["acc"])
    loss_mean, loss_low, loss_high = confidence_interval(results["model"]["loss"])
    bacc_mean, bacc_low, bacc_high = confidence_interval(results["base"]["acc"])
    bloss_mean, bloss_low, bloss_high = confidence_interval(results["base"]["loss"])

    w = 24
    print(
        f"\n{'Metric':<12} {'Model mean':>10}   {'95% CI':<{w}} {'Baseline mean':>13}   {'95% CI'}"
    )
    print(
        f"{'Accuracy':<12} {acc_mean:>10.3f}   [{acc_low:.3f}, {acc_high:.3f}]{'':<{w-16}} {bacc_mean:>13.3f}   [{bacc_low:.3f}, {bacc_high:.3f}]"
    )
    print(
        f"{'Log-loss':<12} {loss_mean:>10.3f}   [{loss_low:.3f}, {loss_high:.3f}]{'':<{w-16}} {bloss_mean:>13.3f}   [{bloss_low:.3f}, {bloss_high:.3f}]"
    )


def print_top_feature_class_pairs(coef_samples, feature_names, classes, top_n=40):
    mean_coefs = coef_samples.mean(axis=0)
    low_coefs = np.percentile(coef_samples, 2.5, axis=0)
    high_coefs = np.percentile(coef_samples, 97.5, axis=0)

    pairs = []
    for f, fname in enumerate(feature_names):
        for k, cls in enumerate(classes):
            pairs.append(
                (
                    abs(mean_coefs[f, k]),
                    fname,
                    cls,
                    mean_coefs[f, k],
                    low_coefs[f, k],
                    high_coefs[f, k],
                )
            )
    pairs.sort(reverse=True)

    print(f"\nTop {top_n} feature-class pairs by |coefficient|  (95% bootstrap CI)")
    print(f"  {'Feature':<30} {'Class':<12} {'Coef':>8}   {'95% CI'}")
    for _, feature_name, class_name, mean, low, high in pairs[:top_n]:
        print(
            f"  {feature_name:<30} {class_name:<12} {mean:>+8.3f}   [{low:.3f}, {high:.3f}]"
        )


if __name__ == "__main__":
    np.random.seed(42)

    X, y, feature_names = load_and_prepare("dataset.csv")
    classes = np.unique(y)

    print("Running bootstrap (n=100)...")
    results = bootstrap(X, y, classes, n_boot=100)
    print(f"  {len(results['model']['acc'])} valid resamples")

    print_metrics(results)
    print_top_feature_class_pairs(
        results["model"]["coefs"], feature_names, classes, top_n=40
    )
