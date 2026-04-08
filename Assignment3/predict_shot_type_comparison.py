import time
import numpy as np
import pandas as pd

from metrics import (
    accuracy,
    log_loss,
    baseline_accuracy,
    baseline_log_loss,
    confidence_interval,
)
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


# Bootstrap: both models on the same resamples, OOB evaluation.
def bootstrap(X, y, classes, n_boot=25, lr=1.0, n_steps=200):
    n_samples, n_classes = X.shape[0], len(classes)

    gd_acc, gd_loss, gd_fit_time = [], [], []
    lbfgs_acc, lbfgs_loss, lbfgs_fit_time = [], [], []
    base_acc, base_loss = [], []
    first_iteration = None

    for i in range(n_boot):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        # ~37% of samples are not drawn and serve as held-out OOB
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[idx] = False
        X_oob, y_oob = X[oob_mask], y[oob_mask]

        # Skip resamples missing a class
        if len(np.unique(y_boot)) < n_classes:
            continue

        t = time.time()
        model_gd = MultinomialLogRegGD(lr=lr, n_steps=n_steps).build(X_boot, y_boot)
        gd_fit_time.append(time.time() - t)

        t = time.time()
        model_lbfgs = MultinomialLogRegLBFGS().build(X_boot, y_boot)
        lbfgs_fit_time.append(time.time() - t)

        probs_gd = model_gd.predict(X_oob)
        probs_lbfgs = model_lbfgs.predict(X_oob)

        gd_acc.append(accuracy(y_oob, probs_gd, classes))
        gd_loss.append(log_loss(y_oob, probs_gd, classes))
        lbfgs_acc.append(accuracy(y_oob, probs_lbfgs, classes))
        lbfgs_loss.append(log_loss(y_oob, probs_lbfgs, classes))
        base_acc.append(baseline_accuracy(y_oob))
        base_loss.append(baseline_log_loss(y_oob, classes))

        if first_iteration is None:  # save for convergence, similarity, stability
            first_iteration = {
                "model_gd": model_gd,
                "model_lbfgs": model_lbfgs,
                "X_boot": X_boot,
                "y_boot": y_boot,
                "X_oob": X_oob,
                "y_oob": y_oob,
            }

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n_boot} done")

    return {
        "gd": {
            "acc": np.array(gd_acc),
            "loss": np.array(gd_loss),
            "fit_time": np.array(gd_fit_time),
        },
        "lbfgs": {
            "acc": np.array(lbfgs_acc),
            "loss": np.array(lbfgs_loss),
            "fit_time": np.array(lbfgs_fit_time),
        },
        "base": {"acc": np.array(base_acc), "loss": np.array(base_loss)},
        "first_iteration": first_iteration,
    }


# Convergence tracking
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
        nll = -log_probs.data[
            np.arange(n_samples), class_indices
        ].mean()  # neg_mean divides by n*K, so compute directly

        if step % log_every == 0:
            loss_history.append((step, float(nll)))

        if step < n_steps:
            loss.backward()
            for param in [W, b]:
                param.data -= lr * param.grad

    return loss_history


def print_performance(results):
    def print_model(label, acc_samples, loss_samples, fit_time_samples=None):
        acc_mean, acc_low, acc_high = confidence_interval(acc_samples)
        loss_mean, loss_low, loss_high = confidence_interval(loss_samples)
        time_str = (
            f"{fit_time_samples.mean():.2f}s" if fit_time_samples is not None else "—"
        )

        print(f"\n  {label}  (fit: {time_str})")
        print(f"  {'':30} {'Mean':>8}   {'95% CI'}")
        print(f"  {'Accuracy':<30} {acc_mean:>8.3f}   [{acc_low:.3f}, {acc_high:.3f}]")
        print(
            f"  {'Log-loss':<30} {loss_mean:>8.3f}   [{loss_low:.3f}, {loss_high:.3f}]"
        )

    print_model("Baseline", results["base"]["acc"], results["base"]["loss"])
    print_model(
        "Solution 1: gradient descent  (lr=1.0, steps=200)",
        results["gd"]["acc"],
        results["gd"]["loss"],
        results["gd"]["fit_time"],
    )
    print_model(
        "Solution 2: L-BFGS-B",
        results["lbfgs"]["acc"],
        results["lbfgs"]["loss"],
        results["lbfgs"]["fit_time"],
    )


def print_convergence(loss_history, lbfgs_loss):
    print(f"\n  Convergence  (L-BFGS-B train log-loss: {lbfgs_loss:.4f})")
    print(f"  {'Step':>6}  {'GD log-loss':>12}  {'L-BFGS-B':>10}")
    for step, loss in loss_history:
        print(f"  {step:>6}  {loss:>12.4f}  {lbfgs_loss:>10.4f}")


def print_similarity(probs_gd, probs_lbfgs, classes):
    pred_gd = classes[probs_gd.argmax(axis=1)]
    pred_lbfgs = classes[probs_lbfgs.argmax(axis=1)]

    print(f"\n  Prediction similarity on OOB samples (first bootstrap iteration)")
    print(f"  Class prediction agreement:      {np.mean(pred_gd == pred_lbfgs):.3f}")
    print(
        f"  Mean absolute prob difference:   {np.abs(probs_gd - probs_lbfgs).mean():.4f}"
    )


def print_stability(probs_gd, probs_lbfgs):
    print(f"\n  Numerical stability on OOB samples (first bootstrap iteration)")
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

    print("Running bootstrap (n=25)...")
    results = bootstrap(X, y, classes, n_boot=25)
    print(f"  {len(results['gd']['acc'])} valid resamples")

    first = results["first_iteration"]

    print("\nPERFORMANCE  (OOB evaluation, 95% bootstrap CI)")
    print_performance(results)

    print("\nCONVERGENCE  (first bootstrap iteration)")
    lbfgs_train_loss = log_loss(
        first["y_boot"], first["model_lbfgs"].predict(first["X_boot"]), classes
    )
    loss_history = track_convergence(
        first["X_boot"], first["y_boot"], classes, lr=1.0, n_steps=1000, log_every=50
    )
    print_convergence(loss_history, lbfgs_train_loss)

    print("\nSIMILARITY")
    probs_gd_oob = first["model_gd"].predict(first["X_oob"])
    probs_lbfgs_oob = first["model_lbfgs"].predict(first["X_oob"])
    print_similarity(probs_gd_oob, probs_lbfgs_oob, classes)

    print("\nNUMERICAL STABILITY")
    print_stability(probs_gd_oob, probs_lbfgs_oob)
