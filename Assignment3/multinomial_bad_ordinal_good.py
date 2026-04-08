import numpy as np
from collections import Counter

from solution1 import MultinomialLogReg, OrdinalLogReg


# Data generating process
# Grade (nezadostno through odlicno) is determined by a latent academic score,
# thresholded at four percentile-based cutpoints to produce balanced classes.
def multinomial_bad_ordinal_good(n_samples=400, seed=42):
    rng = np.random.default_rng(seed)

    attendance = rng.uniform(0, 1, n_samples)  # fraction of classes attended
    homework = rng.uniform(0, 1, n_samples)  # fraction of assignments submitted
    study_hours = rng.uniform(0, 20, n_samples)  # hours studied per week
    physical_activity = rng.uniform(
        0, 10, n_samples
    )  # hours of physical activity per week (weak signal)

    noise = rng.logistic(0, 0.5, n_samples)
    latent = (
        1.5 * attendance
        + 1.2 * homework
        + 0.8 * (study_hours / 20)
        + 0.2 * (physical_activity / 10)
        + noise
    )

    # Thresholds are fixed before sampling, the grade distribution emerges from the data
    thresholds = [0.3, 1.2, 2.3, 3.5]
    # Prefix with digit so alphabetical order matches ordinal order
    grade_names = np.array(
        ["1_nezadostno", "2_zadostno", "3_dobro", "4_prav dobro", "5_odlicno"]
    )
    grades = grade_names[np.digitize(latent, thresholds)]

    X = np.column_stack([attendance, homework, study_hours, physical_activity])
    return X, grades, grade_names


# Metrics
def accuracy(y_true, probs, classes):
    return np.mean(classes[probs.argmax(axis=1)] == y_true)


def log_loss(y_true, probs, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx[c] for c in y_true])
    return -np.mean(np.log(probs[np.arange(len(y_true)), idx] + 1e-15))


def baseline_accuracy(y):
    # Majority class classifier
    return Counter(y).most_common(1)[0][1] / len(y)


def baseline_log_loss(y, classes):
    # Uniform classifier: 1/K probability per class
    return np.log(len(classes))


# Normalization
def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return mean, std


def scale(X, mean, std):
    return (X - mean) / std


# Bootstrap: both models on the same resamples, OOB evaluation.
def bootstrap(X, y, classes, n_boot=25, lr=0.1, n_steps=1000):
    n_samples, n_classes = X.shape[0], len(classes)

    multi_acc, multi_loss = [], []
    ordinal_acc, ordinal_loss = [], []
    base_acc, base_loss = [], []

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

        model_multi = MultinomialLogReg(lr=lr, n_steps=n_steps).build(X_boot, y_boot)
        model_ordinal = OrdinalLogReg(lr=lr, n_steps=n_steps).build(X_boot, y_boot)

        probs_multi = model_multi.predict(X_oob)
        probs_ordinal = model_ordinal.predict(X_oob)

        multi_acc.append(accuracy(y_oob, probs_multi, classes))
        multi_loss.append(log_loss(y_oob, probs_multi, classes))
        ordinal_acc.append(accuracy(y_oob, probs_ordinal, classes))
        ordinal_loss.append(log_loss(y_oob, probs_ordinal, classes))
        base_acc.append(baseline_accuracy(y_oob))
        base_loss.append(baseline_log_loss(y_oob, classes))

    return {
        "multi": {"acc": np.array(multi_acc), "loss": np.array(multi_loss)},
        "ordinal": {"acc": np.array(ordinal_acc), "loss": np.array(ordinal_loss)},
        "base": {"acc": np.array(base_acc), "loss": np.array(base_loss)},
    }


# Printing results
def confidence_interval(samples):  # returns mean, 2.5th and 97.5th percentile
    return samples.mean(), *np.percentile(samples, [2.5, 97.5])


def print_performance(results, n_samples):
    def print_model(label, acc_samples, loss_samples):
        acc_mean, acc_low, acc_high = confidence_interval(acc_samples)
        loss_mean, loss_low, loss_high = confidence_interval(loss_samples)
        print(
            f"  {label:<35} acc={acc_mean:.3f} [{acc_low:.3f}, {acc_high:.3f}]   loss={loss_mean:.3f} [{loss_low:.3f}, {loss_high:.3f}]"
        )

    print(f"\n  n={n_samples}")
    print_model("Baseline", results["base"]["acc"], results["base"]["loss"])
    print_model(
        "Multinomial logistic regression",
        results["multi"]["acc"],
        results["multi"]["loss"],
    )
    print_model(
        "Ordinal logistic regression",
        results["ordinal"]["acc"],
        results["ordinal"]["loss"],
    )


if __name__ == "__main__":
    np.random.seed(42)

    print("PERFORMANCE  (OOB evaluation, 95% bootstrap CI)")
    for n_samples in [400, 4000]:
        X, y, classes = multinomial_bad_ordinal_good(n_samples=n_samples)
        mean, std = fit_scaler(X)
        X_scaled = scale(X, mean, std)

        results = bootstrap(X_scaled, y, classes, n_boot=25)
        print_performance(results, n_samples)
