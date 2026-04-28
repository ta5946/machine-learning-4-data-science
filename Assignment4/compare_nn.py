import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import nn
import nn_pt

# --- Setup ---

DATASET_NAME = "doughnut.tab"
UNITS = [3]
N_EPOCHS = 5000
SEED = 42
LOG_EVERY = 100
N_TIMING_RUNS = 5

sns.set_theme(style="whitegrid")


# --- Comparison 1: probability agreement ---

def compare_probabilities(X, y):
    # Train both implementations and compare predicted probabilities.
    model_np = nn.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)
    model_pt = nn_pt.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)

    probs_np = model_np.predict(X)
    probs_pt = model_pt.predict(X)
    abs_diff = np.abs(probs_np - probs_pt)

    classes_np = np.argmax(probs_np, axis=1)
    classes_pt = np.argmax(probs_pt, axis=1)

    classification_difference = np.mean(classes_np != classes_pt) * 100
    accuracy_np = np.mean(classes_np == y) * 100
    accuracy_pt = np.mean(classes_pt == y) * 100

    print("Probability agreement on doughnut.tab:")
    print(f"  Max probability difference:     {np.max(abs_diff):.2e}")
    print(f"  Average probability difference: {np.mean(abs_diff):.2e}")
    print(f"  Classification difference:      {classification_difference:.2f}%")
    print(f"  nn.py accuracy:                 {accuracy_np:.2f}%")
    print(f"  nn_pt.py accuracy:              {accuracy_pt:.2f}%")


# --- Comparison 2: loss curves over epochs ---

def compare_loss_curves(X, y, log_every=LOG_EVERY):
    # Train with loss tracking enabled and plot both trajectories.
    print("\nTraining nn.py with loss tracking...")
    model_np = nn.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED, log_every=log_every)
    print("Training nn_pt.py with loss tracking...")
    model_pt = nn_pt.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED, log_every=log_every)

    epochs_np, losses_np = zip(*model_np.loss_history)
    epochs_pt, losses_pt = zip(*model_pt.loss_history)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_np, losses_np, label="nn.py (NumPy)", linewidth=2)
    plt.plot(epochs_pt, losses_pt, label="nn_pt.py (PyTorch)", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.yscale("log")
    plt.title("Training loss on doughnut.tab dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nn_loss_curves.png", dpi=120)
    plt.close()

    print("  Saved plot to nn_loss_curves.png")
    print(f"  nn.py final training loss:      {losses_np[-1]:.6f}")
    print(f"  nn_pt.py final training loss:   {losses_pt[-1]:.6f}")


# --- Comparison 3: weight agreement ---

def compare_weights(X, y):
    # Train both implementations and compare learned weight matrices.
    model_np = nn.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)
    model_pt = nn_pt.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)

    weights_np = model_np.weights()
    weights_pt = model_pt.weights()

    flat_np = np.concatenate([W.flatten() for W in weights_np])
    flat_pt = np.concatenate([W.flatten() for W in weights_pt])
    abs_diff = np.abs(flat_np - flat_pt)

    print("\nWeight agreement on doughnut.tab:")
    print(f"  Number of weights:              {len(flat_np)}")
    print(f"  Max absolute difference:        {np.max(abs_diff):.2e}")
    print(f"  Average absolute difference:    {np.mean(abs_diff):.2e}")
    print(f"  Pearson correlation:            {np.corrcoef(flat_np, flat_pt)[0, 1]:.6f}")

    layer_diffs = [(W_np - W_pt).flatten() for W_np, W_pt in zip(weights_np, weights_pt)]
    layer_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    plt.figure(figsize=(9, 4.8))

    start = 0
    legend_handles = []
    for layer_index, (W_np, W_pt, layer_diff, layer_color) in enumerate(
            zip(weights_np, weights_pt, layer_diffs, layer_colors), start=1):
        bias_count = W_np.shape[1]
        x_positions = np.arange(start, start + len(layer_diff))
        bars = plt.bar(
            x_positions,
            layer_diff,
            color=layer_color,
            width=0.8,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )

        for bias_bar in bars[:bias_count]:
            bias_bar.set_hatch("//")
            bias_bar.set_edgecolor("#222222")
            bias_bar.set_linewidth(1.0)

        legend_handles.append(Patch(facecolor=layer_color, edgecolor="white", label=f"Layer {layer_index}"))
        start += len(layer_diff)

    plt.xlabel("Weight index")
    plt.ylabel("Difference")
    plt.xticks(np.arange(len(flat_np)), [str(i) for i in range(len(flat_np))])
    plt.axhline(0, color="#222222", linewidth=1.2)
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.title("Differences between nn.py and nn_pt.py weights")
    legend_handles.append(Patch(facecolor="white", edgecolor="#222222", hatch="//", label="Bias"))
    plt.legend(handles=legend_handles, frameon=True)
    plt.tight_layout()
    plt.savefig("nn_weight_differences.png", dpi=120)
    plt.close()

    print("  Saved plot to nn_weight_differences.png")


# --- Comparison 4: fit and predict time ---

def compare_timing(X, y):
    # Average over several runs to reduce timing noise.
    fit_times_np = []
    fit_times_pt = []

    print(f"\nFit time with {N_EPOCHS} epochs:")

    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        model_np = nn.ANNClassification(units=UNITS, lambda_=0).fit(
            X, y, n_epochs=N_EPOCHS, seed=SEED)
        fit_times_np.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model_pt = nn_pt.ANNClassification(units=UNITS, lambda_=0).fit(
            X, y, n_epochs=N_EPOCHS, seed=SEED)
        fit_times_pt.append(time.perf_counter() - t0)

    average_fit_time_np = np.mean(fit_times_np)
    average_fit_time_pt = np.mean(fit_times_pt)

    model_np = nn.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)
    model_pt = nn_pt.ANNClassification(units=UNITS, lambda_=0).fit(
        X, y, n_epochs=N_EPOCHS, seed=SEED)

    t0 = time.perf_counter()
    for _ in range(N_TIMING_RUNS):
        model_np.predict(X)
    average_predict_time_np = (time.perf_counter() - t0) / N_TIMING_RUNS

    t0 = time.perf_counter()
    for _ in range(N_TIMING_RUNS):
        model_pt.predict(X)
    average_predict_time_pt = (time.perf_counter() - t0) / N_TIMING_RUNS

    print(f"  nn.py fit time:                {average_fit_time_np:.2f}s")
    print(f"  nn_pt.py fit time:             {average_fit_time_pt:.2f}s")
    print(f"  Fit time ratio (nn / nn_pt):   {average_fit_time_np / average_fit_time_pt:.2f}x")
    print(f"  nn.py predict time:            {average_predict_time_np:.4f}s")
    print(f"  nn_pt.py predict time:         {average_predict_time_pt:.4f}s")
    print(f"  Predict time ratio (nn / nn_pt): {average_predict_time_np / average_predict_time_pt:.2f}x")


if __name__ == "__main__":
    if DATASET_NAME == "doughnut.tab":
        X, y = nn.doughnut()
    elif DATASET_NAME == "squares.tab":
        X, y = nn.squares()
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    compare_probabilities(X, y)
    compare_loss_curves(X, y)
    compare_weights(X, y)
    compare_timing(X, y)
