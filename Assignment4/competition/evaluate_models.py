import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --- Setup ---

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The neural-network implementation lives one directory above this competition folder.
from nn_pt import ANNClassification


DATA_FILE = Path(__file__).with_name("image1-competition.hdf5")

PREDICT_ROWS = slice(265, 465)
PREDICT_COLS = slice(360, 660)

COORDINATE_MODE = "image"

UNITS = [128, 64]
ACTIVATION = "relu"
LAMBDA = 1e-3
LEARNING_RATE = 0.05
N_EPOCHS = 5
BATCH_SIZE = 512
SEED = 42


# --- Data loading ---

def load_data():
    with h5py.File(DATA_FILE, "r") as f:
        data = np.array(f["data"])
        classes = np.array(f["classes"])
    return data, classes


def annotated_pixels(classes):
    # The competition labels use -1 for unlabeled pixels; these are not training examples.
    rows, cols = np.where(classes != -1)
    labels = classes[rows, cols]
    return rows, cols, labels


# --- Rectangle grid ---

def anchored_full_rectangles(image_shape):
    # Build the full 200x300 grid aligned to the competition rectangle.
    height = PREDICT_ROWS.stop - PREDICT_ROWS.start
    width = PREDICT_COLS.stop - PREDICT_COLS.start

    row_starts = []
    row_start = PREDICT_ROWS.start
    while row_start - height >= 0:
        row_start -= height
    while row_start + height <= image_shape[0]:
        row_starts.append(row_start)
        row_start += height

    col_starts = []
    col_start = PREDICT_COLS.start
    while col_start - width >= 0:
        col_start -= width
    while col_start + width <= image_shape[1]:
        col_starts.append(col_start)
        col_start += width

    rectangles = []
    for row_index, row_start in enumerate(row_starts):
        for col_index, col_start in enumerate(col_starts):
            cell_number = row_index * len(col_starts) + col_index + 1
            rectangles.append((cell_number, row_start, row_start + height, col_start, col_start + width))

    return rectangles


def selected_validation_rectangles(image_shape):
    # Validate on each full rectangle except the prediction rectangle.
    rectangles = anchored_full_rectangles(image_shape)
    return [rectangle for rectangle in rectangles if not is_prediction_rectangle(rectangle)]


def is_prediction_rectangle(rectangle):
    _cell_number, row_start, row_stop, col_start, col_stop = rectangle
    return (
        row_start == PREDICT_ROWS.start
        and row_stop == PREDICT_ROWS.stop
        and col_start == PREDICT_COLS.start
        and col_stop == PREDICT_COLS.stop
    )


def rectangle_mask(rows, cols, rectangle):
    _cell_number, row_start, row_stop, col_start, col_stop = rectangle
    return (
        (rows >= row_start)
        & (rows < row_stop)
        & (cols >= col_start)
        & (cols < col_stop)
    )


# --- Features ---

def spectrum_features(data, rows, cols):
    return data[rows, cols]


def coordinate_features(data, rows, cols):
    # Image coordinates give the model a simple global location prior.
    if COORDINATE_MODE == "image":
        row_values = rows.astype(float) / (data.shape[0] - 1)
        col_values = cols.astype(float) / (data.shape[1] - 1)
    elif COORDINATE_MODE == "rectangle":
        block_height = PREDICT_ROWS.stop - PREDICT_ROWS.start
        block_width = PREDICT_COLS.stop - PREDICT_COLS.start
        row_values = ((rows - PREDICT_ROWS.start) % block_height).astype(float) / (block_height - 1)
        col_values = ((cols - PREDICT_COLS.start) % block_width).astype(float) / (block_width - 1)
    else:
        raise ValueError(f"Unsupported coordinate mode: {COORDINATE_MODE}")

    return np.column_stack([row_values, col_values])


def spectral_coordinate_features(data, rows, cols):
    # Pixel spectrum plus normalized image position.
    return np.hstack([spectrum_features(data, rows, cols), coordinate_features(data, rows, cols)])


def patch_mean_features(data, rows, cols, radius):
    padded = np.pad(data, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    rows_padded = rows + radius
    cols_padded = cols + radius
    features = np.empty((len(rows), data.shape[-1]), dtype=data.dtype)

    for i, (row, col) in enumerate(zip(rows_padded, cols_padded)):
        patch = padded[row - radius:row + radius + 1, col - radius:col + radius + 1]
        features[i] = patch.mean(axis=(0, 1))

    return features


def local_mean_features(data, rows, cols):
    # Add the average spectrum in the 3x3 neighborhood.
    spectral_values = spectrum_features(data, rows, cols)
    mean_3x3 = patch_mean_features(data, rows, cols, radius=1)
    return np.hstack([spectral_values, mean_3x3, coordinate_features(data, rows, cols)])


def multiscale_features(data, rows, cols):
    # Add local context at several spatial scales.
    spectral_values = spectrum_features(data, rows, cols)
    mean_3x3 = patch_mean_features(data, rows, cols, radius=1)
    mean_5x5 = patch_mean_features(data, rows, cols, radius=2)
    mean_9x9 = patch_mean_features(data, rows, cols, radius=4)
    return np.hstack([spectral_values, mean_3x3, mean_5x5, mean_9x9, coordinate_features(data, rows, cols)])


# --- Metrics ---

def log_loss(y_true, probabilities):
    # Competition metric: multiclass cross entropy over predicted probabilities.
    eps = 1e-15
    selected = probabilities[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(np.clip(selected, eps, 1)))


def accuracy(y_true, probabilities):
    predicted_classes = np.argmax(probabilities, axis=1)
    return np.mean(predicted_classes == y_true)


# --- Model fitting ---

def fit_logistic_regression(X_train, y_train, _seed):
    model = LogisticRegression(max_iter=10000, C=0.001)
    model.fit(X_train, y_train)
    return model


def fit_neural_network(X_train, y_train, seed):
    fitter = ANNClassification(units=UNITS, lambda_=LAMBDA, activation=ACTIVATION)
    return fitter.fit(
        X_train,
        y_train,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def predict_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


# --- Validation ---

def evaluate_rectangle(X, labels, rows, cols, rectangle, fit_model):
    cell_number = rectangle[0]
    in_rectangle = rectangle_mask(rows, cols, rectangle)

    X_train = X[~in_rectangle]
    y_train = labels[~in_rectangle]
    X_val = X[in_rectangle]
    y_val = labels[in_rectangle]

    # Fit preprocessing only on the training pixels for this rectangle.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = fit_model(X_train, y_train, SEED + cell_number)
    probabilities = predict_probabilities(model, X_val)
    return y_val, probabilities


def evaluate_model(name, data, rows, cols, labels, rectangles, build_features, fit_model):
    rectangle_losses = []
    rectangle_accuracies = []
    pooled_labels = []
    pooled_probabilities = []

    print(f"\n{name}")
    model_start = time.perf_counter()

    # Features are deterministic, so build them once and split by rectangle later.
    X = build_features(data, rows, cols)
    feature_time = time.perf_counter() - model_start
    print(f"  Input features: {X.shape[1]}")
    print(f"  Feature time:   {feature_time:.1f}s")

    for rectangle in rectangles:
        y_val, probabilities = evaluate_rectangle(X, labels, rows, cols, rectangle, fit_model)

        rectangle_loss = log_loss(y_val, probabilities)
        rectangle_accuracy = accuracy(y_val, probabilities)
        rectangle_losses.append(rectangle_loss)
        rectangle_accuracies.append(rectangle_accuracy)
        pooled_labels.append(y_val)
        pooled_probabilities.append(probabilities)

    return print_results(rectangle_losses, rectangle_accuracies, pooled_labels, pooled_probabilities, model_start)


def print_results(rectangle_losses, rectangle_accuracies, pooled_labels, pooled_probabilities, model_start):
    rectangle_losses = np.array(rectangle_losses)
    rectangle_accuracies = np.array(rectangle_accuracies)
    pooled_labels = np.concatenate(pooled_labels)
    pooled_probabilities = np.vstack(pooled_probabilities)

    results = {
        "average_loss": rectangle_losses.mean(),
        "std_loss": rectangle_losses.std(ddof=1),
        "average_accuracy": rectangle_accuracies.mean(),
        "std_accuracy": rectangle_accuracies.std(ddof=1),
        "pooled_loss": log_loss(pooled_labels, pooled_probabilities),
        "pooled_accuracy": accuracy(pooled_labels, pooled_probabilities),
        "pooled_pixels": len(pooled_labels),
        "total_time": time.perf_counter() - model_start,
    }

    # Average metrics weight each rectangle equally; pooled metrics weight each pixel equally.
    print("  Results:")
    print(f"    Average log loss: {results['average_loss']:.4f} +/- {results['std_loss']:.4f}")
    print(f"    Average accuracy: {results['average_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    print(f"    Pooled pixels:    {results['pooled_pixels']}")
    print(f"    Pooled log loss:  {results['pooled_loss']:.4f}")
    print(f"    Pooled accuracy:  {results['pooled_accuracy']:.4f}")
    print(f"    Total time:       {results['total_time']:.1f}s")
    return results


def model_specs():
    return [
        ("Spectral LR", spectrum_features, fit_logistic_regression),
        ("Spectral NN", spectrum_features, fit_neural_network),
        ("Spectral-coordinate NN", spectral_coordinate_features, fit_neural_network),
        ("Local-mean NN", local_mean_features, fit_neural_network),
        ("Multiscale NN", multiscale_features, fit_neural_network),
    ]


if __name__ == "__main__":
    data, classes = load_data()
    rows, cols, labels = annotated_pixels(classes)
    rectangles = selected_validation_rectangles(data.shape)

    print("Anchored rectangle model evaluation")
    print(f"  Data shape:          {data.shape}")
    print(f"  Annotated examples: {len(labels)}")
    print(f"  Prediction rectangle: rows {PREDICT_ROWS.start}:{PREDICT_ROWS.stop}, "
          f"cols {PREDICT_COLS.start}:{PREDICT_COLS.stop}")
    print(f"  Validation cells:   all except prediction rectangle")
    print(f"  Coordinate mode:    {COORDINATE_MODE}")

    for name, build_features, fit_model in model_specs():
        evaluate_model(name, data, rows, cols, labels, rectangles, build_features, fit_model)
