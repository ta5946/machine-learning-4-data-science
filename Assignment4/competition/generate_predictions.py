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
OUTPUT_DIR = Path(__file__).with_name("predictions")

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


def prediction_pixels(data):
    crop = data[PREDICT_ROWS, PREDICT_COLS]
    crop_rows, crop_cols = np.indices(crop.shape[:2])
    rows = crop_rows.reshape(-1) + PREDICT_ROWS.start
    cols = crop_cols.reshape(-1) + PREDICT_COLS.start
    return rows, cols, crop.shape[:2]


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


def multiscale_5x5_features(data, rows, cols):
    # Same multiscale idea, but without the widest 9x9 context.
    spectral_values = spectrum_features(data, rows, cols)
    mean_3x3 = patch_mean_features(data, rows, cols, radius=1)
    mean_5x5 = patch_mean_features(data, rows, cols, radius=2)
    return np.hstack([spectral_values, mean_3x3, mean_5x5, coordinate_features(data, rows, cols)])


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


def fit_neural_network_128(X_train, y_train, seed):
    fitter = ANNClassification(units=[128], lambda_=LAMBDA, activation=ACTIVATION)
    return fitter.fit(
        X_train,
        y_train,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def fit_neural_network_10_epochs(X_train, y_train, seed):
    fitter = ANNClassification(units=UNITS, lambda_=LAMBDA, activation=ACTIVATION)
    return fitter.fit(
        X_train,
        y_train,
        learning_rate=LEARNING_RATE,
        n_epochs=10,
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def predict_probabilities(model, X, n_classes):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if len(model.classes_) == n_classes:
            return probabilities

        aligned = np.zeros((len(X), n_classes))
        aligned[:, model.classes_] = probabilities
        return aligned

    return model.predict(X)


def model_specs():
    return [
        ("Spectral LR", "spectral_lr.npy", spectrum_features, fit_logistic_regression),
        ("Spectral NN", "spectral_nn.npy", spectrum_features, fit_neural_network),
        ("Spectral-coordinate NN", "spectral_coordinate_nn.npy", spectral_coordinate_features, fit_neural_network),
        ("Local-mean NN", "local_mean_nn.npy", local_mean_features, fit_neural_network),
        ("Multiscale NN", "multiscale_nn.npy", multiscale_features, fit_neural_network),
        ("Spectral-coordinate LR", "spectral_coordinate_lr.npy", spectral_coordinate_features, fit_logistic_regression),
        ("Spectral-coordinate NN [128]", "spectral_coordinate_nn_128.npy", spectral_coordinate_features,
         fit_neural_network_128),
        ("Spectral-coordinate NN 10 epochs", "spectral_coordinate_nn_10epochs.npy", spectral_coordinate_features,
         fit_neural_network_10_epochs),
        ("Multiscale 5x5 NN", "multiscale_5x5_nn.npy", multiscale_5x5_features, fit_neural_network),
    ]


# --- Prediction generation ---

def generate_predictions(name, output_name, data, rows, cols, labels, predict_rows, predict_cols, predict_shape,
                         build_features, fit_model, n_classes):
    print(f"\n{name}")
    start_time = time.perf_counter()
    output_path = OUTPUT_DIR / output_name

    if output_path.exists():
        print(f"  Skipped existing file: {output_path}")
        return

    X_train = build_features(data, rows, cols)
    X_predict = build_features(data, predict_rows, predict_cols)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_predict = scaler.transform(X_predict)

    print(f"  Training examples: {len(labels)}")
    print(f"  Prediction pixels: {len(predict_rows)}")
    print(f"  Input features:    {X_train.shape[1]}")

    model = fit_model(X_train, labels, SEED)
    probabilities = predict_probabilities(model, X_predict, n_classes)
    probabilities = probabilities.reshape(predict_shape + (n_classes,))

    np.save(output_path, probabilities.astype(np.float32))

    print(f"  Saved:             {output_path}")
    print(f"  Shape:             {probabilities.shape}")
    print(f"  Total time:        {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    data, classes = load_data()
    rows, cols, labels = annotated_pixels(classes)
    predict_rows, predict_cols, predict_shape = prediction_pixels(data)
    n_classes = int(labels.max()) + 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating competition predictions")
    print(f"  Data shape:          {data.shape}")
    print(f"  Annotated examples: {len(labels)}")
    print(f"  Prediction rectangle: rows {PREDICT_ROWS.start}:{PREDICT_ROWS.stop}, "
          f"cols {PREDICT_COLS.start}:{PREDICT_COLS.stop}")
    print(f"  Coordinate mode:    {COORDINATE_MODE}")
    print(f"  Output directory:   {OUTPUT_DIR}")

    for name, output_name, build_features, fit_model in model_specs():
        generate_predictions(
            name,
            output_name,
            data,
            rows,
            cols,
            labels,
            predict_rows,
            predict_cols,
            predict_shape,
            build_features,
            fit_model,
            n_classes,
        )
