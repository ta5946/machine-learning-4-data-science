import numpy as np
import torch
import csv

# Use float64 throughout so the implementation matches NumPy's default precision.
torch.set_default_dtype(torch.float64)

# CPU is faster than GPU here because the networks are too small to amortize launch overhead
device = torch.device("cpu")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Activation functions ---

def softmax(z):
    # Converts raw scores into a probability distribution (all outputs sum to 1).
    return torch.softmax(z, dim=1)


def apply_activation(z, activation):
    if activation == "sigmoid":
        return torch.sigmoid(z)
    if activation == "relu":
        return torch.relu(z)
    raise ValueError(f"Unsupported activation: {activation}")


def resolve_activations(activation, n_hidden_layers):
    # A single activation name is reused for every hidden layer.
    if isinstance(activation, str):
        activations = [activation] * n_hidden_layers
    else:
        activations = list(activation)

    if len(activations) != n_hidden_layers:
        raise ValueError("Number of activations must match number of hidden layers")

    for name in activations:
        apply_activation(torch.zeros(1), name)

    return activations


# --- Shared network helpers ---

def initialize_layers(layer_sizes, activations):
    # Draw the same folded weight matrices as nn.py, then copy them into nn.Linear.
    linear_layers = []
    for layer_index in range(len(layer_sizes) - 1):
        n_in = layer_sizes[layer_index]
        n_out = layer_sizes[layer_index + 1]
        activation = activations[layer_index] if layer_index < len(activations) else "sigmoid"

        if activation == "relu":
            limit = np.sqrt(6.0 / n_in)
        else:
            limit = np.sqrt(2.0 / (n_in + n_out))

        W = np.random.uniform(-limit, limit, (n_in + 1, n_out))

        linear = torch.nn.Linear(n_in, n_out).to(device)
        with torch.no_grad():
            linear.bias.copy_(torch.as_tensor(W[0]))
            linear.weight.copy_(torch.as_tensor(W[1:].T))
        linear_layers.append(linear)

    return linear_layers


def forward_raw(X_tensor, linear_layers, activations):
    # Hidden layers use the requested activations; the output layer stays raw.
    A = X_tensor
    for layer_index, linear in enumerate(linear_layers):
        Z = linear(A)
        is_last_layer = (layer_index == len(linear_layers) - 1)
        A = Z if is_last_layer else apply_activation(Z, activations[layer_index])
    return A


def regularization_loss(linear_layers, lambda_, batch_m):
    # Only regularize ordinary weights, not bias parameters.
    if lambda_ == 0:
        return 0

    penalty = sum(torch.sum(linear.weight ** 2) for linear in linear_layers)
    return lambda_ * penalty / (2 * batch_m)


def folded_weights(linear_layers):
    # Reconstruct the folded (n_in + 1, n_out) matrices used in nn.py.
    folded = []
    for linear in linear_layers:
        W = linear.weight.detach().cpu().numpy().T  # (n_in, n_out)
        b = linear.bias.detach().cpu().numpy().reshape(1, -1)  # (1, n_out)
        folded.append(np.vstack([b, W]))  # (n_in + 1, n_out)
    return folded


# --- Trained models returned by fit() ---

class ANNClassificationModel:
    def __init__(self, linear_layers, activations, loss_history=None):
        self._linear_layers = linear_layers
        self._activations = activations
        # List of (epoch, loss) pairs if log_every was set during fit, else empty.
        self.loss_history = loss_history if loss_history is not None else []

    def predict(self, X):
        # Run the forward pass on new data to get class probabilities.
        X_tensor = torch.as_tensor(X, dtype=torch.float64).to(device)
        with torch.no_grad():
            logits = forward_raw(X_tensor, self._linear_layers, self._activations)
            probs = softmax(logits)
        return probs.cpu().numpy()

    def weights(self):
        return folded_weights(self._linear_layers)


class ANNRegressionModel:
    def __init__(self, linear_layers, activations, loss_history=None):
        self._linear_layers = linear_layers
        self._activations = activations
        # List of (epoch, loss) pairs if log_every was set during fit, else empty.
        self.loss_history = loss_history if loss_history is not None else []

    def predict(self, X):
        # Run the forward pass on new data to get numeric predictions.
        X_tensor = torch.as_tensor(X, dtype=torch.float64).to(device)
        with torch.no_grad():
            predictions = forward_raw(X_tensor, self._linear_layers, self._activations)
        return predictions.cpu().numpy().reshape(-1)

    def weights(self):
        return folded_weights(self._linear_layers)


# --- Shared fitter class ---

class _ANNBase:
    def __init__(self, units, lambda_=0, activation="sigmoid"):
        # units: list of hidden layer sizes, [10, 5] means two hidden layers of size 10 and 5
        # lambda_: regularization strength; bias weights are not regularized
        self.units = units
        self.lambda_ = lambda_
        self.activation = activation

    def _fit(self, X, y, task, learning_rate, n_epochs, batch_size, seed, log_every):
        # Fix random seed for reproducibility
        np.random.seed(seed)

        m, n_features = X.shape
        activations = resolve_activations(self.activation, len(self.units))

        if task == "classification":
            n_outputs = len(np.unique(y))
            y_tensor = torch.as_tensor(y, dtype=torch.long).to(device)
        elif task == "regression":
            n_outputs = 1
            y_tensor = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1).to(device)
        else:
            raise ValueError(f"Unsupported task: {task}")

        X_tensor = torch.as_tensor(X, dtype=torch.float64).to(device)

        # Full network structure: input -> hidden layers -> output
        layer_sizes = [n_features] + self.units + [n_outputs]
        linear_layers = initialize_layers(layer_sizes, activations)

        # Plain SGD with no momentum, matching the manual gradient step in nn.py.
        all_parameters = [p for linear in linear_layers for p in linear.parameters()]
        optimizer = torch.optim.SGD(all_parameters, lr=learning_rate, momentum=0)

        # --- Mini-batch gradient descent ---
        # Each epoch goes through all data in small batches.
        # We shuffle before each epoch so batches differ each time.
        loss_history = []
        for epoch in range(n_epochs):
            shuffle_order = np.random.permutation(m)
            X_shuffled = X_tensor[shuffle_order]
            y_shuffled = y_tensor[shuffle_order]

            for batch_start in range(0, m, batch_size):
                X_batch = X_shuffled[batch_start:batch_start + batch_size]
                y_batch = y_shuffled[batch_start:batch_start + batch_size]
                batch_m = X_batch.shape[0]

                output = forward_raw(X_batch, linear_layers, activations)
                if task == "classification":
                    loss = torch.nn.functional.cross_entropy(output, y_batch)
                elif task == "regression":
                    loss = torch.nn.functional.mse_loss(output, y_batch)
                else:
                    raise ValueError(f"Unsupported task: {task}")
                loss = loss + regularization_loss(linear_layers, self.lambda_, batch_m)

                # --- Backward pass and weight update ---
                # autograd computes gradients, SGD applies W <- W - lr * dW.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log full-data loss every log_every epochs
            if log_every is not None and epoch % log_every == 0:
                with torch.no_grad():
                    output = forward_raw(X_tensor, linear_layers, activations)
                    if task == "classification":
                        loss_val = torch.nn.functional.cross_entropy(output, y_tensor).item()
                    elif task == "regression":
                        loss_val = torch.nn.functional.mse_loss(output, y_tensor).item()
                    else:
                        raise ValueError(f"Unsupported task: {task}")
                    loss_history.append((epoch, loss_val))

        if log_every is not None and (not loss_history or loss_history[-1][0] != n_epochs):
            with torch.no_grad():
                output = forward_raw(X_tensor, linear_layers, activations)
                if task == "classification":
                    loss_val = torch.nn.functional.cross_entropy(output, y_tensor).item()
                elif task == "regression":
                    loss_val = torch.nn.functional.mse_loss(output, y_tensor).item()
                else:
                    raise ValueError(f"Unsupported task: {task}")
                loss_history.append((n_epochs, loss_val))

        if task == "classification":
            return ANNClassificationModel(linear_layers, activations, loss_history)
        return ANNRegressionModel(linear_layers, activations, loss_history)


# --- Fitter classes that train the network ---

class ANNClassification(_ANNBase):
    def fit(self, X, y, learning_rate=0.5, n_epochs=10000, batch_size=64, seed=42, log_every=None):
        return self._fit(X, y, "classification", learning_rate, n_epochs, batch_size, seed, log_every)


class ANNRegression(_ANNBase):
    def fit(self, X, y, learning_rate=0.5, n_epochs=10000, batch_size=64, seed=42, log_every=None):
        return self._fit(X, y, "regression", learning_rate, n_epochs, batch_size, seed, log_every)


# --- Helper: classification accuracy ---

def accuracy(model, X, y):
    # Fraction of samples where the predicted class matches the true class
    predicted_classes = np.argmax(model.predict(X), axis=1)
    return np.mean(predicted_classes == y)


# --- Helper: count total weights in a network ---

def count_weights(units, n_features, n_classes):
    # Total number of weights including bias weights across all weight matrices
    layer_sizes = [n_features] + units + [n_classes]
    return sum((layer_sizes[i] + 1) * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))


# --- Helper: find the minimal network for a dataset ---

def find_minimal_network(X, y, dataset_name, n_epochs=5000):
    # Search all single hidden layer networks up to 5 neurons and all
    # two hidden layer networks up to 3 neurons per layer.
    # 5000 epochs is sufficient for convergence on simple 2D datasets.
    configs = [
        [1], [2], [3], [4], [5],
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3],
    ]

    print(f"\nFitting {dataset_name}:")
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    best_units = None
    best_weights = np.inf

    for units in configs:
        model = ANNClassification(units=units, lambda_=0).fit(X, y, n_epochs=n_epochs)
        acc = accuracy(model, X, y)
        n_weights = count_weights(units, n_features, n_classes)
        print(f"  units={units}: accuracy={acc:.3f}, weights={n_weights}")
        if acc == 1.0 and n_weights < best_weights:
            best_weights = n_weights
            best_units = units

    print(f"  Minimal network: units={best_units} with {best_weights} weights")


# --- Data reading ---

def read_tab(fn, adict):
    with open(fn, "rt") as f:
        content = list(csv.reader(f, delimiter="\t"))
    legend = content[0][1:]
    data = content[1:]
    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])
    return legend, X, y


def doughnut():
    _legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    _legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


if __name__ == "__main__":
    print(f"Using device: {device}")

    # --- Template test ---
    fitter = ANNClassification(units=[3, 4], lambda_=0)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
    print("Template test passed!")

    # Load datasets used for fitting
    X_d, y_d = doughnut()
    X_s, y_s = squares()

    # --- Fit doughnut.tab and squares.tab ---
    find_minimal_network(X_d, y_d, "doughnut.tab")
    find_minimal_network(X_s, y_s, "squares.tab")
