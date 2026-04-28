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


# --- Trained model returned by fit() ---

class ANNClassificationModel:
    def __init__(self, linear_layers):
        # Each nn.Linear holds weight (n_out, n_in) and bias (n_out,),
        # corresponding to the (n_in + 1, n_out) folded matrix used in nn.py.
        self._linear_layers = linear_layers

    def predict(self, X):
        # Run the forward pass on new data to get class probabilities.
        # torch.no_grad disables autograd since we are only doing inference.
        X_tensor = torch.as_tensor(X, dtype=torch.float64).to(device)
        with torch.no_grad():
            A = X_tensor
            for layer_index, linear in enumerate(self._linear_layers):
                Z = linear(A)

                is_last_layer = (layer_index == len(self._linear_layers) - 1)
                A = softmax(Z) if is_last_layer else torch.sigmoid(Z)

        return A.cpu().numpy()

    def weights(self):
        # Reconstruct the folded (n_in + 1, n_out) matrices used in nn.py.
        folded = []
        for linear in self._linear_layers:
            W = linear.weight.detach().cpu().numpy().T  # (n_in, n_out)
            b = linear.bias.detach().cpu().numpy().reshape(1, -1)  # (1, n_out)
            folded.append(np.vstack([b, W]))  # (n_in + 1, n_out)
        return folded


# --- Fitter class that trains the network ---

class ANNClassification:
    def __init__(self, units, lambda_=0):
        # units: list of hidden layer sizes, [10, 5] means two hidden layers of size 10 and 5
        # lambda_: regularization strength (not used in part 1)
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y, learning_rate=0.5, n_epochs=10000, batch_size=64, seed=42):
        # Fix random seed for reproducibility
        np.random.seed(seed)

        m, n_features = X.shape
        n_classes = len(np.unique(y))

        # F.cross_entropy takes integer labels and fuses softmax + log + CE
        # into one kernel, so we don't need to one-hot encode y here.
        y_tensor = torch.as_tensor(y, dtype=torch.long).to(device)
        X_tensor = torch.as_tensor(X, dtype=torch.float64).to(device)

        # Full network structure: input -> hidden layers -> output
        layer_sizes = [n_features] + self.units + [n_classes]

        # Xavier initialization keeps activation variance stable across layers,
        # which is especially important for sigmoid. We draw with NumPy and
        # copy into nn.Linear (row 0 -> bias, rows 1: -> weight transposed).
        linear_layers = []
        for layer_index in range(len(layer_sizes) - 1):
            n_in = layer_sizes[layer_index]
            n_out = layer_sizes[layer_index + 1]
            xavier_limit = np.sqrt(2.0 / (n_in + n_out))
            W = np.random.uniform(-xavier_limit, xavier_limit, (n_in + 1, n_out))

            linear = torch.nn.Linear(n_in, n_out).to(device)
            with torch.no_grad():
                linear.bias.copy_(torch.as_tensor(W[0]))
                linear.weight.copy_(torch.as_tensor(W[1:].T))
            linear_layers.append(linear)

        # Plain SGD with no momentum, matching the manual gradient step in nn.py
        all_parameters = [p for linear in linear_layers for p in linear.parameters()]
        optimizer = torch.optim.SGD(all_parameters, lr=learning_rate, momentum=0)

        # --- Mini-batch gradient descent ---
        # Each epoch goes through all data in small batches.
        # We shuffle before each epoch so batches differ each time.
        for _ in range(n_epochs):
            shuffle_order = np.random.permutation(m)
            X_shuffled = X_tensor[shuffle_order]
            y_shuffled = y_tensor[shuffle_order]

            for batch_start in range(0, m, batch_size):
                X_batch = X_shuffled[batch_start:batch_start + batch_size]
                y_batch = y_shuffled[batch_start:batch_start + batch_size]

                # --- Forward pass ---
                # Output raw logits: F.cross_entropy applies softmax internally.
                A = X_batch
                for layer_index, linear in enumerate(linear_layers):
                    Z = linear(A)
                    is_last_layer = (layer_index == len(linear_layers) - 1)
                    A = Z if is_last_layer else torch.sigmoid(Z)

                loss = torch.nn.functional.cross_entropy(A, y_batch)

                # --- Backward pass and weight update ---
                # autograd computes gradients, SGD applies W <- W - lr * dW.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return ANNClassificationModel(linear_layers)


class ANNRegression:
    # implement me too, please
    pass


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
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))
    legend = content[0][1:]
    data = content[1:]
    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])
    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
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
