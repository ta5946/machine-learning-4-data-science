import numpy as np
import csv


# --- Activation functions ---

def sigmoid(z):
    # Squashes any value into the range (0, 1), used for hidden layer neurons.
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    # If A = sigmoid(z), then dA/dz = A * (1 - A).
    # We pass the already-computed sigmoid output to avoid recomputing it.
    return z * (1 - z)


def softmax(z):
    # Converts raw scores into a probability distribution (all outputs sum to 1).
    # We subtract the max per row for numerical stability to avoid overflow in exp.
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# --- Loss function ---

def cross_entropy_loss(predicted_probs, Y_true):
    # Measures how wrong our predictions are.
    # We add a tiny epsilon to avoid log(0) which is undefined.
    epsilon = 1e-15
    return -np.mean(np.sum(Y_true * np.log(predicted_probs + epsilon), axis=1))


# --- Helper: convert class labels to one-hot matrix ---

def one_hot_encode(y, n_classes):
    # Converts class labels like [0, 1, 2] into [[1,0,0], [0,1,0], [0,0,1]]
    m = len(y)
    Y = np.zeros((m, n_classes))
    Y[np.arange(m), y] = 1
    return Y


# --- Trained model returned by fit() ---

class ANNClassificationModel:
    def __init__(self, weights):
        # Each W in weights has shape (n_inputs + 1, n_outputs),
        # where the extra +1 row holds the bias weights.
        self._weights = weights

    def predict(self, X):
        # Run the forward pass on new data to get class probabilities.
        A = X
        for layer_index, W in enumerate(self._weights):
            # Prepend a column of 1s so the bias is included in the matrix multiply
            A_with_bias = np.hstack([np.ones((A.shape[0], 1)), A])
            Z = A_with_bias @ W

            is_last_layer = (layer_index == len(self._weights) - 1)
            A = softmax(Z) if is_last_layer else sigmoid(Z)

        return A

    def weights(self):
        return self._weights


# --- Fitter class that trains the network ---

class ANNClassification:
    def __init__(self, units, lambda_=0):
        # units: list of hidden layer sizes, [10, 5] means two hidden layers of size 10 and 5
        # lambda_: regularization strength (not used in part 1)
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y, learning_rate=0.5, n_epochs=10000, batch_size=32, seed=42):
        # Fix random seed for reproducibility
        np.random.seed(seed)
        m, n_features = X.shape
        n_classes = len(np.unique(y))

        # Convert integer class labels to one-hot matrix for loss computation
        Y = one_hot_encode(y, n_classes)

        # Full network structure: input -> hidden layers -> output
        layer_sizes = [n_features] + self.units + [n_classes]

        # Xavier initialization keeps activation variance stable across layers,
        # which is especially important for sigmoid. Each W has shape (n_in + 1, n_out)
        # where the extra row holds the bias weights.
        weights = []
        for layer_index in range(len(layer_sizes) - 1):
            n_in = layer_sizes[layer_index]
            n_out = layer_sizes[layer_index + 1]
            xavier_limit = np.sqrt(2.0 / (n_in + n_out))
            W = np.random.uniform(-xavier_limit, xavier_limit, (n_in + 1, n_out))
            weights.append(W)

        # --- Mini-batch gradient descent ---
        # Each epoch goes through all data in small batches.
        # We shuffle before each epoch so batches differ each time.
        for _ in range(n_epochs):
            shuffle_order = np.random.permutation(m)
            X_shuffled = X[shuffle_order]
            Y_shuffled = Y[shuffle_order]

            for batch_start in range(0, m, batch_size):
                X_batch = X_shuffled[batch_start:batch_start + batch_size]
                Y_batch = Y_shuffled[batch_start:batch_start + batch_size]
                batch_m = X_batch.shape[0]

                # --- Forward pass ---
                # Compute and store A at every layer; we need these in backprop.
                layer_activations = [X_batch]  # layer 0 is just the input
                A = X_batch

                for layer_index, W in enumerate(weights):
                    A_with_bias = np.hstack([np.ones((A.shape[0], 1)), A])
                    Z = A_with_bias @ W
                    is_last_layer = (layer_index == len(weights) - 1)
                    A = softmax(Z) if is_last_layer else sigmoid(Z)
                    layer_activations.append(A)

                # --- Backward pass ---
                # For softmax + cross entropy, the output error simplifies to:
                # delta = predictions - true_labels (derived in the backprop notes)
                delta = layer_activations[-1] - Y_batch

                weight_gradients = []
                for layer_index in reversed(range(len(weights))):
                    A_prev = layer_activations[layer_index]
                    A_prev_with_bias = np.hstack([np.ones((A_prev.shape[0], 1)), A_prev])

                    # Gradient for W: how much does the loss change per weight?
                    dW = A_prev_with_bias.T @ delta / batch_m
                    weight_gradients.insert(0, dW)

                    if layer_index > 0:
                        # Propagate delta back. We skip the bias row [:,1:] since
                        # bias neurons have no incoming weights.
                        delta = (delta @ weights[layer_index].T)[:, 1:] * sigmoid_derivative(
                            layer_activations[layer_index])

                # --- Weight update ---
                # Move each W in the direction that reduces the loss
                for layer_index in range(len(weights)):
                    weights[layer_index] -= learning_rate * weight_gradients[layer_index]

        return ANNClassificationModel(weights)


class ANNRegression:
    # implement me too, please
    pass


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
