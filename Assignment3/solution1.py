import numpy as np


# AUTOGRAD
# A Node wraps a numpy array and records how gradients flow back to its inputs.
# Calling backward() on the loss traverses the graph in reverse and accumulates
# .grad on every node, giving us d(loss)/d(node) for all nodes.
class Node:

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents = parents  # nodes that this node was computed from
        self.grad_fn = None  # pushes self.grad back into each parent.grad

    def backward(self):
        # Topological sort: visit all parents before the node that depends on them,
        # so when we reverse the order, each node is processed before its parents
        topo_order = []
        visited = set()

        def visit(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node.parents:
                    visit(parent)
                topo_order.append(node)

        visit(self)
        self.grad = np.ones_like(self.data)  # d(loss)/d(loss) = 1
        for node in reversed(topo_order):
            if node.grad_fn:
                node.grad_fn()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


# When numpy broadcasts a smaller array to a larger shape during a forward op,
# its gradient must be summed back down to the original shape during backward.
def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(original_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# PRIMITIVE OPS
# Each op computes a forward value and registers a grad_fn that pushes
# the output gradient back into the input nodes via the chain rule.


def add(x, y):
    out = Node(x.data + y.data, parents=(x, y))

    def grad_fn():
        x.grad += unbroadcast(out.grad, x.data.shape)
        y.grad += unbroadcast(out.grad, y.data.shape)

    out.grad_fn = grad_fn
    return out


def mul(x, y):
    out = Node(x.data * y.data, parents=(x, y))

    def grad_fn():
        x.grad += unbroadcast(y.data * out.grad, x.data.shape)
        y.grad += unbroadcast(x.data * out.grad, y.data.shape)

    out.grad_fn = grad_fn
    return out


def matmul(x, y):
    out = Node(x.data @ y.data, parents=(x, y))

    def grad_fn():
        x.grad += out.grad @ y.data.T
        y.grad += x.data.T @ out.grad

    out.grad_fn = grad_fn
    return out


def log(x):
    out = Node(np.log(x.data + 1e-15), parents=(x,))

    def grad_fn():
        x.grad += out.grad / (x.data + 1e-15)

    out.grad_fn = grad_fn
    return out


def exp(x):
    out = Node(np.exp(x.data), parents=(x,))

    def grad_fn():
        x.grad += out.grad * out.data  # d/dx exp(x) = exp(x)

    out.grad_fn = grad_fn
    return out


def sum_all(x):
    out = Node(np.sum(x.data), parents=(x,))

    def grad_fn():
        x.grad += out.grad * np.ones_like(x.data)  # gradient flows to every element

    out.grad_fn = grad_fn
    return out


def neg_mean(x):
    # Scalar output: -mean(x), used as the last step of the negative log-likelihood (NLL)
    n_elements = x.data.size
    out = Node(-np.sum(x.data) / n_elements, parents=(x,))

    def grad_fn():
        x.grad += out.grad * (-1.0 / n_elements) * np.ones_like(x.data)

    out.grad_fn = grad_fn
    return out


# GRADIENT DESCENT
# Each step: zero grads -> compute loss and build graph -> backward -> update params
def gradient_descent(params, loss_fn, lr=0.01, n_steps=1000):
    for _ in range(n_steps):
        for param in params:
            param.zero_grad()

        loss = loss_fn()
        loss.backward()

        for param in params:
            param.data -= lr * param.grad


# HELPERS (plain numpy, not part of the autograd graph)


def softmax(logits):
    # Subtract row-wise max before exp for numerical stability
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def sigmoid(z):
    # Two-branch formula avoids overflow for both large positive and negative z
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


# MULTINOMIAL LOGISTIC REGRESSION
# Model: P(y=k | x) = softmax(X @ W + b)[k]
# Loss:  NLL = -mean_i log P(y=y_i | x_i)
class MultinomialLogReg:

    def __init__(self, lr=0.01, n_steps=1000):
        self.lr = lr
        self.n_steps = n_steps

    def build(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self.classes = classes

        # Integer class indices and one-hot targets, both shape (n_samples, n_classes)
        class_indices = np.searchsorted(classes, y)
        Y_onehot = np.eye(n_classes)[class_indices]

        # Parameters: weight matrix and bias
        W = Node(np.zeros((n_features, n_classes)))
        b = Node(np.zeros((1, n_classes)))

        def loss_fn():
            # Raw class scores, shape (n_samples, n_classes)
            logits = add(matmul(Node(X), W), b)

            # log P(y=k|x) = logit_k - log(sum_k' exp(logit_k'))
            # Computed with the max-subtraction trick for numerical stability
            row_max = logits.data.max(axis=1, keepdims=True)
            log_sum_exp_val = (
                np.log(np.exp(logits.data - row_max).sum(axis=1, keepdims=True) + 1e-15)
                + row_max
            )
            log_sum_exp = Node(log_sum_exp_val, parents=(logits,))

            def log_sum_exp_grad_fn():
                # d log(sum_k exp(logit_k)) / d logit_j = softmax(logits)_j
                logits.grad += log_sum_exp.grad * softmax(logits.data)

            log_sum_exp.grad_fn = log_sum_exp_grad_fn

            # Log-probabilities for each class, shape (n_samples, n_classes)
            log_class_probs = Node(
                logits.data - log_sum_exp.data, parents=(logits, log_sum_exp)
            )

            def log_class_probs_grad_fn():
                logits.grad += log_class_probs.grad
                log_sum_exp.grad -= log_class_probs.grad.sum(axis=1, keepdims=True)

            log_class_probs.grad_fn = log_class_probs_grad_fn

            # Zero out all classes except the true one per sample: Y_onehot * log P
            true_class_log_probs = Node(
                Y_onehot * log_class_probs.data, parents=(log_class_probs,)
            )

            def true_class_log_probs_grad_fn():
                log_class_probs.grad += Y_onehot * true_class_log_probs.grad

            true_class_log_probs.grad_fn = true_class_log_probs_grad_fn

            # NLL = -mean over samples of the true-class log-probability
            return neg_mean(true_class_log_probs)

        gradient_descent([W, b], loss_fn, lr=self.lr, n_steps=self.n_steps)

        self.W = W.data
        self.b = b.data
        return self

    def predict(self, X):
        return softmax(X @ self.W + self.b)


# ORDINAL LOGISTIC REGRESSION (proportional-odds model)
# Model: P(y <= k | x) = sigmoid(threshold_k - x @ beta)
# Class probabilities: P(y=k|x) = P(y<=k|x) - P(y<=k-1|x)
# Thresholds must be ordered: parameterised as t0, t0+exp(g0), t0+exp(g0)+exp(g1), ...
# so that raw_gaps are unconstrained and ordering is always satisfied via exp.
class OrdinalLogReg:

    def __init__(self, lr=0.01, n_steps=1000):
        self.lr = lr
        self.n_steps = n_steps

    def build(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self.classes = classes

        class_indices = np.searchsorted(classes, y)

        # Parameters
        beta = Node(np.zeros(n_features))  # regression coefficients
        t0 = Node(np.zeros(1))  # first (lowest) threshold
        raw_gaps = Node(
            np.zeros(n_classes - 2)
        )  # log-gaps that keep thresholds ordered

        def get_thresholds():
            # Thresholds: [t0, t0+exp(g0), t0+exp(g0)+exp(g1), ...]
            if n_classes > 2:
                gap_sizes = np.exp(raw_gaps.data)
                return np.concatenate([[t0.data[0]], t0.data[0] + np.cumsum(gap_sizes)])
            return t0.data.copy()  # only one threshold for binary case

        def loss_fn():
            thresholds = get_thresholds()  # shape (n_classes - 1,)

            # One score per sample: how far along the ordinal scale each sample sits
            linear_predictor = X @ beta.data  # shape (n_samples,)

            # P(y <= k | x_i) = sigmoid(threshold_k - linear_predictor_i)
            # A higher linear predictor shifts probability mass toward higher classes.
            # Shape: (n_samples, n_classes - 1)
            cum_probs = sigmoid(
                thresholds[np.newaxis, :] - linear_predictor[:, np.newaxis]
            )

            # P(y=k|x) = P(y<=k|x) - P(y<=k-1|x), with P(y<=-1)=0 and P(y<=K)=1 as boundaries
            cum_with_bounds = np.concatenate(
                [np.zeros((n_samples, 1)), cum_probs, np.ones((n_samples, 1))], axis=1
            )
            class_probs = np.clip(
                np.diff(cum_with_bounds), 1e-15, 1
            )  # shape (n_samples, n_classes)

            # NLL: negative mean log-probability of the true class
            true_class_probs = class_probs[np.arange(n_samples), class_indices]
            nll = -np.mean(np.log(true_class_probs))

            # Single custom node for the whole ordinal forward pass, with hand-written backward
            out = Node(nll, parents=(beta, t0, raw_gaps))

            def grad_fn():
                upstream = out.grad  # scalar flowing in from the loss

                # Step 1: gradient of NLL w.r.t. each true class probability
                # d(-mean log p_i) / d p_i = -1 / (n * p_i)
                dL_d_class_probs = np.zeros((n_samples, n_classes))
                dL_d_class_probs[np.arange(n_samples), class_indices] = -1.0 / (
                    n_samples * true_class_probs
                )

                # Step 2: chain through np.diff
                # class_prob[k] = cum[k] - cum[k-1], so raising cum[k] raises class_prob[k]
                # but lowers class_prob[k+1], giving: dL/d cum[k] = dL/d class_prob[k] - dL/d class_prob[k+1]
                dL_d_cum_probs = (
                    dL_d_class_probs[:, :-1] - dL_d_class_probs[:, 1:]
                )  # (n_samples, n_classes-1)

                # Step 3: chain through sigmoid
                # cum[k] = sigmoid(threshold_k - linear_predictor), so
                # d cum[k] / d (threshold_k - linear_predictor) = cum[k] * (1 - cum[k])
                dL_d_sigmoid_input = (
                    dL_d_cum_probs * cum_probs * (1 - cum_probs)
                )  # (n_samples, n_classes-1)

                # Step 4: split into gradients for beta and thresholds
                # sigmoid input = threshold_k - linear_predictor, so:
                #   d / d linear_predictor = -1  =>  chain with X.T to get grad for beta
                #   d / d threshold_k      = +1  =>  sum over samples to get grad per threshold
                beta.grad += upstream * (-X.T @ dL_d_sigmoid_input.sum(axis=1))
                dL_d_thresholds = upstream * dL_d_sigmoid_input.sum(
                    axis=0
                )  # (n_classes-1,)

                # Step 5: chain into threshold parameters
                # All thresholds share t0 (t0 shifts every threshold), so t0 gets the full sum
                t0.grad += dL_d_thresholds.sum(keepdims=True)

                # raw_gap[j] controls how far threshold[j+1] is above threshold[j], and also
                # shifts all thresholds after it, so its gradient sums over thresholds j+1 onward
                if n_classes > 2:
                    gap_sizes = np.exp(raw_gaps.data)
                    raw_gaps.grad += np.array(
                        [
                            dL_d_thresholds[gap_idx + 1 :].sum() * gap_sizes[gap_idx]
                            for gap_idx in range(n_classes - 2)
                        ]
                    )

            out.grad_fn = grad_fn
            return out

        gradient_descent(
            [beta, t0, raw_gaps], loss_fn, lr=self.lr, n_steps=self.n_steps
        )

        self.beta = beta.data.copy()
        self.thresholds = get_thresholds()
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        linear_predictor = X @ self.beta
        cum_probs = sigmoid(
            self.thresholds[np.newaxis, :] - linear_predictor[:, np.newaxis]
        )
        cum_with_bounds = np.concatenate(
            [np.zeros((n_samples, 1)), cum_probs, np.ones((n_samples, 1))], axis=1
        )
        return np.clip(np.diff(cum_with_bounds), 0, 1)


if __name__ == "__main__":
    pass
