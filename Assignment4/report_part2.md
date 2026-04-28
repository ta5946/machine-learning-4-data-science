# Part 2: PyTorch ANN and Extensions

## Task

Implement the same neural network as in Part 1 using PyTorch, with the goal of making the NumPy and PyTorch implementations behave as similarly as possible. Then extend one implementation with regression, regularization, and configurable activation functions.

---

## PyTorch Implementation

### Matching the NumPy Classifier

The PyTorch implementation is in `nn_pt.py`. The public interface is kept the same as in `nn.py`: `fit()` returns a model object, `predict()` returns predictions, and `weights()` returns the learned weight matrices.

The classifier uses the same network structure as Part 1: an input layer, zero or more hidden layers, and an output layer. Hidden layers use sigmoid activation by default, and the output layer returns softmax probabilities. The default training settings also match the NumPy version:

| Parameter | Value |
|---|---|
| Learning rate | 0.5 |
| Epochs | 10000 |
| Batch size | 64 |
| Random seed | 42 |
| Default activation | Sigmoid |

PyTorch stores each layer as `torch.nn.Linear`, where the ordinary weights and biases are stored separately. To keep the API compatible with `nn.py`, the `weights()` method reconstructs the folded matrix representation used in Part 1:

$$W_{\text{folded}} =
\begin{bmatrix}
b \\
W
\end{bmatrix}$$

where the first row contains the bias weights. This allows direct comparison between corresponding NumPy and PyTorch weight matrices.

### Training

Training uses plain stochastic gradient descent with no momentum, matching the update rule used manually in `nn.py`.

For classification, the model outputs raw logits during training and uses PyTorch's cross entropy loss:

$$E_{\text{class}} = \text{cross entropy}(\text{logits}, y)$$

This is equivalent to applying softmax followed by cross entropy, but PyTorch computes it in a numerically stable fused operation. During prediction, softmax is applied explicitly so `predict()` returns class probabilities, matching `nn.py`.

The implementation uses `torch.float64` throughout so that floating-point precision matches NumPy's default more closely.

---

## Code Sharing and Extensions

### Shared Classification and Regression Code

Classification and regression share most of the PyTorch implementation through the `_ANNBase` class and shared helper functions. Layer initialization, the forward pass, non-bias regularization, folded weight export, and the mini-batch training loop are all implemented once and reused. The public `ANNClassification` and `ANNRegression` classes are therefore small wrappers around the shared training code. They differ mainly in the loss function and output interpretation.

### Regression

Regression is implemented in `ANNRegression`. It uses the same hidden-layer machinery as classification, but the output layer is linear and returns one numeric value per input example.

For a normally distributed target variable, maximizing likelihood corresponds to minimizing mean squared error, so regression uses:

$$E_{\text{reg}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

The returned predictions have shape `(m,)`, matching the expectations in `test_nn.py`. The `weights()` method returns the same folded weight format as classification.

### Regularization

The `lambda_` parameter now controls L2 regularization in the PyTorch implementation. The regularized objective is:

$$E_{\text{total}} = E_{\text{data}} + \frac{\lambda}{2m} \sum_{\ell} \|W^{(\ell)}\|_2^2$$

Biases are not regularized. In PyTorch this is straightforward because `nn.Linear` stores ordinary weights in `linear.weight` and biases in `linear.bias`, so the penalty is computed only over the ordinary weight tensors. This follows the standard convention that bias terms should not be penalized.

### Activation Functions

Hidden-layer activation functions are configurable with the `activation` parameter. The implementation supports:

| Name | Function |
|---|---|
| `"sigmoid"` | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| `"relu"` | $\max(0, z)$ |

The user can pass either one activation for all hidden layers, such as `activation="relu"`, or one activation per hidden layer, such as `activation=["relu", "sigmoid"]`.

If the number of activation names does not match the number of hidden layers, the implementation raises an error. The output layer does not use a hidden activation: classification uses raw logits during training and softmax during prediction, while regression uses a linear output.

For sigmoid layers, the implementation keeps the Xavier-style initialization from Part 1:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}},\ \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$

For ReLU hidden layers, the implementation uses a fan-in based uniform initialization:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}},\ \sqrt{\frac{6}{n_{\text{in}}}}\right)$$

---

## Comparison with NumPy Implementation

### Setup

The comparison script is `compare_nn.py`. It compares `nn.py` and `nn_pt.py` on `doughnut.tab` using the minimal architecture from Part 1:

| Parameter | Value |
|---|---|
| Dataset | `doughnut.tab` |
| Architecture | `units=[3]` |
| Epochs | 5000 |
| Batch size | 64 |
| Random seed | 42 |
| Regularization | 0 |

The batch size is matched in both implementations. This is important because mini-batch gradient descent depends on how examples are grouped into batches. With the same seed and same batch size, both implementations follow the same optimization path up to floating-point precision.

### Probability and Classification Agreement

The first comparison checks the softmax output probabilities and final predicted classes.

| Metric | Value |
|---|---|
| Max probability difference | 1e-15 |
| Average probability difference | 4e-17 |
| Classification difference | 0.00% |
| `nn.py` accuracy | 100.00% |
| `nn_pt.py` accuracy | 100.00% |

The probability differences are at numerical precision. Both models classify all training examples correctly, and they assign essentially identical class probabilities.

### Loss Curves

Both implementations were trained with loss tracking enabled, logging loss every 100 epochs. The resulting plot is saved as `nn_loss_curves.png`.

The final training losses were:

| Implementation | Final training loss |
|---|---|
| `nn.py` | 0.0030 |
| `nn_pt.py` | 0.0030 |

The matching final losses and overlapping loss curves show that both implementations optimize the same objective in the same way.

### Weight Agreement

The models also expose their learned weights in the same folded matrix format, so corresponding parameters can be compared directly.

| Metric | Value |
|---|---|
| Number of weights | 17 |
| Max absolute difference | 1e-13 |
| Average absolute difference | 3e-14 |
| Pearson correlation | 1.000 |

The weight differences are also at numerical precision. The corresponding plot is saved as `nn_weight_differences.png`. This plot shows signed differences computed as:

$$W_{\text{NumPy}} - W_{\text{PyTorch}}$$

Bias weights are marked separately in the plot. Since the differences are around $10^{-14}$, the two implementations learned the same parameters for practical purposes.

### Timing

Timing was averaged over 5 runs. The PyTorch version is expected to be slower here because the networks and datasets are very small, so PyTorch's framework overhead is not amortized.

| Metric | `nn.py` | `nn_pt.py` |
|---|---:|---:|
| Fit time | 1.9s | 10.7s |
| Predict time | <0.001s | <0.001s |

The fit time ratio was:

$$\frac{t_{\text{NumPy}}}{t_{\text{PyTorch}}} = 0.17$$

This means the NumPy implementation was faster for this small training problem. Prediction times were effectively the same at the printed precision, with a ratio of 0.89.

---

## Verification

The provided tests were run after adding regression, regularization, and activation support. All tests passed:

```text
Ran 9 tests
OK
```

These tests verify:

| Test area | Result |
|---|---|
| Classification with no hidden layer | Passed |
| Classification on nonlinear XOR-style data | Passed |
| Classification with one and two hidden layers | Passed |
| Regression with no hidden layer | Passed |
| Regression on nonlinear XOR-style data | Passed |
| Regression with one and two hidden layers | Passed |
| `weights()` output shapes | Passed |

Additional checks were also run for ReLU activation, per-layer activation lists, regularized training, regression output shape, and bias-only learning under strong regularization. These confirmed that the new extension paths are usable and that regularization does not prevent bias parameters from learning.
