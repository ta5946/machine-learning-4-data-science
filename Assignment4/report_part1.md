# Part 1: ANNClassification

## Task

Implement a multi-layer fully-connected artificial neural network for classification in the "old-school" style, without autograd, using analytically derived gradients and gradient descent for optimization. The implementation must support an arbitrary number of hidden layers of any size, include a bias at each layer, use sigmoid activation functions in hidden layers, and be structured so that the corresponding tests in `test_nn.py` pass. Only `numpy` is allowed as an external library.

---

## Implementation

### Architecture

The network is organized as a sequence of fully connected layers: an input layer, zero or more hidden layers, and an output layer. The input and output sizes are determined automatically from the data. The hidden layer sizes are specified by the user via the `units` parameter, where `units=[]` is a valid input meaning no hidden layers.

Hidden layers use the **sigmoid** activation function as required by the instructions:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The output layer uses **softmax**, which converts raw scores into a probability distribution over classes:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Each layer includes a bias, implemented as a standard bias neuron that always outputs 1. This is folded into the matrix multiply by prepending a column of ones to the input, so instead of computing `Z = A @ W + b` separately, we compute `Z = A_with_bias @ W`. Each weight matrix therefore has shape `(n_inputs + 1, n_outputs)`, where the first row holds the bias weights. The `fit()` method returns a separate model object. Its `predict()` method returns the full probability matrix over classes, and its `weights()` method returns the list of weight matrices following this shape convention.

### Loss Function

Training minimizes the **cross entropy loss** between the predicted class probabilities and the true one-hot encoded targets:

$$E = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} t_{ik} \log(y_{ik})$$

where $t$ is the one-hot target matrix, $y$ is the softmax output, $m$ is the number of samples, and $K$ is the number of classes.

### Forward Pass

For each layer $\ell$, the network computes:

$$Z^{(\ell)} = A^{(\ell-1)}_{\text{bias}} \, W^{(\ell)}, \quad A^{(\ell)} = \sigma(Z^{(\ell)})$$

where $A^{(\ell-1)}_{\text{bias}}$ is the activation from the previous layer with a bias column of ones prepended. At the output layer, softmax is applied instead of sigmoid. All layer activations are stored during the forward pass as they are needed in backpropagation.

### Backward Pass

The gradient derivation follows Sadowski's backpropagation notes. For softmax with cross entropy, the output error simplifies to:

$$\delta^{(L)} = A^{(L)} - T$$

This is the same result derived in the notes for both logistic and softmax outputs. The weight gradient for each layer is then:

$$\frac{\partial E}{\partial W^{(\ell)}} = \frac{1}{m} \left( A^{(\ell-1)}_{\text{bias}} \right)^{\top} \delta^{(\ell)}$$

The error signal is propagated to lower layers as:

$$\delta^{(\ell)} = \left( \delta^{(\ell+1)} \left( W^{(\ell+1)} \right)^{\top} \right)_{[1:]} \odot \sigma'(A^{(\ell)})$$

where $[\,1:\,]$ denotes dropping the bias row, $\odot$ is elementwise multiplication, and $\sigma'$ is the sigmoid derivative:

$$\sigma'(A) = A \odot (1 - A)$$

### Optimization

Training uses **mini-batch gradient descent**. Mini-batch was chosen over full-batch gradient descent to scale better to larger datasets. Data is reshuffled before each epoch so batches differ across epochs.

| Parameter | Value | Reason |
|---|---|---|
| Learning rate | 0.5 | Converges reliably with Xavier initialization and sigmoid activations |
| Epochs | 10000 | Sufficient for convergence on small to medium datasets |
| Batch size | 32 | Standard mini-batch size, scales well to larger datasets |
| Random seed | 42 | Fixed for reproducibility across platforms |

Regularization is not applied in part 1. The `lambda_` parameter is accepted but ignored.

### Weight Initialization

Weights are initialized using **Xavier initialization**, which is recommended by the course notes for sigmoid activations. It keeps the variance of activations stable across layers by drawing weights uniformly from:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}},\ \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$

---

## Results on Datasets

We searched for the network with the fewest total parameters (weights) that perfectly classifies the training data on `doughnut.tab` and `squares.tab`. The search covered all single hidden layer networks up to 5 neurons and all two hidden layer networks up to 3 neurons per layer, for a total of 14 configurations. The number of weights includes bias weights across all weight matrices. Training used 5000 epochs per configuration, which was sufficient for convergence on both datasets.

### `doughnut.tab`

| Architecture | Training accuracy | Total weights |
|---|---|---|
| `[1]` | 0.681 | 7 |
| `[2]` | 0.903 | 12 |
| `[3]` | 1.000 | 17 |
| `[1, 2]` | 0.685 | 13 |
| `[2, 1]` | 0.699 | 13 |
| `[3, 2]` | 1.000 | 23 |
| ... | ... | ... |

**Minimal network:** `units=[3]` with **17 weights**.

### `squares.tab`

| Architecture | Training accuracy | Total weights |
|---|---|---|
| `[3]` | 0.738 | 17 |
| `[4]` | 1.000 | 22 |
| `[2, 1]` | 0.892 | 13 |
| `[2, 2]` | 1.000 | 18 |
| `[3, 2]` | 1.000 | 23 |
| ... | ... | ... |

**Minimal network:** `units=[2, 2]` with **18 weights**.

`squares.tab` requires a two hidden layer network to achieve perfect accuracy with fewer parameters than the best single hidden layer solution (`units=[4]` with 22 weights). This reflects the more complex decision boundary: four separate square regions rather than a single enclosed ring shape.

---

## Gradient Verification

After identifying the minimal architectures, we verified that the analytically derived gradients are correct by comparing them to numerically approximated gradients using the definition of the derivative:

$$\frac{\partial E}{\partial w} \approx \frac{E(w + \varepsilon) - E(w)}{\varepsilon}, \quad \varepsilon = 10^{-5}$$

For each weight in the network, we perturbed it by $\varepsilon$, measured the change in loss, and compared the result to the analytical gradient from backpropagation. The relative difference between the two was computed as:

$$\text{relative difference} = \frac{|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}|}{|\nabla_{\text{analytical}}| + |\nabla_{\text{numerical}}|}$$

Results are reported per weight matrix using the minimal architectures found above. XOR is also included as a sanity check on a known nonlinear problem.

| Dataset | Architecture | Weight matrix | Max relative difference |
|---|---|---|---|
| XOR | `[3]` | Layer 1 (input → hidden) | 1.61e-06 |
| XOR | `[3]` | Layer 2 (hidden → output) | 2.49e-06 |
| `doughnut.tab` | `[3]` | Layer 1 (input → hidden) | 6.40e-07 |
| `doughnut.tab` | `[3]` | Layer 2 (hidden → output) | 2.24e-06 |
| `squares.tab` | `[2, 2]` | Layer 1 (input → hidden) | 7.58e-07 |
| `squares.tab` | `[2, 2]` | Layer 2 (hidden → hidden) | 1.40e-06 |
| `squares.tab` | `[2, 2]` | Layer 3 (hidden → output) | 3.30e-06 |

The relative differences are in the range of $10^{-7}$ to $10^{-6}$, which is well within the approximation error of the one-sided numerical formula, proportional to $\varepsilon = 10^{-5}$. This confirms that the analytical backpropagation gradients are correct across all tested datasets and architectures.
