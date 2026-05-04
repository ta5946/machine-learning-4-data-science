import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Generate simple nonlinear data
np.random.seed(0)
X = np.linspace(-3, 3, 30)[:, None]
y = np.sin(X).ravel() + 0.3 * np.random.randn(30)

def rbf_kernel(X1, X2, gamma=0.01):
    # Gaussian (RBF) kernel
    X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

def polynomial_kernel(X1, X2, degree=3, coef0=1):
    # Polynomial kernel
    return (np.dot(X1, X2.T) + coef0) ** degree

# kernel, label, filename = polynomial_kernel, 'Polynomial Kernel', 'kernel-rr-poly.svg'
kernel, label, filename = rbf_kernel, 'Gaussian (RBF) Kernel', 'kernel-rr-rbf.svg'

# Compute the Gram matrix
K = kernel(X, X)

# Solve for alpha using kernel ridge regression
lmbda = 0.1  # regularization strength
n = K.shape[0]
alpha = np.linalg.solve(K + lmbda * np.eye(n), y)

# Predict on new data
X_test = np.linspace(-4, 4, 200)[:, None]
K_test = kernel(X_test, X)
y_pred = np.dot(K_test, alpha)

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(X, y, color='red', label='Training data')
plt.plot(X_test, y_pred, color='blue', label=label)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig(filename)
