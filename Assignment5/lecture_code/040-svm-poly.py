import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.datasets import make_moons

# Generate non-linearly separable data
X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
y = 2 * y - 1  # Convert labels from {0,1} to {-1,1}

# Plot the data
plt.figure(figsize=(4, 4))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='o', label='Positive')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', marker='x', label='Negative')
plt.xlabel('x1', labelpad=10)
plt.ylabel('x2', labelpad=10)
plt.tight_layout()
plt.savefig('svm-moons-data.svg')
plt.close()

# Polynomial kernel function
def polynomial_kernel(X1, X2=None, degree=3, c=1):
    if X2 is None:
        X2 = X1
    return (np.dot(X1, X2.T) + c) ** degree

# Compute the kernel matrix
K = polynomial_kernel(X)
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(len(y)))

# For soft-margin SVM: enforce 0 <= alpha_i <= C
C_value = 1.0  # Example value of C

# Stack two constraints: -alpha <= 0 and alpha <= C
G = matrix(np.vstack((-np.eye(len(y)), np.eye(len(y)))))
h = matrix(np.hstack((np.zeros(len(y)), C_value * np.ones(len(y)))))

A = matrix(y, (1, len(y)), 'd')
b = matrix(0.0)

# Solve the dual problem
solution = solvers.qp(P, q, G, h, A, b)

# Extract alphas
alphas = np.array(solution['x']).flatten()

# Identify support vectors (now including those with alpha = C)
support_vectors = (alphas > 1e-5) & (alphas < C_value - 1e-5)
margin_vectors = (alphas > 1e-5)  # All vectors with alpha > 0

# Compute bias term using support vectors
def compute_bias():
    # Use only true support vectors (not bounded ones) to compute bias
    sv_idx = support_vectors
    if np.sum(sv_idx) > 0:
        K_sv = polynomial_kernel(X[sv_idx], X)
        return np.mean(y[sv_idx] - np.sum(alphas * y * K_sv, axis=1))
    return 0.0

b = compute_bias()

# Function to compute decision value for a point
def decision_function(x_test):
    x_test = np.atleast_2d(x_test)
    K_test = polynomial_kernel(x_test, X, degree=3, c=1)
    return np.sum(alphas * y * K_test, axis=1) + b

# Plotting
plt.figure(figsize=(6, 6*0.75))

# Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = decision_function(xy).reshape(xx.shape)

# Plot data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='o', label='Positive')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', marker='x', label='Negative')

# Plot decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linewidths=[1, 2, 1], linestyles=['--', '-', '--'])

# Highlight support vectors
plt.scatter(X[margin_vectors, 0], X[margin_vectors, 1],
            s=100, facecolors='none', edgecolors='k', linewidths=2)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x1', labelpad=10)
plt.ylabel('x2', labelpad=10)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('svm-poly.svg')
plt.close()

# Number of support vectors
print(f"Number of support vectors: {np.sum(margin_vectors)}")
print(f"Number of margin violations (alpha = C): {np.sum(alphas > C_value - 1e-5)}")