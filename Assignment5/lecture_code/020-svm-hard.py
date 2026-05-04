import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Generate synthetic data
np.random.seed(0)
num_samples = 20
X_pos = np.random.randn(num_samples, 2) + np.array([2, 2])
X_neg = np.random.randn(num_samples, 2) + np.array([-2, -2])
X = np.vstack((X_pos, X_neg))
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X_pos[:, 0], X_pos[:, 1], c='r', marker='o', label='Positive')
plt.scatter(X_neg[:, 0], X_neg[:, 1], c='b', marker='x', label='Negative')
plt.savefig('svm-data.svg')
plt.close()

# Compute the Gram matrix
K = np.dot(X, X.T)
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(2 * num_samples))
G = matrix(-np.eye(2 * num_samples))
h = matrix(np.zeros(2 * num_samples))
A = matrix(y, (1, 2 * num_samples), 'd')
b = matrix(0.0)

# Solve the dual problem
solution = solvers.qp(P, q, G, h, A, b)

# Extract alphas
alphas = np.array(solution['x']).flatten()

# Reconstruct w
w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)

# Identify support vectors
support_vectors = alphas > 1e-5

# Compute b using any support vector
b_value = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))

# Plotting
plt.figure(figsize=(4, 4))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

# Highlight support vectors
plt.scatter(X[support_vectors, 0], X[support_vectors, 1],
            s=100, facecolors='none', edgecolors='k', linewidths=2)

# Plot decision boundary
xx = np.linspace(-6, 6, 50)
yy = -(w[0] * xx + b_value) / w[1]
plt.plot(xx, yy, 'k-')

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
plt.savefig('svm-hard.svg')
plt.close()

# Number of support vectors
print(f"Number of support vectors: {np.sum(support_vectors)}")