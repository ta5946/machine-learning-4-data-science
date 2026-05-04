import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Generate concentric circles data with overlap
def generate_circles(n_samples=100, noise=0.15):
    # Generate points for the inner circles (class 1)
    n_inner = n_samples // 2
    theta = np.random.uniform(0, 2*np.pi, n_inner)
    r = np.random.uniform(0, 1.2, n_inner)  # Reduced radius to 1.2
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    
    # Generate points for the outer circle (class -1)
    n_outer = n_samples
    theta = np.random.uniform(0, 2*np.pi, n_outer)
    r = np.random.uniform(1.3, 2.5, n_outer)  # Increased inner radius to 1.3
    x2 = r * np.cos(theta)
    y2 = r * np.sin(theta)
    
    # Add noise
    x1 += np.random.normal(0, noise, n_inner)
    y1 += np.random.normal(0, noise, n_inner)
    x2 += np.random.normal(0, noise, n_outer)
    y2 += np.random.normal(0, noise, n_outer)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.ones(n_inner), -np.ones(n_outer)])
    return X, y

# Generate data
X, y = generate_circles(n_samples=200, noise=0.2)

# Plot the data
plt.figure(figsize=(4, 4))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='o', label='Positive')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', marker='x', label='Negative')
plt.xlabel('x1', labelpad=10)
plt.ylabel('x2', labelpad=10)
plt.tight_layout()
plt.savefig('svm-circles-data.svg')
plt.close()

# RBF kernel function
def rbf_kernel(X1, X2=None, gamma=1.0):
    if X2 is None:
        X2 = X1
    # Compute pairwise squared distances
    X1_sq = np.sum(X1**2, axis=1)
    X2_sq = np.sum(X2**2, axis=1)
    distances = X1_sq[:, np.newaxis] + X2_sq[np.newaxis, :] - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * distances)

# Compute the kernel matrix
gamma = 1.0  # RBF kernel parameter
K = rbf_kernel(X, gamma=gamma)
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
        K_sv = rbf_kernel(X[sv_idx], X, gamma=gamma)
        return np.mean(y[sv_idx] - np.sum(alphas * y * K_sv, axis=1))
    return 0.0

b = compute_bias()

# Function to compute decision value for a point
def decision_function(x_test):
    x_test = np.atleast_2d(x_test)
    K_test = rbf_kernel(x_test, X, gamma=gamma)
    return np.sum(alphas * y * K_test, axis=1) + b

# Plotting
plt.figure(figsize=(6, 6*0.75))

# Create a mesh grid for plotting - limit to data region plus small margin
max_radius = np.max(np.sqrt(np.sum(X**2, axis=1))) * 1.1  # Tighter margin
xx, yy = np.meshgrid(np.linspace(-max_radius, max_radius, 200),
                     np.linspace(-max_radius, max_radius, 200))
xy = np.vstack([xx.ravel(), yy.ravel()]).T

# Only compute decision function for points within the circular region
distances = np.sqrt(np.sum(xy**2, axis=1))
mask = distances <= max_radius
Z = np.zeros(len(xy))
Z[mask] = decision_function(xy[mask])
Z = Z.reshape(xx.shape)

# Set figure background
plt.gca().set_facecolor('white')
plt.gca().patch.set_alpha(1.0)

# Plot data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='o', label='Positive')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', marker='x', label='Negative')

# Mask values outside the region of interest and set them to NaN
r = np.sqrt(xx**2 + yy**2)
Z = np.where(r <= max_radius, Z, np.nan)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)  # Decision boundary
plt.contour(xx, yy, Z, levels=[-1, 1], colors='k', linewidths=1, linestyles='--')  # Margins

# Highlight support vectors
plt.scatter(X[margin_vectors, 0], X[margin_vectors, 1],
            s=100, facecolors='none', edgecolors='k', linewidths=2)

# Set equal aspect ratio and limits
plt.gca().set_aspect('equal')
plt.xlim(-max_radius, max_radius)
plt.ylim(-max_radius, max_radius)
plt.xlabel('x1', labelpad=10)
plt.ylabel('x2', labelpad=10)
plt.legend()
plt.tight_layout()
plt.savefig('svm-rbf.svg', bbox_inches='tight')
plt.close()

# Number of support vectors
print(f"Number of support vectors: {np.sum(margin_vectors)}")
print(f"Number of margin violations (alpha = C): {np.sum(alphas > C_value - 1e-5)}")