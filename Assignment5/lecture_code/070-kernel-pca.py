import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

# Generate Swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, noise=0.05, random_state=42)

# Define the RBF (Gaussian) kernel function
def rbf_kernel(X, gamma=0.02):
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

# Compute the kernel matrix
K = rbf_kernel(X)

# Center the kernel matrix
n = K.shape[0]
one_n = np.ones((n, n)) / n
K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

# Solve the eigenvalue problem, sort eigenvalues and eigenvectors in descending order
eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Normalize eigenvectors (important because in feature space, eigenvectors need normalization)
alphas = eigenvectors / np.sqrt(eigenvalues[np.newaxis, :])

# Project the data onto the first two principal components
X_kpca = K_centered @ alphas[:, :2]

# Visualization
fig = plt.figure(figsize=(18, 5))

# Original 3D Swiss roll
ax = fig.add_subplot(1, 3, 1, projection='3d')
scatter1 = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis')
ax.set_title('Original Swiss Roll')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Standard PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
ax = fig.add_subplot(1, 3, 2)
scatter2 = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax.set_title('Standard PCA projection')
ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd principal component')

# Kernel PCA projection
ax = fig.add_subplot(1, 3, 3)
scatter3 = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap='viridis')
ax.set_title('PCA projection with Gaussian kernel')
ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd principal component')

plt.tight_layout()
plt.savefig('kernel-pca.svg')
plt.close('all')  # Close all figures after saving
