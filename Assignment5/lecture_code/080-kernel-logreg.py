import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from sklearn.datasets import make_moons

# Generate two moons dataset
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# Define RBF (Gaussian) kernel
def rbf_kernel(X1, X2, gamma=0.5):
    dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    return np.exp(-gamma * dists)

# Prepare the kernel matrix
gamma = 0.5
K = rbf_kernel(X, X, gamma)

# Define logistic loss function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(alpha):
    pred = sigmoid(K @ alpha)
    # Cross-entropy loss
    return -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))

# Train the kernelized logistic regression
result = minimize(logistic_loss, np.zeros(X.shape[0]), method='L-BFGS-B')
alpha = result.x

# Plot decision boundary
# Create a grid of points
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1, 2, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Compute kernel between training points and grid points
K_test = rbf_kernel(X, grid_points, gamma=gamma)

# Predict probabilities
probs = sigmoid(K_test.T @ alpha)
probs = probs.reshape(xx.shape)

# Plot
plt.figure(figsize=(6, 6 * 0.75))

# Plot probability contours
contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contour = plt.contour(xx, yy, probs, levels=contour_levels, colors='black', alpha=0.5)
plt.clabel(contour, inline=True, fontsize=8)

# Fill the decision regions
plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])

# Plot the data points
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1', edgecolor='black')
plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0', edgecolor='black')

# plt.title("Kernelized Logistic Regression with RBF Kernel")
plt.xlabel("x1")
plt.ylabel("x2")
# plt.legend()
plt.show()
plt.savefig("kernel-logreg.png")
plt.close('all')
