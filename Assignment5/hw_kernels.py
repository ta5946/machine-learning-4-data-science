import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


# --- Kernel functions ---

class Polynomial:
    # κ(x, x') = (1 + x·x')^M

    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        # Handles 1D+1D → scalar, 1D+2D → 1D, 2D+2D → 2D
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        result = (1 + np.dot(A, B.T)) ** self.M
        if A.shape[0] == 1 and B.shape[0] == 1:
            return result[0, 0]
        elif A.shape[0] == 1:
            return result[0]
        elif B.shape[0] == 1:
            return result[:, 0]
        return result


class RBF:
    # κ(x, x') = exp(-||x - x'||² / 2σ²)

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        # Handles 1D+1D → scalar, 1D+2D → 1D, 2D+2D → 2D
        A_was_1d = A.ndim == 1
        B_was_1d = B.ndim == 1
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        # ||a-b||² = a·a - 2a·b + b·b, vectorized over all pairs
        sq_dist = (np.sum(A ** 2, axis=1, keepdims=True)
                   - 2 * np.dot(A, B.T)
                   + np.sum(B ** 2, axis=1, keepdims=True).T)
        result = np.exp(-sq_dist / (2 * self.sigma ** 2))
        if A_was_1d and B_was_1d:
            return result[0, 0]
        elif A_was_1d:
            return result[0]
        elif B_was_1d:
            return result[:, 0]
        return result


# --- Kernelized ridge regression ---

class KernelizedRidgeRegressionModel:

    def __init__(self, kernel, X_train, alpha):
        self.kernel = kernel
        self.X_train = X_train
        self.alpha = alpha

    def predict(self, X):
        return np.dot(self.kernel(X, self.X_train), self.alpha)


class KernelizedRidgeRegression:

    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        n = len(y)
        K = self.kernel(X, X)
        # Solve (K + λI)α = y instead of inverting directly
        alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)
        return KernelizedRidgeRegressionModel(self.kernel, X, alpha)


# --- Support vector regression ---

class SVRModel:

    def __init__(self, kernel, X_train, alpha, alpha_star, b):
        self.kernel = kernel
        self.X_train = X_train
        self.alpha = alpha       # α
        self.alpha_star = alpha_star  # α*
        self.b = b

    def predict(self, X):
        return np.dot(self.kernel(X, self.X_train), self.alpha - self.alpha_star) + self.b

    def get_alpha(self):
        # Returns shape (n, 2): each row is [αᵢ, αᵢ*]
        return np.column_stack([self.alpha, self.alpha_star])

    def get_b(self):
        return self.b


class SVR:

    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self, X, y):
        n = len(y)
        K = self.kernel(X, X)
        C = 1.0 / self.lambda_  # C = 1/λ is the standard SVR box constraint

        # Decision variables ordered as [α₁, α₁*, α₂, α₂*, ...]
        # P (2n×2n): Kronecker structure encodes K[i,j] * [[1,-1],[-1,1]] per block
        P = np.kron(K, np.array([[1, -1], [-1, 1]])) + 1e-8 * np.eye(2 * n)

        # q (2n): linear costs from the ε-insensitive loss and targets
        q = np.zeros(2 * n)
        q[0::2] = self.epsilon - y  # α terms
        q[1::2] = self.epsilon + y  # α* terms

        # G, h: box constraints 0 ≤ α, α* ≤ C
        G = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
        h = np.hstack([np.zeros(2 * n), C * np.ones(2 * n)])

        # A, b_eq: equality constraint Σ(αᵢ - αᵢ*) = 0
        A = np.zeros((1, 2 * n))
        A[0, 0::2] = 1
        A[0, 1::2] = -1
        b_eq = np.array([0.0])

        sol = solvers.qp(matrix(P.astype(float)), matrix(q.astype(float)),
                         matrix(G.astype(float)), matrix(h.astype(float)),
                         matrix(A.astype(float)), matrix(b_eq.astype(float)))

        x = np.array(sol['x']).flatten()
        alpha = x[0::2]
        alpha_star = x[1::2]

        # Bias from margin support vectors (0 < α < C or 0 < α* < C)
        # Points strictly on the margin satisfy f(x) = y ∓ ε exactly
        diff = alpha - alpha_star
        tol = 1e-4
        b_values = []
        for i in range(n):
            f_i = np.dot(K[i], diff)
            if tol < alpha[i] < C - tol:
                b_values.append(y[i] - self.epsilon - f_i)
            if tol < alpha_star[i] < C - tol:
                b_values.append(y[i] + self.epsilon - f_i)
        b = np.mean(b_values) if b_values else -float(sol['y'][0])

        return SVRModel(self.kernel, X, alpha, alpha_star, b)


# --- Visualization on sine.csv ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv("sine.csv")
    X = data['x'].values.reshape(-1, 1)
    y = data['y'].values

    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

    # Parameters from suggested ranges
    sigma = 1.0       # σ ∈ {0.5, 1.0, 2.0}
    M = 5             # M ∈ {3, 5, 7}
    lambda_ = 0.01    # λ ∈ {0.001, 0.01, 0.1}
    lambda_poly = 1.0 # Polynomial is ill-conditioned on x∈[0,20]; needs heavier regularization
    epsilon = 0.3     # ε ∈ {0.1, 0.3, 0.5}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- KRR with RBF ---
    ax = axes[0, 0]
    y_pred = KernelizedRidgeRegression(RBF(sigma), lambda_).fit(X, y).predict(X_plot)
    ax.scatter(X, y, c='steelblue', s=15, alpha=0.6, label='Data')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label='Fit')
    ax.set_title(f'KRR with RBF (σ={sigma}, λ={lambda_})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- KRR with Polynomial ---
    ax = axes[0, 1]
    y_pred = KernelizedRidgeRegression(Polynomial(M), lambda_poly).fit(X, y).predict(X_plot)
    ax.scatter(X, y, c='steelblue', s=15, alpha=0.6, label='Data')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label='Fit')
    ax.set_title(f'KRR with Polynomial (M={M}, λ={lambda_poly})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- SVR with RBF ---
    ax = axes[1, 0]
    model = SVR(RBF(sigma), lambda_, epsilon).fit(X, y)
    y_pred = model.predict(X_plot)
    residuals = np.abs(y - model.predict(X))
    sv_mask = residuals >= epsilon - 1e-3

    ax.fill_between(X_plot.flatten(), y_pred - epsilon, y_pred + epsilon,
                    alpha=0.15, color='red', label='ε-tube')
    ax.scatter(X[~sv_mask], y[~sv_mask], c='steelblue', s=15, alpha=0.6, label='Data')
    ax.scatter(X[sv_mask], y[sv_mask], c='darkblue', s=15, label=f'SVs ({sv_mask.sum()})')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label='Fit')
    ax.set_title(f'SVR with RBF (σ={sigma}, λ={lambda_}, ε={epsilon})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- SVR with Polynomial ---
    ax = axes[1, 1]
    model = SVR(Polynomial(M), lambda_poly, epsilon).fit(X, y)
    y_pred = model.predict(X_plot)
    residuals = np.abs(y - model.predict(X))
    sv_mask = residuals >= epsilon - 1e-3

    ax.fill_between(X_plot.flatten(), y_pred - epsilon, y_pred + epsilon,
                    alpha=0.15, color='red', label='ε-tube')
    ax.scatter(X[~sv_mask], y[~sv_mask], c='steelblue', s=15, alpha=0.6, label='Data')
    ax.scatter(X[sv_mask], y[sv_mask], c='darkblue', s=15, label=f'SVs ({sv_mask.sum()})')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label='Fit')
    ax.set_title(f'SVR with Polynomial (M={M}, λ={lambda_poly}, ε={epsilon})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kernel_methods_comparison.png', dpi=150)
    plt.show()
    print("Visualization saved to 'kernel_methods_comparison.png'")
