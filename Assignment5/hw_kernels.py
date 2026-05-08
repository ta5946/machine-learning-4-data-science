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
        # kernel: callable kernel object (e.g. RBF, Polynomial)
        # lambda_: regularization strength; larger = smoother fit
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
        # kernel: callable kernel object (e.g. RBF, Polynomial)
        # lambda_: regularization strength; sets box constraint C = 1/lambda_
        # epsilon: width of the ε-insensitive tube; larger = sparser solution
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Part 1: visualization on sine.csv ---

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
    ax.plot(X_plot, y_pred, color='darkred', linewidth=2, label='Fit')
    ax.set_title(f'KRR with RBF (σ={sigma}, λ={lambda_})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- KRR with Polynomial ---
    ax = axes[0, 1]
    y_pred = KernelizedRidgeRegression(Polynomial(M), lambda_poly).fit(X, y).predict(X_plot)
    ax.scatter(X, y, c='steelblue', s=15, alpha=0.6, label='Data')
    ax.plot(X_plot, y_pred, color='darkred', linewidth=2, label='Fit')
    ax.set_title(f'KRR with Polynomial (M={M}, λ={lambda_poly})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- SVR with RBF ---
    ax = axes[1, 0]
    model = SVR(RBF(sigma), lambda_, epsilon).fit(X, y)
    y_pred = model.predict(X_plot)
    residuals = np.abs(y - model.predict(X))
    sv_mask = residuals >= epsilon - 1e-3

    ax.fill_between(X_plot.flatten(), y_pred - epsilon, y_pred + epsilon,
                    alpha=0.15, color='darkred', label='ε-tube')
    ax.scatter(X[~sv_mask], y[~sv_mask], c='steelblue', s=15, alpha=0.6, label='Data')
    ax.scatter(X[sv_mask], y[sv_mask], c='darkblue', s=15, label=f'SVs ({sv_mask.sum()})')
    ax.plot(X_plot, y_pred, color='darkred', linewidth=2, label='Fit')
    ax.set_title(f'SVR with RBF (σ={sigma}, λ={lambda_}, ε={epsilon})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- SVR with Polynomial ---
    ax = axes[1, 1]
    model = SVR(Polynomial(M), lambda_poly, epsilon).fit(X, y)
    y_pred = model.predict(X_plot)
    residuals = np.abs(y - model.predict(X))
    sv_mask = residuals >= epsilon - 1e-3

    ax.fill_between(X_plot.flatten(), y_pred - epsilon, y_pred + epsilon,
                    alpha=0.15, color='darkred', label='ε-tube')
    ax.scatter(X[~sv_mask], y[~sv_mask], c='steelblue', s=15, alpha=0.6, label='Data')
    ax.scatter(X[sv_mask], y[sv_mask], c='darkblue', s=15, label=f'SVs ({sv_mask.sum()})')
    ax.plot(X_plot, y_pred, color='darkred', linewidth=2, label='Fit')
    ax.set_title(f'SVR with Polynomial (M={M}, λ={lambda_poly}, ε={epsilon})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kernel_methods_comparison.png', dpi=150)
    plt.show()
    print("Visualization saved to 'kernel_methods_comparison.png'")


    # --- Part 2: evaluation on housing2r.csv ---

    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    housing = pd.read_csv("housing2r.csv")
    X_h = housing.drop(columns='y').values
    y_h = housing['y'].values

    # Standardize features; distances and polynomial values depend on scale
    scaler = StandardScaler()
    X_h = scaler.fit_transform(X_h)

    M_values = list(range(1, 11))
    sigma_values = [0.1, 0.3, 0.5, 1, 2, 5, 10]
    lambda_grid = [0.001, 0.01, 0.1, 1, 10, 100]
    epsilon_h = 2.0  # ~20% of std(y); keeps SVR sparse while fitting well
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_mse(fitter_cls, kernel, lambda_, epsilon=None):
        # 5-fold CV MSE; epsilon only used for SVR
        mses = []
        for train_idx, test_idx in kf.split(X_h):
            if epsilon is not None:
                m = fitter_cls(kernel, lambda_, epsilon).fit(X_h[train_idx], y_h[train_idx])
            else:
                m = fitter_cls(kernel, lambda_).fit(X_h[train_idx], y_h[train_idx])
            mses.append(np.mean((y_h[test_idx] - m.predict(X_h[test_idx])) ** 2))
        return np.mean(mses)

    def best_lambda_mse(fitter_cls, kernel, epsilon=None):
        # Find λ with lowest 5-fold CV MSE from lambda_grid, return (best_mse, best_lambda)
        scores = [(cv_mse(fitter_cls, kernel, lam, epsilon), lam) for lam in lambda_grid]
        return min(scores, key=lambda pair: pair[0])

    def count_svs(kernel, lambda_, epsilon):
        # Train on full data; count points with α or α* above threshold
        model = SVR(kernel, lambda_, epsilon).fit(X_h, y_h)
        alphas = model.get_alpha()
        return int(((alphas[:, 0] > 1e-4) | (alphas[:, 1] > 1e-4)).sum())

    def run_evaluation(fitter_cls, kernel_fn, param_values, param_name, is_svr):
        mse_fixed, mse_cv, best_lams, sv_counts = [], [], [], []
        for i, p in enumerate(param_values):
            print(f"  {param_name}={p} ({i+1}/{len(param_values)})")
            kernel = kernel_fn(p)
            eps = epsilon_h if is_svr else None
            mse_fixed.append(cv_mse(fitter_cls, kernel, lambda_=1, epsilon=eps))
            best_mse, best_lam = best_lambda_mse(fitter_cls, kernel, epsilon=eps)
            mse_cv.append(best_mse)
            best_lams.append(best_lam)
            if is_svr:
                sv_counts.append(count_svs(kernel, best_lam, epsilon_h))
        return mse_fixed, mse_cv, best_lams, sv_counts

    print("Evaluating KRR (Polynomial)...")
    krr_poly_fixed, krr_poly_cv, krr_poly_lams, _ = run_evaluation(
        KernelizedRidgeRegression, Polynomial, M_values, "M", is_svr=False)

    print("Evaluating KRR (RBF)...")
    krr_rbf_fixed, krr_rbf_cv, krr_rbf_lams, _ = run_evaluation(
        KernelizedRidgeRegression, RBF, sigma_values, "sigma", is_svr=False)

    print("Evaluating SVR (Polynomial)...")
    svr_poly_fixed, svr_poly_cv, svr_poly_lams, svr_poly_svs = run_evaluation(
        SVR, Polynomial, M_values, "M", is_svr=True)

    print("Evaluating SVR (RBF)...")
    svr_rbf_fixed, svr_rbf_cv, svr_rbf_lams, svr_rbf_svs = run_evaluation(
        SVR, RBF, sigma_values, "sigma", is_svr=True)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def plot_mse(ax, x_vals, mse_fixed, mse_cv, xlabel, title,
                 sv_counts=None, x_log=False):
        ax.plot(x_vals, mse_fixed, 'o-', color='steelblue', linewidth=2, label='λ=1 (fixed)')
        ax.plot(x_vals, mse_cv, 's--', color='darkred', linewidth=2, label='λ from CV')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if x_log:
            ax.set_xscale('log')

        if sv_counts is not None:
            ax2 = ax.twinx()
            ax2.plot(x_vals, sv_counts, '^:', color='gray', linewidth=1.5, alpha=0.7)
            ax2.set_ylabel('# Support Vectors', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            lines1, _ = ax.get_legend_handles_labels()
            lines2, _ = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, ['λ=1 (fixed)', 'λ from CV', '# SVs'],
                      loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=8)

    plot_mse(axes[0, 0], M_values, krr_poly_fixed, krr_poly_cv,
             xlabel='M (degree)', title='KRR with Polynomial')

    plot_mse(axes[0, 1], sigma_values, krr_rbf_fixed, krr_rbf_cv,
             xlabel='σ', title='KRR with RBF', x_log=True)

    plot_mse(axes[1, 0], M_values, svr_poly_fixed, svr_poly_cv,
             xlabel='M (degree)', title=f'SVR with Polynomial (ε={epsilon_h})',
             sv_counts=svr_poly_svs)

    plot_mse(axes[1, 1], sigma_values, svr_rbf_fixed, svr_rbf_cv,
             xlabel='σ', title=f'SVR with RBF (ε={epsilon_h})',
             sv_counts=svr_rbf_svs, x_log=True)

    plt.tight_layout()
    plt.savefig('housing_evaluation.png', dpi=150)
    plt.show()
    print("Visualization saved to 'housing_evaluation.png'")
