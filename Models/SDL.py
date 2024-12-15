import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize
from joblib import Parallel, delayed


class SupervisedDictionaryLearning(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10, lambda0=0.1, lambda1=0.1, lambda2=0.1, max_iter=100, tol=1e-4, n_jobs=-1):
        """
        Parameters:
        - n_components: int, size of the dictionary (k)
        - lambda0, lambda1, lambda2: float, regularization parameters
        - max_iter: int, maximum number of iterations
        - tol: float, convergence tolerance
        - n_jobs: int, number of parallel jobs to run for sparse coding (default is -1 for all CPUs)
        """
        self.n_components = n_components
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs


    def _initialize(self, X):
        n_features = X.shape[1]
        self.D_ = np.random.randn(n_features, self.n_components)
        self.D_ /= np.linalg.norm(self.D_, axis=0)  # Normalize columns
        self.theta_ = np.zeros(self.n_components + 1)  # For linear model (w, b)

    def _sparse_coding(self, X):
        """Solve supervised sparse coding for each sample in parallel."""
        m = X.shape[0]
        # Parallelize the computation of alpha for each sample
        alpha_neg = Parallel(n_jobs=self.n_jobs)(delayed(self._solve_alpha)(X[i], -1) for i in range(m))
        alpha_pos = Parallel(n_jobs=self.n_jobs)(delayed(self._solve_alpha)(X[i], 1) for i in range(m))
        return np.array(alpha_neg), np.array(alpha_pos)

    def _solve_alpha(self, xi, label):
        """Solve the sparse coding problem for a single sample."""
        def objective(alpha):
            reconstruction_error = np.linalg.norm(xi - self.D_ @ alpha)**2
            sparsity_penalty = self.lambda1 * np.linalg.norm(alpha, 1)
            classification_loss = np.log(1 + np.exp(-label * (self.theta_[:-1] @ alpha + self.theta_[-1])))
            return classification_loss + self.lambda0 * reconstruction_error + sparsity_penalty

        result = minimize(objective, np.zeros(self.n_components), method='L-BFGS-B')
        return result.x

    def _update_dictionary_and_params(self, X, y, alpha_neg, alpha_pos, lr_D=0.01, lr_theta=0.01):
        """Update D and theta using provided gradients."""
        m = X.shape[0]
        self.list_mu = 0*np.linspace(0, 1, m)
        grad_D = np.zeros_like(self.D_)
        grad_theta_w = np.zeros(self.n_components)
        grad_theta_b = 0

        for i in range(m):
            xi = X[i]
            yi = y[i]

            # Compute omega
            S_neg = self._compute_S(alpha_neg[i], xi, yi, -1)
            S_pos = self._compute_S(alpha_pos[i], xi, yi, +1)
            omega_neg = - self.list_mu[i] * (-1) * self._nabla_C(S_neg - S_pos) + (1 - self.list_mu[i]) * int(yi == -1)
            omega_pos = -self.list_mu[i] * (+1) * self._nabla_C(S_neg - S_pos) + (1 - self.list_mu[i]) * int(yi == +1)

            # Update gradients for D
            grad_D += omega_neg * np.outer((xi - self.D_ @ alpha_neg[i]), alpha_neg[i])
            grad_D += omega_pos * np.outer((xi - self.D_ @ alpha_pos[i]), alpha_pos[i])

            # Update gradients for theta (w and b)
            pred_neg = self.theta_[:-1] @ alpha_neg[i] + self.theta_[-1]
            pred_pos = self.theta_[:-1] @ alpha_pos[i] + self.theta_[-1]
            grad_theta_w += omega_neg * (-1) * self._nabla_C(pred_neg) * alpha_neg[i]
            grad_theta_w += omega_pos * (+1) * self._nabla_C(pred_pos) * alpha_pos[i]
            grad_theta_b += omega_neg * (-1) * self._nabla_C(pred_neg)
            grad_theta_b += omega_pos * (+1) * self._nabla_C(pred_pos)

        grad_D *= - 2 * self.lambda0
        # Gradient updates
        self.D_ -= lr_D * grad_D  # Learning rate for D
        self.D_ /= np.linalg.norm(self.D_, axis=0)  # Re-normalize columns
        self.theta_[:-1] -= lr_theta * grad_theta_w  # Learning rate for w
        self.theta_[-1] -= lr_theta * grad_theta_b  # Learning rate for b

    def _compute_S(self, alpha, xi, yi, label):
        """Compute the loss S for a given alpha."""
        reconstruction_error = np.linalg.norm(xi - self.D_ @ alpha)**2
        sparsity_penalty = self.lambda1 * np.linalg.norm(alpha, 1)
        classification_loss = np.log(1 + np.exp(-label * (self.theta_[:-1] @ alpha + self.theta_[-1])))
        return classification_loss + self.lambda0 * reconstruction_error + sparsity_penalty

    def _nabla_C(self, x):
        """Gradient of the logistic loss."""
        return -1 / (1 + np.exp(x))

    def fit(self, X, y):
        """Fit the model to the data."""
        self._initialize(X)
        for iteration in range(self.max_iter):
            alpha_neg, alpha_pos = self._sparse_coding(X)
            self._update_dictionary_and_params(X, y, alpha_neg, alpha_pos)

            # Log progress
            current_loss = np.mean([self._compute_S(alpha_neg[i], X[i], y[i], -1) +
                                    self._compute_S(alpha_pos[i], X[i], y[i], +1) for i in range(len(y))])
            print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {current_loss:.4f}")

            # Check convergence
            if iteration > 0 and np.linalg.norm(alpha_neg - alpha_pos) < self.tol:
                print("Convergence reached.")
                break

        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        alpha_list = np.array([self._solve_alpha(xi, 1) for xi in X])  # Assume yi=1 for prediction
        scores = alpha_list @ self.theta_[:-1] + self.theta_[-1]
        return np.sign(scores)

    def score(self, X, y):
        """Return the classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
