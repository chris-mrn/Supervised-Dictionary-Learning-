import numpy as np
import scipy

# In the case of binary classification


class PrimalDualSolver:
    def __init__(self, theta, b, x_i, y_i, D, lambda_0, lambda_1, lambd, mu, tol=1e-6, max_iter=1000):
        """
        This Primal-Dual Solver aim to solve the following optimization problem:
        \min_{alpha} {f1(alpha) + f2(alpha) + \lambda_1 ||alpha||_1}
        where f1(alpha) = ||y_i - theta^T alpha - b||^2
        and f2(alpha) = lambda_0 ||x_i - D alpha||^2 + lambda_1 ||alpha||_1

        Initialize the solver with problem parameters.

        Parameters:
            theta: Parameter vector for f1 (np.ndarray).
            b: Scalar bias term for f1 (float).
            x_i: Input vector for f2 (np.ndarray).
            y_i: Target value for f1 (float).
            D: Dictionary matrix for f2 (np.ndarray).
            lambda_0: Regularization parameter for f2 (float).
            lambda_1: Regularization parameter for f3 (float).
            lambd: Step size (float).
            mu: Scaling factor (float).
            tol: Convergence tolerance (float).
            max_iter: Maximum number of iterations (int).
        """
        self.theta = theta
        self.b = b
        self.x_i = x_i
        self.y_i = y_i
        self.D = D
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambd = lambd
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter

    def gradient_f1(self, alpha):
        """Gradient of f1 with respect to alpha."""
        return 2 * self.lambda_1 * (self.theta.T @ alpha + self.b - self.y_i) * self.theta

    def gradient_f2(self, alpha):
        """Gradient of f2 with respect to alpha."""
        return 2 * self.lambda_0 * (self.D.T @ (self.D @ alpha - self.x_i))

    @staticmethod
    def prox_l1(v, lambd):
        """Proximal operator for the L1 norm with lambda."""
        return np.sign(v) * np.maximum(np.abs(v) - lambd, 0)

    def objective(self, alpha):
        """Compute the objective function value."""
        f1 = np.linalg.norm(self.y_i - (self.theta.T @ alpha + self.b)) ** 2
        f2 = self.lambda_0 * np.linalg.norm(self.x_i - self.D @ alpha) ** 2
        f3 = self.lambda_1 * np.linalg.norm(alpha, 1)
        return f1 + f2 + f3

    def solve(self, x0):
        """
        Solve the optimization problem using the primal-dual splitting algorithm.

        Parameters:
            x0: Initial value of alpha (np.ndarray).

        Returns:
            alpha: Optimized alpha (np.ndarray).
            history: List of objective function values at each iteration (list).
        """
        alpha = x0
        history = []

        for k in range(self.max_iter):
            # Compute gradients
            grad_f1 = self.gradient_f1(alpha)
            grad_f2 = self.gradient_f2(alpha)
            grad_f = grad_f1 + grad_f2

            # Update rule
            alpha_new = self.mu * self.prox_l1(alpha - self.lambd * grad_f, self.lambd * self.lambda_1)

            # Convergence check
            if np.linalg.norm(alpha_new - alpha) < self.tol:
                break

            # Update alpha and track function value
            alpha = alpha_new
            history.append(self.objective(alpha))

        return alpha, history


class ProjectedGradientDescent:
    def __init__(self, D_init, theta_init, b, x, y, alphas, lambda_0, lambda_1, lambda_2, lr=0.01, max_iter=1000, tol=1e-6):
        self.D = D_init.copy()
        self.theta = theta_init.copy()
        self.b = b
        self.x = x
        self.y = y
        self.alphas = alphas
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    @staticmethod
    def compute_gradient_theta(theta, b, y, alphas, lambda_2):
        grad_theta = 2 * lambda_2 * theta
        for i in range(len(y)):
            grad_theta += 2 * alphas[i] * (np.dot(theta, alphas[i]) + b - y[i])
        return grad_theta

    @staticmethod
    def compute_gradient_D(D, x, alphas, lambda_0):
        grad_D = np.zeros_like(D)
        for i in range(len(x)):
            grad_D -= 2 * lambda_0 * np.outer((x[i] - np.dot(D, alphas[i])), alphas[i])
        return grad_D

    @staticmethod
    def project_D(D):
        norm_D = np.linalg.norm(D)
        if norm_D > 1:
            D /= norm_D
        return D

    def objective(self):
        S = 0
        for i in range(len(self.y)):
            reconstruction_error = self.lambda_0 * np.linalg.norm(self.x[i] - np.dot(self.D, self.alphas[i])) ** 2
            prediction_error = np.linalg.norm(self.y[i] - (np.dot(self.theta, self.alphas[i]) + self.b)) ** 2
            sparsity_penalty = self.lambda_1 * np.linalg.norm(self.alphas[i], 1)
            S += prediction_error + reconstruction_error + sparsity_penalty
        regularization = self.lambda_2 * np.linalg.norm(self.theta) ** 2
        return S + regularization

    def optimize(self):
        for iter in range(self.max_iter):
            grad_theta = self.compute_gradient_theta(self.theta, self.b, self.y, self.alphas, self.lambda_2)
            grad_D = self.compute_gradient_D(self.D, self.x, self.alphas, self.lambda_0)
            grad_b = 2 * self.b * len(self.y)

            # Gradient updates
            self.theta -= self.lr * grad_theta
            self.b -= self.lr * grad_b
            self.D -= self.lr * grad_D

            # Project D onto the feasible set
            self.D = self.project_D(self.D)

            # Compute objective and check convergence
            obj_value = self.objective()
            self.history.append(obj_value)

            if iter > 0 and abs(self.history[-1] - self.history[-2]) < self.tol:
                break

        return self.D, self.theta, self.b, self.history



def function_S(alpha, x, D, w, b, lambda_0, lambda_1, pos=True, method="linear"):
    if method == "linear":
        if pos:
            logistic_loss = np.log(1 + np.exp(-(w.T @ alpha + b)))
        else:
            logistic_loss = np.log(1 + np.exp(w.T @ alpha + b))
    else:
        print("Method not implemented")
        return np.nan

    reg_1 = lambda_0 * np.linalg.norm(x - D @ alpha, ord=2)**2
    reg_2 = lambda_1 * np.linalg.norm(alpha, ord=1)

    return logistic_loss + reg_1 + reg_2


def supervised_sparse_coding(alpha, signals, D, w, b):
    """Supervised sparse coding step for 2 classes"""

    n = signals.shape[0]  #nbr of signals
    size_dict = D.shape[1]  #dictionary size
    lambda_0 = 1
    lambda_1 = 0.15

    alpha_opt_pos = np.zeros((n, size_dict))
    alpha_opt_neg = np.zeros((n, size_dict))

    alpha_pos = alpha
    alpha_neg = alpha

    for i in range(n):
        print(i)
        x = signals[i, :]
        # optimize on alpha S_neg(-1 f(alpha, x_i, D, theta)
        alpha_temp_neg = scipy.optimize.minimize(function_S, alpha_neg, args= (x, D, w, b, lambda_0, lambda_1, False), tol=1e-6)
        # optimize on alpha S_neg(1 f(alpha, x_i, D, theta)
        alpha_temp_pos = scipy.optimize.minimize(function_S, alpha_pos, args= (x, D, w, b, lambda_0, lambda_1, True), tol=1e-6)

        alpha_opt_pos[i, :] = alpha_temp_pos.x
        alpha_opt_neg[i, :] = alpha_temp_neg.x
        alpha_pos = alpha_temp_pos.x
        alpha_neg = alpha_temp_neg.x

    return alpha_opt_neg, alpha_opt_pos


def compute_gradients(D, w, b, signals, alpha, lambda_0, mu):
    """Compute gradients of E with respect to D, w, and b."""
    # Initialize gradients
    grad_D = np.zeros_like(D)
    grad_w = np.zeros_like(w)
    grad_b = 0.0

    # Iterate over data samples
    for i in range(len(signals)):
        for z in [-1, +1]:
            # Compute residual and coefficients
            alpha_i_z = alpha[:, i, z]
            residual = signals[i] - D @ alpha_i_z

            # Compute omega_i_z
            grad_C = (w.T @ alpha_i_z + b)  # Assume a placeholder gradient of C
            omega_i_z = -mu * z * grad_C

            # Gradients
            grad_D -= 2 * lambda_0 * omega_i_z * np.outer(residual, alpha_i_z)
            grad_w += omega_i_z * z * grad_C * alpha_i_z
            grad_b += omega_i_z * z * grad_C

    return grad_D, grad_w, grad_b


def project_D(D):
    """Project dictionary D to satisfy column constraints (e.g., unit-norm)."""
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-8)


def projected_gradient_descent(D, w, b, signals, alpha, lambda_0, mu, grad_steps=100, step_size=0.01, tol=1e-6):
    """
    Projected Gradient Descent for optimizing E(D, theta) with dictionary D and linear parameters w, b.

    Parameters:
        D (ndarray): Initial dictionary, shape (d, k)
        w (ndarray): Initial linear weights, shape (k,)
        b (float): Initial bias term
        X (ndarray): Input data matrix, shape (m, d)
        alpha (ndarray): Sparse codes, shape (k, m, 2) for each z={-1,+1}
        lambda_0 (float): Regularization parameter
        mu (float): Classification tradeoff parameter
        grad_steps (int): Number of gradient descent iterations
        step_size (float): Step size for updates
        tol (float): Tolerance for convergence

    Returns:
        D, w, b: Updated parameters
    """

    m, d = signals.shape  # m samples, d dimensions
    k = D.shape[1]  # k dictionary atoms

    # Gradient Descent Loop
    for step in range(grad_steps):
        # Compute gradients
        grad_D, grad_w, grad_b = compute_gradients(D, w, b, signals, alpha, lambda_0, mu)

        # Update parameters
        D -= step_size * grad_D
        w -= step_size * grad_w
        b -= step_size * grad_b

        # Project dictionary D
        D = project_D(D)

        # Convergence Check (based on gradient norms)
        if np.linalg.norm(grad_D) < tol and np.linalg.norm(grad_w) < tol and abs(grad_b) < tol:
            print(f"Converged at step {step + 1}")
            break

    return D, w, b
