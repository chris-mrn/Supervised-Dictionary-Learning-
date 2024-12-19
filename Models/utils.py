import numpy as np


class PrimalDualSolver_l2:
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


class PrimalDualSolver_logistic:
    def __init__(self, theta, b, x_i, y_i, D, lambda_0, lambda_1, lambd, mu, tol=1e-6, max_iter=1000):
        """
        This Primal-Dual Solver aim to solve the following optimization problem:
        \min_{alpha} {f1(alpha) + f2(alpha) + \lambda_1 ||alpha||_1}
        where f1(alpha) = C(y_i (theta^T alpha - b)) with C(x) = log(1 + exp(-x))
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
        inner_product = np.dot(self.theta.T, alpha) + self.b

        # Compute the exponential term
        exp_term = np.exp(-self.y_i * inner_product)

        # Compute the gradient
        gradient = -exp_term / (1 + exp_term) * self.y_i * self.theta.T

        return gradient

    def gradient_f2(self, alpha):
        """Gradient of f2 with respect to alpha."""
        return 2 * self.lambda_0 * (self.D.T @ (self.D @ alpha - self.x_i))

    @staticmethod
    def prox_l1(v, lambd):
        """Proximal operator for the L1 norm with lambda."""
        return np.sign(v) * np.maximum(np.abs(v) - lambd, 0)

    def objective(self, alpha):
        """Compute the objective function value."""
        x = self.y_i * self.theta.T @ alpha + self.b
        f1 = np.log(1 + np.exp(-x))
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


class ProjectedGradientDescent_l2:
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


class ProjectedGradientDescent_logistic:
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
    def logistic_loss(x):
        return np.log(1 + np.exp(-x))

    @staticmethod
    def grad_logistic_loss(x):
        return - 1 / (1 + np.exp(x))

    def compute_gradient_theta(self, theta, b, y, alphas, lambda_2):
        grad_theta = np.zeros_like(theta)
        for i in range(len(y)):
            x = y[i] * (np.dot(theta, alphas[i]) + b)
            grad_theta += self.grad_logistic_loss(x) * y[i] * alphas[i]
        return grad_theta + 2 * lambda_2 * theta

    def compute_gradient_b(self, theta, b, y, alphas):
        grad_b = 0
        for i in range(len(y)):
            x = y[i] * (np.dot(theta, alphas[i]) + b)
            grad_b += self.grad_logistic_loss(x) * y[i]
        return grad_b

    @staticmethod
    def compute_gradient_D(D, x, alphas, lambda_0):
        grad_D = np.zeros_like(D)
        for i in range(len(x)):
            grad_D +=  (x[i] - np.dot(D, alphas[i])) @ alphas[i].T
        return - 2 * lambda_0 * grad_D

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
            prediction_error = self.logistic_loss(self.y[i]*(np.dot(self.theta, self.alphas[i]) + self.b))
            sparsity_penalty = self.lambda_1 * np.linalg.norm(self.alphas[i], 1)
            S += prediction_error + reconstruction_error + sparsity_penalty
        regularization = self.lambda_2 * np.linalg.norm(self.theta) ** 2
        return S + regularization

    def optimize(self):
        for iter in range(self.max_iter):
            grad_theta = self.compute_gradient_theta(self.theta, self.b, self.y, self.alphas, self.lambda_2)
            grad_D = self.compute_gradient_D(self.D, self.x, self.alphas, self.lambda_0)
            grad_b = self.compute_gradient_b(self.theta, self.b, self.y, self.alphas)

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