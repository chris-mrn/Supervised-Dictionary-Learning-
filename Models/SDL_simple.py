import numpy as np


class SDL:
    """
    Supervised Dictionary Learning (SDL) Class.

    Combines sparse coding with supervised learning by jointly learning:
    - A dictionary `D` for sparse representation.
    - A linear model `(theta, b)` for predicting labels.
    """

    def __init__(self, n_iter=1000,
                 lamnda0=0.01,
                 lambda1=0.01,
                 lambda2=0.01,
                 lr_D=0.01,
                 lr_theta=0.01,
                 lr_alpha=0.01):
        self.n_iter = n_iter
        self.lamnda0 = lamnda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr_D = lr_D
        self.lr_theta = lr_theta
        self.lr_alpha = lr_alpha

    def objective(self, X, y, D, theta, alpha):
        """Computes the objective function value."""
        objective = 0
        for i in range(X.shape[0]):
            xi, yi, ai = X[i], y[i], alpha[i]
            loss_dict = np.linalg.norm(xi - D @ ai)**2
            loss_class = np.linalg.norm(yi - theta @ ai)**2
            sparse_penalty = self.lambda1 * np.linalg.norm(ai, 1)
            objective += loss_dict + loss_class + sparse_penalty
        return objective

    def solve_alpha(self, X, y, D, theta):
        """
        Optimizes sparse codes `alpha` for fixed `D` and `theta`.
        We can have here a explicit expression of our gradient regarding
        to alpha, so we can use a gradient descent to optimize it.
        """

        return np.zeros((X.shape[0], D.shape[1]))

    def grad_E_theta(y, D, theta, alpha, b, lambda_2):
        """
        Compute the gradient of E with respect to theta.
        """
        grad = 2 * lambda_2 * theta  # Regularization term
        for i in range(len(y)):
            error = np.dot(theta.T, alpha[:, i]) + b - y[i]
            grad += 2 * alpha[:, i].reshape(-1, 1) * error  # Data fidelity term
        return grad

    def grad_E_D(x, D, alpha, lambda_0):
        """
        Compute the gradient of E with respect to D.
        """
        grad = np.zeros_like(D)  # Initialize gradient matrix
        for i in range(x.shape[1]):
            error = x[:, i] - np.dot(D, alpha[:, i])
            grad -= 2 * lambda_0 * np.outer(error, alpha[:, i])
        return grad

    def solve_D_theta(self, alpha_opt, X, y, D_opt, theta_opt):
        """
        Updates `D` and `theta` given the optimal `alpha`.
        Do a projective gradient descent
        """

        D_init = D_opt
        theta_init = theta_opt

        for j in range(self.n_iter):
            # do the gradient descent for D
            grad_D = self.grad_E_D(X, D_init, alpha_opt, self.lambda0)
            D_opt = D_init - self.lr_D * grad_D
            D_opt /= np.linalg.norm(D_opt, axis=0)

            # do the gradient descent for theta
            grad_theta = self.grad_E_theta(y, D_opt, theta_init, alpha_opt, self.b, self.lambda2)
            theta_opt = theta_init - self.lr_theta * grad_theta

        return D_opt, theta_opt

    def fit(self, X, y):
        """Fits the model to the data."""
        n_samples, n_features = X.shape
        self.n_components = n_features
        D_opt = np.random.randn(n_features, self.n_components)
        D_opt /= np.linalg.norm(D_opt, axis=0)
        theta_opt = np.zeros(self.n_components)

        for _ in range(self.n_iter):
            alpha_opt = self.solve_alpha(X, y, D_opt, theta_opt)
            D_opt, theta_opt = self.solve_D_theta(alpha_opt, X, y, D_opt, theta_opt)

        self.alpha = alpha_opt
        self.D = D_opt
        self.theta = theta_opt[:-1]
        self.b = theta_opt[-1]

    def compute_alpha_from_D(self, x, D):
        """Computes sparse code `alpha` for a given sample `x`."""
        return np.zeros(D.shape[1])

    def predict(self, X):
        """Predicts labels for input data `X`."""
        alpha = np.array([self.compute_alpha_from_D(x, self.D) for x in X])
        return self.theta @ alpha.T + self.b

    def score(self, X, y):
        """Computes classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(np.round(y_pred) == y)
