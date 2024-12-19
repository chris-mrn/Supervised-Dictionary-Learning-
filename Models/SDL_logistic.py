import numpy as np
from Models.utils import PrimalDualSolver_logistic
from Models.utils import ProjectedGradientDescent_logistic


class SDL_logistic:
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

    def solve_alpha(self, X, y, D, theta, b):
        """
        Optimizes sparse codes `alpha` for fixed `D` and `theta`.
        We can have here a explicit expression of our gradient regarding
        to alpha, so we can use a proximal gradient descent to optimize it.
        """
        n_samples, n_features = X.shape
        alpha = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            solver = PrimalDualSolver_logistic(
                theta=theta, b=b, x_i=x_i, y_i=y_i, D=D,
                lambda_0=self.lamnda0, lambda_1=self.lambda1,
                lambd=0.01, mu=1.0
            )
            # Solve the problem
            x0 = np.random.randn(n_features)  # Random initialization
            alpha_opt, _ = solver.solve(x0)

            alpha[i] = alpha_opt

        return alpha

    def solve_D_theta(self, alpha_opt, X, y, D_opt, theta_opt, b):
        """
        Updates `D` and `theta` given the optimal `alpha`.
        Do a projective gradient descent.
        """
        pgd = ProjectedGradientDescent_logistic(
            D_init=D_opt, theta_init=theta_opt,
            b=b, x=X, y=y, alphas=alpha_opt,
            lambda_0=self.lamnda0,
            lambda_1=self.lambda1, lambda_2=self.lambda2,
            lr=self.lr_D, max_iter=self.n_iter
        )
        D_opt, theta_opt, _ = pgd.optimize()
        return D_opt, theta_opt

    def fit(self, X, y):
        """Fits the model to the data."""
        n_samples, n_features = X.shape
        self.n_components = n_features
        D_opt = np.random.randn(n_features, self.n_components)
        D_opt /= np.linalg.norm(D_opt, axis=0)
        theta_opt = np.zeros(self.n_components)
        b_opt = 0

        for i in range(self.n_iter):
            alpha_opt = self.solve_alpha(X, y, D_opt, theta_opt, b_opt)
            D_opt, theta_opt, b_opt, _ = self.solve_D_theta(alpha_opt,
                                                            X,
                                                            y,
                                                            D_opt,
                                                            theta_opt,
                                                            b_opt)
            # Print the loss of after each iteration
            print(f"Iteration {i+1}/{self.n_iter}, Loss: {self.objective(X, y, D_opt, theta_opt, b_opt, alpha_opt)}")

        self.alpha = alpha_opt
        self.D = D_opt
        self.theta = theta_opt[:-1]
        self.b = theta_opt[-1]

    def predict(self, X):
        """Predicts labels for input data `X`."""
        return self.theta @ self.alpha.T + self.b

    def score(self, X, y):
        """Computes classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(np.round(y_pred) == y)
