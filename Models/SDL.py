import numpy as np


class SDL:
    def __init__(self, dict_size, lambda_0=1e-4, lambda_1=1e-4, lambda_2=1e-4, max_iter=1000, eps=1e-4, gtol=0.2):
        """
        Initializes the Supervised Dictionary Learning (SDL) model.

        Parameters:
        - dict_size: int, size of the dictionary
        - lambda_0: float, regularization parameter for the smooth part
        - lambda_1: float, regularization parameter for the sparsity constraint
        - lambda_2: float, additional regularization term (if needed)
        - max_iter: int, maximum number of iterations for optimization
        - eps: float, tolerance for convergence
        - gtol: float, tolerance for gradient norm convergence
        """
        self.dict_size = dict_size
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter
        self.eps = eps
        self.gtol = gtol

    def fit(self, signals, alpha_init=None):
        """
        Fit the dictionary and coefficients to the provided signals.

        Parameters:
        - signals: ndarray, shape (n_classes, s_per_class, len_s)
        - alpha_init: ndarray, initial alpha coefficients (optional)

        Returns:
        - self: the fitted SDL model
        """
        len_s = signals.shape[2]
        s_per_class = signals.shape[1]
        n_classes = signals.shape[0]

        # Initialize dictionary, weights, and biases
        self.D = np.random.randn(len_s, self.dict_size)
        self.W = np.random.randn(self.dict_size, n_classes)
        self.b = np.random.randn(n_classes)

        if alpha_init is None:
            self.alpha = np.zeros((n_classes, s_per_class, self.D.shape[1]))
        else:
            self.alpha = alpha_init

        for iter in range(self.max_iter):
            # Optimization step for alpha
            self.alpha = self._supervised_sparse_coding(signals)

            # TODO: Implement optimization steps for D, W, and b (based on the original paper)

            if iter % 10 == 0:
                print(f"Iteration {iter}/{self.max_iter}")

        return self

    def _soft_thresholding(self, z, threshold):
        """Applies the soft-thresholding operator."""
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    def _gradient_f(self, alpha, W, b, l, D, x):
        """Compute the gradient of the linear function."""
        grad_l2 = -2 * self.lambda_0 * D.T @ (x - D @ alpha)

        diff_W = np.zeros(W.shape)
        diff_b = np.zeros(b.shape)
        for i in range(W.shape[1]):
            diff_W[:,i] = W[:,i] - W[:,l]
            diff_b[i] = b[i] - b[l]
        exp_terms = np.exp((diff_W).T @ alpha + diff_b)
        softmax_weights = exp_terms / np.sum(exp_terms)
        grad_logsumexp = diff_W @ softmax_weights

        return grad_logsumexp + grad_l2

    def _fixed_point_continuation(self, alpha, W, b, l, D, x):
        """
        Fixed Point Continuation (FPC) Algorithm.

        Solves: min_alpha f(alpha) + lambda * ||alpha||_1
        """
        mu_i = 0.99 * np.linalg.norm(alpha, ord=np.inf)
        list_alpha = []

        for iteration in range(self.max_iter):
            mu = min(mu_i * 4**(iteration - 1), 1 / self.lambda_1)
            tau = 1.999
            grad_f = self._gradient_f(alpha, W, b, l, D, x)
            alpha_temp = alpha - tau * grad_f

            v = tau * self.lambda_1
            alpha_next = self._soft_thresholding(alpha_temp, v)

            # Check for convergence
            if (np.linalg.norm(alpha_next - alpha, ord=2) / max(np.linalg.norm(alpha), 1) < self.eps) and \
               (mu * np.linalg.norm(grad_f, ord=np.inf) - 1 < self.gtol):
                print(f"Converged in {iteration + 1} iterations.")
                break

            alpha = alpha_next
            list_alpha.append(alpha)

        return alpha, list_alpha

    def _supervised_sparse_coding(self, signals):
        """
        Compute the supervised sparse coding for the dictionary learning algorithm.
        """
        s_per_class = signals.shape[1]
        n_classes = signals.shape[0]
        alpha_opt = np.zeros((n_classes, s_per_class, self.D.shape[1]))

        for l in range(n_classes):
            for j in range(s_per_class):
                alpha_opt[l, j, :] = self._fixed_point_continuation(self.alpha, self.W, self.b, l, self.D, signals[l, j, :])[0]

        return alpha_opt

    def transform(self, signals):
        """
        Predict the sparse coefficients for new signals using the learned dictionary.

        Parameters:
        - signals: ndarray, shape (n_classes, s_per_class, len_s)

        Returns:
        - alpha_opt: ndarray, sparse coefficients for the signals
        """
        return self._supervised_sparse_coding(signals)
