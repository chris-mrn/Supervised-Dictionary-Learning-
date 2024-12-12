import numpy as np


def soft_thresholding(z, threshold):
    """Applies the soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)



def gradient_f(alpha, W, b, l, D, x, lambda_0, method="linear"):
    """Compute the gradient of the linear function"""
    grad_l2 = -2 * lambda_0 * D.T @ (x - D @ alpha)

    diff_W = np.zeros(W.shape)
    diff_b = np.zeros(b.shape)
    if method == "linear":
        for i in range(W.shape[1]):
            diff_W[:,i] = W[:,i]- W[:,l] 
            diff_b[i] = b[i] - b[l]
        exp_terms = np.exp((diff_W).T @ alpha + diff_b)
        softmax_weights = exp_terms / np.sum(exp_terms)
        grad_logsumexp = diff_W @ softmax_weights
        return grad_logsumexp + grad_l2
    
    if method== "bilinear":
        print("Method not implemented")
        return np.nan
    else:
        print("Method not implemented")
        return np.nan



def fixed_point_continuation(alpha, W, b, l, D, x, lambda_0, lambda_1, eps=1e-4, gtol = 0.2, max_iter=1000):
    """
    Fixed Point Continuation (FPC) Algorithm.

    Solves: min_alpha f(alpha) + lambda * ||alpha||_1
    """
    list_alpha = []
    mu_i = 0.99 * np.linalg.norm(alpha, ord=np.inf)

    for iteration in range(max_iter):

        mu = min(mu_i * 4**(iteration-1), 1/lambda_1)

        # Gradient descent step for the smooth part f(alpha)
        tau = 1.999

        # Compute gradient
        grad_f = gradient_f(alpha, W, b, l, D, x, lambda_0)
        alpha_temp = alpha - tau* grad_f

        v = tau * lambda_1
        # Proximal step for the non-smooth part (g(x) = ||alpha||_1)
        alpha_next = soft_thresholding(alpha_temp, v)  #check where the lambda 0 goes

        # Check for convergence
        if (np.linalg.norm(alpha_next - alpha, ord=2) /max(np.linalg.norm(alpha), 1) < eps) and (mu * np.linalg.norm(grad_f, ord=np.inf) -1 < gtol):
            print(f"Converged in {iteration + 1} iterations.")
            break

        alpha = alpha_next

        if mu == 1/lambda_1:
            print(f"Converged in {iteration + 1} iterations.")
            break

        list_alpha.append(alpha)

    return alpha, list_alpha



def supervised_sparse_coding(alpha, W, b, D, x, lambda_0, lambda_1):
    """
    Compute the supervised sparse coding part of the supervised dictionnary learning algorithm
    """

    s_per_class = x.shape[1]
    n_classes = x.shape[0]
    alpha_opt = np.zeros((n_classes, s_per_class, D.shape[1]))

    for l in range(n_classes):
        for j in range(s_per_class):
            alpha_opt[l,j,:] = fixed_point_continuation(alpha, W, b, l, D, x, lambda_0, lambda_1, eps=1e-4, gtol = 0.2, max_iter=1000)

    return alpha_opt



def supervised_dictionary_learning(alpha, signals, dict_size, mu, lambda_1, lambda_0, lambda_2, eps=1e-4, max_iter =1000):

    len_s =signals.shape[2]
    s_per_class = signals.shape[1]
    n_classes = signals.shape[0]
    D = np.random.randn(len_s, dict_size)

    W = np.random.randn(dict_size, n_classes)
    b = np.random.randn(n_classes)

    for mu_i in mu: 
        iter = 0
        while iter<max_iter:
            #optimization step on alpha 
            alpha_opt = supervised_sparse_coding(alpha, W, b, D, signals, lambda_0, lambda_1)

            #optimization step on D, W  and b

            # A FINIR 
            iter += 1

    return D, W, b