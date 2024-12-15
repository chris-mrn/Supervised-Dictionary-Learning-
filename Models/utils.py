import numpy as np
import scipy


#In the case of binary classification

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
