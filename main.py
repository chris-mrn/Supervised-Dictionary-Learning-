import numpy as np 
import matplotlib.pyplot as plt
from functions import fixed_point_continuation, gradient_f

p=5 #nbr of classes
k=256 #dimension of the dictionary
n=100 # dimension of the signal

#random variables
alpha_0 = np.random.randn(k) 
W = np.random.randn(k, p)
b = np.random.randn(p)
l = 1
D = np.random.randn(n, k)
x = np.random.randn(n)
lambda_0 = 1 #see lateer how to choose this
lambda_1 = 0.15 #see lateer how to choose this
mu_bar = 1/lambda_1


derivative = gradient_f(alpha_0, W, b, l, D, x, lambda_0, method="linear")

alpha_opt, all_alpha = fixed_point_continuation(alpha_0, W, b, l, D, x, lambda_0, mu_bar, eps=1e-4, gtol = 0.2, max_iter=1000)

print(alpha_opt)
print(all_alpha)
plt.plot([alpha_opt - all_alpha[i] for i in range(len(all_alpha))])
plt.show()