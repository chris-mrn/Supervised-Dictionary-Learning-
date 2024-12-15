from Models.SDL import SupervisedDictionaryLearning
from Datasets.data import SyntheticTimeSeriesDataset
from sklearn.model_selection import train_test_split

# Generate the entire dataset

dataset = SyntheticTimeSeriesDataset(num_classes=2,
                                     num_samples_per_class=100,
                                     sequence_length=100)
X, y = dataset.create_dataset()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# Use the train set to fit the model
# choice of parameters so the lambda1/lambda0 ratio is 0.15
model = SupervisedDictionaryLearning(n_components=10,
                                     lambda0=1,
                                     lambda1=0.15,
                                     lambda2=0.2,
                                     max_iter=1)
# do the training
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)

print("Test set accuracy:", accuracy)


""""""""""

import numpy as np
import matplotlib.pyplot as plt
from Models.utils import fixed_point_continuation, gradient_f

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

alpha_opt, all_alpha =
fixed_point_continuation(alpha_0,W, b, l, D, x, lambda_0, mu_bar, eps=1e-4,
gtol = 0.2, max_iter=1000)

print(alpha_opt)
print(all_alpha)
plt.plot([alpha_opt - all_alpha[i] for i in range(len(all_alpha))])
plt.show()
"""""""""""