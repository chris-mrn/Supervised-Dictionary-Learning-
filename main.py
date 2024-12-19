import argparse
from Models.SDL_simple import SDL_simple
from Models.SDL_logistic import SDL_logistic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skorch.helper import to_numpy
from Models.SDL_fixD import SDL_fixD  # Import SDL_fixD model
from Datasets.utils import load_dataset, PCA
import matplotlib.pyplot as plt


# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Load dataset and train model.")
parser.add_argument(
    "-d", "--dataset", type=str, required=True, choices=["bnci", "synthetic_eeg", "synthetic", "synthetic2"],
    help="Name of the dataset to load"
)
parser.add_argument(
    "-m", "--model", type=str, required=True, choices=["rf", "sdl_simple", "sdl_fixd", "sdl_logistic", "sdl_PCA"],
    help="Model to use: 'rf' for RandomForest, 'sdl' for SDL, 'sdl_fixd' for fixed dictionary SDL"
)

# Parse the arguments
args = parser.parse_args()

# Load the dataset based on the argument
X, y = load_dataset(args.dataset)

X = to_numpy(X)
y = to_numpy(y)

if args.model == 'sdl_PCA':
    n_components = 10
    X = PCA(X, n_components=n_components)

    # X is of shape (n_samples, n_channels, signal_length)
    # We can reshape it to (n_samples * n_channels, signal_length)
    X = X.reshape(X.shape[0], -1)

    y_new = []
    for i in range(len(y)):
        y_new = y_new + [y[i]] * n_components

    y = y_new
    X = X[:20]
    y = y[:20]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model based on the command-line argument
if args.model == "rf":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Reshape X_train from (n_samples, n_channels, signal_length)
    # to (n_samples, n_channels * signal_length)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train)
elif args.model == "sdl_simple":
    model = SDL_simple(n_iter=10)
    model.fit(X_train, y_train)
elif args.model == "sdl_logistic":
    model = SDL_logistic()
    model.fit(X_train, y_train)
elif args.model == "sdl_fixd":
    # Initialize SDL_fixD model
    # (assuming we need to define n_atoms and n_times_atom)
    model = SDL_fixD(n_atoms=10, n_times_atom=5, regularization_param=1.0)
    model.fit(X_train, y_train)
elif args.model == "sdl_PCA":
    model = SDL_simple()
    model.fit(X_train, y_train)


# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Test set accuracy:", accuracy)

# get the atoms and create X_reconstruct with the atoms
print(model.D.shape)
print(model.alpha.shape)

X_reconstruct = model.alpha @ model.D

# do a plot of X_reconstruct[0] and X[0] in the same plot
# with different colors for each
plt.plot(X_reconstruct[0], label='Reconstructed')
plt.plot(X[0], label='Original')
plt.legend()
plt.show()
