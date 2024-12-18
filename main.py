import argparse
from Models.SDL_simple import SDL
from Datasets.data import SyntheticTimeSeriesDataset
from sklearn.model_selection import train_test_split
from Datasets.data import BNCI_Dataset
from sklearn.ensemble import RandomForestClassifier
from skorch.helper import to_numpy

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Load dataset and train model.")
parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="Name of the dataset to load"
)
parser.add_argument(
    "-m", "--model", type=str, required=True, choices=["rf", "sdl"],
    help="Model to use: 'rf' for RandomForest or 'sdl' for SDL"
)

# Parse the arguments
args = parser.parse_args()


def load_dataset(name):
    """
    Load the appropriate dataset based on the name provided.

    Parameters:
    -----------
    name: str
        Name of the dataset to load.

    Returns:
    --------
    X: np.array
        Features of the dataset.
    y: np.array
        Labels of the dataset.
    """
    if name == "synthetic":
        dataset = SyntheticTimeSeriesDataset(
            num_classes=2, num_samples_per_class=100, sequence_length=100
        )
        return dataset.create_dataset()
    elif name == "bnci":
        bnci_data = BNCI_Dataset(subject_ids=[1], paradigm_name='LeftRightImagery')
        X, y = bnci_data.get_X_y()
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {name}")


# Load the dataset based on the argument
X, y = load_dataset(args.dataset)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model based on the command-line argument
if args.model == "rf":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = to_numpy(X_train)
    X_test = to_numpy(X_test)
    # Reshape X_train from (n_samples, n_channels, signal_length) to (n_samples, n_channels * signal_length)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(X_test.shape)
    print(y_test.shape)
    model.fit(X_train, y_train)
elif args.model == "sdl":
    model = SDL()
    model.fit(X_train, y_train)

# Train the model
# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Test set accuracy:", accuracy)
