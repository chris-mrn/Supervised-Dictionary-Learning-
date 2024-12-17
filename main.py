from Models.SDL_simple import SDL
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

model = SDL()

# do the training
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)

print("Test set accuracy:", accuracy)
