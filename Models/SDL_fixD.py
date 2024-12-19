from sklearn.metrics import accuracy_score
from alphacsc import BatchCDL
from alphacsc.init_dict import init_dictionary
from sklearn.linear_model import Ridge


class SDL_fixD:
    """
    Fix the dictionary to a wavelet basis and
    only learn the linear model theta and b for binary classification.

    The model computes the coefficients of the sparse coding
    for each signal and then learns the linear model theta and b.
    """

    def __init__(self, n_atoms, n_times_atom, n_iter=3, regularization_param=None, max_iter=1000):
        """
        Initialize the model with a chunk of the data as dictionary and optional regularization parameter.

        Args:
            n_atoms: Number of atoms in the dictionary.
            n_times_atom: The size of the time window for each atom.
            regularization_param: Regularization parameter for the linear model (optional).
        """
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.regularization_param = regularization_param
        self.theta = None  # Linear model parameter (weights)
        self.b = None  # Linear model parameter (bias)
        self.D = None  # Empty dictionary
        self.alpha = None  # Coefficients for each signal
        self.max_iter = 1000
        self.n_iter = n_iter

    def CDL_model(self, X):
        """
        Computes the coefficients and dictionary using BatchCDL.

        Args:
            X: The input signal matrix.

        Returns:
            None
        """
        D_init = init_dictionary(
            X,
            n_atoms=self.n_atoms,
            n_times_atom=self.n_times_atom,
            rank1=False,
            window=True,
            D_init='chunk',
            random_state=60)

        cdl = BatchCDL(
            n_atoms=self.n_atoms,
            n_times_atom=self.n_times_atom,
            rank1=False,
            uv_constraint='auto',
            n_iter=self.n_iter,
            n_jobs=6,
            solver_z='lgcd',
            solver_z_kwargs={'tol': 1e-2, 'max_iter': self.max_iter},
            window=True,
            D_init=D_init,
            solver_d='fista',
            random_state=0)

        # Compute coefficients and dictionary
        cdl.fit(X)
        self.alpha = cdl._z_hat  # Coefficients for each signal
        self.D = cdl.D_hat_  # Learned dictionary

    def fit(self, X, y):
        """
        Fit the model by first computing the sparse coding coefficients and then learning the linear model.

        Args:
            X: The input signal matrix.
            y: The binary target values corresponding to the signals (0 or 1).

        Returns:
            None
        """
        # Step 1: Compute coefficients for each signal
        self.CDL_model(X)

        # Step 2: Learn the linear model (theta and b) using the coefficients
        model = Ridge(alpha=self.regularization_param)
        # if dim of alpha is more than 2, then reshape it to 2D

        self.alpha = self.alpha.reshape(self.alpha.shape[0], -1)
        model.fit(self.alpha, y)  # Fit a linear model
        self.theta = model.coef_  # Linear model parameters (weights)
        self.b = model.intercept_  # Linear model bias

    def predict(self, X):
        """
        Predict using the learned linear model and classify the output for binary classification.

        Args:
            X: The input signal matrix (shape [n_samples, n_features]).

        Returns:
            Predicted binary labels (0 or 1) based on the linear model.
        """
        # Compute coefficients for the new data using the learned dictionary
        self.CDL_model(X)
        self.alpha = self.alpha.reshape(self.alpha.shape[0], -1)

        # Predict based on the learned linear model (theta, b)
        y_pred_continuous = self.alpha.dot(self.theta.T) + self.b

        # Apply threshold to classify as binary (0 or 1)
        y_pred = (y_pred_continuous > 0.5).astype(int)
        return y_pred

    def get_parameters(self):
        """
        Return the learned parameters of the model.

        Returns:
            Tuple containing theta and b.
        """
        return self.theta, self.b

    def score(self, X, y):
        """
        Compute the classification accuracy of the model on the given data.

        Args:
            X: The input features (signal matrix).
            y: The true binary target values (0 or 1).

        Returns:
            Accuracy score.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy
