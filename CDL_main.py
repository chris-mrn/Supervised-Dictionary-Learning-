from alphacsc import BatchCDL
from alphacsc.init_dict import init_dictionary
from alphacsc import BatchCDL
import numpy as np
from Datasets.utils import load_dataset



if __name__ == "__main__":


    #load data
    X_train, y_train, Y_test, y_test, sfreq = load_dataset()

    # Define the shape of the dictionary
    n_atoms = 25
    n_times_atom = int(round(sfreq * 1.0))


    alpha, D = CDL_model(X=X_train, n_atoms=n_atoms, n_times_atom=n_times_atom)
