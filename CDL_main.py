from alphacsc import BatchCDL
from alphacsc.init_dict import init_dictionary
from alphacsc import BatchCDL
import numpy as np
from Datasets.utils import load_dataset


def CDL_model(X, n_atoms, n_times_atom):


    D_init = init_dictionary(X,
                            n_atoms=n_atoms,
                            n_times_atom=n_times_atom,
                            rank1=False,
                            window=True,
                            D_init='chunk',
                            random_state=60)


    cdl = BatchCDL(
        # Shape of the dictionary
        n_atoms,
        n_times_atom,
        rank1=False,
        uv_constraint='auto',
        # Number of iteration for the alternate minimization and cvg threshold
        n_iter=3,
        # number of workers to be used for dicodile
        n_jobs=6,
        # solver for the z-step
        solver_z='lgcd',
        solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
        window=True,
        D_init=D_init,
        solver_d='fista',
        random_state=0)
    
    alpha = cdl._z_hat
    dictionary = cdl.D_hat_
    
    return alpha, dictionary 





if __name__ == "__main__":


    #load data
    X_train, y_train, Y_test, y_test, sfreq = load_dataset()    

    # Define the shape of the dictionary
    n_atoms = 25
    n_times_atom = int(round(sfreq * 1.0))  


    alpha, D = CDL_model(X=X_train, n_atoms=n_atoms, n_times_atom=n_times_atom)

