from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from numpy import multiply
from skorch.helper import SliceDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
)
from braindecode.preprocessing import create_windows_from_events
from sklearn.model_selection import train_test_split
import numpy as np
from Datasets.data import BNCI_Dataset
from Datasets.data import SyntheticTimeSeriesDataset
from Datasets.data import SyntheticEEGDataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Utility functions to preprocess the dataset

def pre_process_windows_dataset(
    dataset, low_cut_hz=4.0, high_cut_hz=38.0, factor=1e6, n_jobs=-1
):
    """
    Preprocess the window dataset.
        Function to apply preprocessing to the window (epoched) dataset.
        We proceed as follows:
        - Pick only EEG channels
        - Convert from V to uV
        - Bandpass filter
        - Apply exponential moving standardization
    Parameters:
    -----------
    dataset: WindowsDataset or BaseConcatDataset
        Dataset to use.
    low_cut_hz: float
        Low cut frequency for the bandpass filter.
    high_cut_hz: float
        High cut frequency for the bandpass filter.
    factor: float
        Factor to convert from V to uV.
    n_jobs: int
        Number of jobs to use for parallelization.
    Returns:
    --------
    dataset: WindowsDataset or BaseConcatDataset
        Preprocessed dataset.
    """
    # Parameters for exponential moving standardization
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(
            lambda data, factor: multiply(data, factor),
            # Convert from V to uV
            factor=factor,
        ),
        # Bandpass filter
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    return dataset


def windows_data(
    dataset,
    paradigm_name,
    trial_start_offset_seconds=-0.5,
    low_cut_hz=4.0,
    high_cut_hz=38.0,
    factor=1e6,
    n_jobs=-1,
):
    """Create windows from the dataset.
    """
    # Define mapping of classes to integers
    # We use two classes from the dataset
    # 1. left-hand vs right-hand motor imagery
    if paradigm_name == "LeftRightImagery":
        mapping = {"left_hand": 1, "right_hand": 2}

    elif paradigm_name == "MotorImagery":
        mapping = {"left_hand": 1, "right_hand": 2, "feet": 4, "tongue": 3}

    dataset = pre_process_windows_dataset(
        dataset,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        factor=factor,
        n_jobs=n_jobs,
    )

    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this.
    # It needs parameters to define how trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=mapping,
    )

    return windows_dataset, sfreq


def load_dataset(name):
    """
    Load the appropriate dataset based on the name provided.
    """
    if name == "synthetic":
        dataset = SyntheticTimeSeriesDataset(
            num_classes=2, num_samples_per_class=100, sequence_length=100
        )
        return dataset.create_dataset()
    elif name == "synthetic_eeg":
        dataset = SyntheticEEGDataset(
            num_classes=2, num_samples_per_class=100, num_electrodes=22)
        return dataset.create_dataset()
    elif name == "bnci":
        bnci_data = BNCI_Dataset(subject_ids=[1], paradigm_name='LeftRightImagery')
        X, y = bnci_data.get_X_y()
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {name}")
    

def PCA(data, n_componante):
    data_reduct = np.zeros((data.shape[0], n_componante, data.shape[2]))
    for i in range(data.shape[0]):
        standardized_data = StandardScaler().fit_transform(data[i])
        pca = PCA(n_components=8)
        X_reduct = pca.fit_transform(standardized_data.T)
        data_reduct[i] = X_reduct.T
        
    return data_reduct
