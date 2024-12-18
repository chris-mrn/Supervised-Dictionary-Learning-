import numpy as np
import matplotlib.pyplot as plt
from braindecode.datasets import BNCI2014001
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
from skorch.helper import SliceDataset

class SyntheticTimeSeriesDataset:
    def __init__(self, num_classes=3, num_samples_per_class=100,
                 sequence_length=100):
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.sequence_length = sequence_length
        self.data = None
        self.labels = None

    def generate_class_data(self, class_id):
        np.random.seed(class_id)
        data, labels = [], []
        time = np.linspace(0, 2 * np.pi, self.sequence_length)

        for _ in range(self.num_samples_per_class):
            if class_id == 0:
                signal = (
                    np.sin((class_id + 1) * time) +
                    0.1 * np.random.randn(self.sequence_length)
                )
            elif class_id == 1:
                signal = (
                    np.cos((class_id + 1) * time) +
                    0.1 * np.random.randn(self.sequence_length)
                )
            elif class_id == 2:
                signal = (
                    np.sin((class_id + 1) * time) *
                    np.exp(-0.05 * time) +
                    0.1 * np.random.randn(self.sequence_length)
                )
            data.append(signal)
            labels.append(class_id)
        return np.array(data), np.array(labels)

    def create_dataset(self):
        data, labels = [], []
        for class_id in range(self.num_classes):
            class_data, class_labels = self.generate_class_data(class_id)
            data.append(class_data)
            labels.append(class_labels)
        self.data, self.labels = np.vstack(data), np.hstack(labels)
        return self.data, self.labels  # Explicitly return X and y

    def plot_examples(self):
        if self.data is None or self.labels is None:
            raise ValueError("Dataset has not been created. "
                             "Call create_dataset() first.")
        plt.figure(figsize=(10, 6))
        for class_id in range(self.num_classes):
            idx = class_id * self.num_samples_per_class
            plt.plot(self.data[idx], label=f"Class {class_id}")
        plt.title("Example Time Series for Each Class")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


class SyntheticEEGDataset:
    def __init__(self, num_classes=3, num_samples_per_class=100,
                 sequence_length=100, num_electrodes=32, noise_level=0.1,
                 random_seed=None):
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.sequence_length = sequence_length
        self.num_electrodes = num_electrodes
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.data = None
        self.labels = None
        if random_seed:
            np.random.seed(random_seed)

    def generate_class_data(self, class_id):
        np.random.seed(class_id + (self.random_seed if self.random_seed else 0))
        data, labels = [], []
        time = np.linspace(0, 2 * np.pi, self.sequence_length)

        for _ in range(self.num_samples_per_class):
            # Create empty data for all electrodes (n_electrodes x sequence_length)
            signal_matrix = np.zeros((self.num_electrodes, self.sequence_length))

            for electrode in range(self.num_electrodes):
                if class_id == 0:
                    signal = np.sin((class_id + 1) * time) + self.noise_level * np.random.randn(self.sequence_length)
                elif class_id == 1:
                    signal = np.cos((class_id + 1) * time) + self.noise_level * np.random.randn(self.sequence_length)
                elif class_id == 2:
                    signal = np.sin((class_id + 1) * time) * np.exp(-0.05 * time) + self.noise_level * np.random.randn(self.sequence_length)
                elif class_id == 3:
                    signal = np.sin((class_id + 1) * time) + np.cos((class_id + 2) * time) + self.noise_level * np.random.randn(self.sequence_length)
                else:
                    signal = np.sin((class_id + 1) * time) * np.exp(-0.01 * time) + self.noise_level * np.random.randn(self.sequence_length)

                # Assign the signal to the corresponding electrode
                signal_matrix[electrode] = signal

            # Append the generated signal matrix for the current sample
            data.append(signal_matrix)
            labels.append(class_id)
        return np.array(data), np.array(labels)

    def create_dataset(self):
        self.data = []
        self.labels = []
        for class_id in range(self.num_classes):
            class_data, class_labels = self.generate_class_data(class_id)
            self.data.append(class_data)
            self.labels.append(class_labels)

        self.data = np.vstack(self.data)
        self.labels = np.hstack(self.labels)
        return self.data, self.labels


class BNCI_Dataset:
    def __init__(self, subject_ids=[1], paradigm_name='LeftRightImagery',
                 low_cut_hz=4.0, high_cut_hz=38.0, factor=1e6,
                 trial_start_offset_seconds=-0.5, n_jobs=-1):
        self.subject_ids = subject_ids
        self.paradigm_name = paradigm_name
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.factor = factor
        self.trial_start_offset_seconds = trial_start_offset_seconds
        self.n_jobs = n_jobs
        self.dataset = None
        self.sfreq = None
        self._load_and_preprocess_data()

    def _pre_process_windows_dataset(self, dataset):
        """
        Preprocess the window (epoched) dataset.
        - Pick only EEG channels
        - Convert from V to uV
        - Bandpass filter
        - Apply exponential moving standardization
        """
        preprocessors = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data, factor: np.multiply(data, factor), factor=self.factor),
            Preprocessor("filter", l_freq=self.low_cut_hz, h_freq=self.high_cut_hz),
        ]

        preprocess(dataset, preprocessors, n_jobs=self.n_jobs)
        return dataset

    def _windows_data(self):
        """
        Create windows from the dataset.
        """
        # Define mapping of classes to integers
        if self.paradigm_name == "LeftRightImagery":
            mapping = {"left_hand": 1, "right_hand": 2}
        elif self.paradigm_name == "MotorImagery":
            mapping = {"left_hand": 1, "right_hand": 2, "feet": 4, "tongue": 3}

        self.dataset = self._pre_process_windows_dataset(self.dataset)
        sfreq = self.dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in self.dataset.datasets])
        self.sfreq = sfreq

        trial_start_offset_samples = int(self.trial_start_offset_seconds * self.sfreq)
        windows_dataset = create_windows_from_events(
            self.dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
            mapping=mapping,
        )

        return windows_dataset

    def _load_and_preprocess_data(self):
        """
        Load the BNCI dataset and preprocess it.
        """
        self.dataset = BNCI2014001(subject_ids=self.subject_ids)
        self.dataset = self._windows_data()

    def get_X_y(self):
        """
        Return X and y for the dataset.
        """
        X = SliceDataset(self.dataset, idx=0)
        y = np.array(list(SliceDataset(self.dataset, idx=1))) - 1  # Subtract 1 for compatibility
        return X, y
