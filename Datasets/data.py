import numpy as np
import matplotlib.pyplot as plt


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
