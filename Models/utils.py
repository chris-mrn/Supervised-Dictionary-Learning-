import numpy as np


class SoftThresholding:
    """Soft thresholding operator"""
    @staticmethod
    def transform(z, threshold):
        """Applies the soft-thresholding operator."""
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
