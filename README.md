# Supervised Dictionary Learning 
Time Series MVA-2024

# Dataset Loader and Model Trainer

This repository contains a Python script for loading datasets and training a model using the `SDL` algorithm. The script allows users to choose between different datasets via command-line arguments.

## Requirements

Before running the script, ensure you have the following installed:

- Python 3.7+
- Required Python packages:
  - `scikit-learn`
  - `numpy`
  - `argparse`

## Supported Datasets

1. **Synthetic Dataset**
   - A time-series dataset with configurable classes, samples, and sequence length.

2. **BNCI Dataset**
   - EEG-based dataset supporting different paradigms, such as motor imagery tasks.

## How to Run

1. Clone the repository and navigate to the directory:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Run the script using the `-d` argument to specify the dataset:

   ```bash
   python main.py -d <dataset_name>
   ```

   Replace `<dataset_name>` with one of the following:

   - `synthetic` for the synthetic dataset.
   - `bnci` for the BNCI EEG dataset.

## Examples

### Run with the Synthetic Dataset:

```bash
python main.py -d synthetic
```

### Run with the BNCI Dataset:

```bash
python main.py -d bnci
```

## Output

The script will print the test set accuracy after training the model:

```bash
Test set accuracy: <accuracy_value>
```

## Extending the Script

To add more datasets, extend the `load_dataset` function in `main.py` and provide the appropriate logic for loading and preprocessing the new dataset.

## Notes

- Ensure that the `Models` and `Datasets` directories are properly set up and contain the required modules/classes for training and dataset handling.
- For BNCI datasets, ensure you have access to the raw EEG data files as required by the `BNCI_Dataset` class.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

