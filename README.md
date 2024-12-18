

# Supervised Dictionary Learning  
**Time Series MVA-2024**

## Dataset Loader and Model Trainer

This repository contains a Python script that allows users to load time-series datasets and train models using various algorithms. The script provides functionality to load multiple datasets and choose between different models via command-line arguments, including Supervised Dictionary Learning (SDL) and other `sklearn` classifiers.

## Requirements

Before running the script, ensure you have the following installed:

- Python 3.7+
- Required Python packages:
  - `scikit-learn`
  - `numpy`
  - `argparse`
  - `torch` (if using SDL or other PyTorch-based models)
  
You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Supported Datasets

1. **Synthetic Dataset**
   - A synthetic time-series dataset with configurable classes, samples, and sequence length. This dataset is useful for benchmarking and testing various models.

2. **BNCI Dataset**
   - An EEG-based dataset, ideal for motor imagery tasks, supporting different paradigms (e.g., Left-Right Motor Imagery). This dataset is used in BCI (Brain-Computer Interface) applications.

## Supported Models

You can choose between multiple models to train on the selected dataset:

1. **RandomForestClassifier** (`rf`)
   - A simple and efficient classifier based on decision trees. Suitable for quick baselines.
   
2. **Supervised Dictionary Learning (SDL)** (`sdl`)
   - A more advanced algorithm suitable for time-series and high-dimensional data. Can be used for feature extraction and classification in a single step.

## How to Run

1. Clone the repository and navigate to the directory:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script using the `-d` argument to specify the dataset and the `-m` argument to specify the model:

   ```bash
   python main.py -d <dataset_name> -m <model_name>
   ```

   Replace `<dataset_name>` with one of the following:
   - `synthetic` for the synthetic dataset.
   - `bnci` for the BNCI EEG dataset.

   Replace `<model_name>` with one of the following:
   - `rf` for RandomForestClassifier.
   - `sdl` for Supervised Dictionary Learning.

### Example Commands

#### Run with the Synthetic Dataset using RandomForest:

```bash
python main.py -d synthetic -m rf
```

#### Run with the BNCI Dataset using SDL:

```bash
python main.py -d bnci -m sdl
```

#### Run with the BNCI Dataset using RandomForest:

```bash
python main.py -d bnci -m rf
```

## Output

After running the script, the model will train on the provided dataset, and the script will output the test set accuracy:

```bash
Test set accuracy: <accuracy_value>
```

## Extending the Script

To add more datasets or models, you can extend the following parts of the code:

- **Adding New Datasets**: Modify the `load_dataset` function in `main.py` to include logic for loading and preprocessing the new dataset.
- **Adding New Models**: Extend the script by adding new model options in the argument parser and initialize them in the model selection section.

## Notes

- Ensure that the `Models` and `Datasets` directories are properly set up and contain the required modules/classes for training and dataset handling.
- For BNCI datasets, ensure you have access to the raw EEG data files as required by the `BNCI_Dataset` class.
- If using `SDL`, make sure you have `torch` installed in your environment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

