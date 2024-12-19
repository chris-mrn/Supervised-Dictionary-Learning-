

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
  - `braindecode`
  - `moabb`
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

1. **Baseline Models such**
   - A simple and efficient classifier based on decision trees. Suitable for quick baselines.

2. **Supervised Dictionnare Learning l2**
   - A more advanced algorithm suitable for time-series and high-dimensional data. Can be used for feature extraction and classification in a single step.
   - It uses a l2 penality term for classification
   
3. **Supervised Dictionnare Learning logictic**
   - Same but with a logistic penalty for classifcation
     
   
### Example Commands
The files are composed of several notebooks that allows to run and test the different models of this project. 
- Dataset analysis can be done in the EEG_data_analysis.ipynb notebook. 
- Test_Simple_SDL.ipynb can be used to run the simplified version of the Supervised Dictionary Learning model version
- Test_Logistic_SDL.ipynb can be used to run the logistic version of the Supervised Dictionary Learning model version
- Test_CDL.ipynb can be used to run the convolutional dictionary leanring model version

Additional files contain the codes needed to run the different python notebooks. 
- Test_PGD_l2 and Test_PGD_logistic where used to test the convergence of the projective gradient descent algorithms
- Test_prox_l2 and Test_prox_logistic where used to test the convergence of the proximal algorithm

## Notes

- Ensure that the `Models` and `Datasets` directories are properly set up and contain the required modules/classes for training and dataset handling.
- For BNCI datasets, ensure you have access to the raw EEG data files as required by the `BNCI_Dataset` class.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


