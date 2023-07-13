# Indoor Localization Framework

## Introduction
This repository is a framework for Indoor Positioning with Machine Learning approach to achieve the smallest error rate possible. Based on a paper in [WiFi](https://ieeexplore.ieee.org/document/7275492) Indoor Positioning which introduced data collection using a Client-based fingerprinting method.

## Dataset Loader

This is a Python class (`Loader`) that facilitates loading and processing of data from a dataset. The class provides methods to load, preprocess, and cache the data.

### Features

- Loads dataset from a file in cloudpickle format.
- Allows specifying a fraction of the dataset to load.
- Supports different preprocessing methods (standard scaler, min-max scaler, or normalization).
- Handles replacing the minimum RSSI value with a specified value.
- Caches the processed data for faster subsequent access.

### Requirements

- Python 3.x
- NumPy
- Pandas
- Cloudpickle
- Scikit-learn

### Usage

1. Import the `Loader` class:

    ```python
    from loader import Loader
    ```

2. Instantiate the `Loader` object with the desired parameters:

    ```python
    loader = Loader(path='path/to/dataset.pkl', frac=0.2)
    ```

    Available parameters:

    - `path` (str): The path to the dataset file.
    - `cache` (bool): Flag indicating whether to use cache.
    - `cache_fname` (str): The filename for caching the processed data.
    - `frac` (float): Fraction of the dataset to load.
    - `preprocessor` (str): Preprocessing method to use ('standard_scaler', 'min_max_scaler', or 'normalization').
    - `unnamed_ap_val` (int): Value to replace the minimum RSSI value.
    - `prefix` (str): Prefix for log messages.

3. Access the processed training and testing data using the `training_df` and `testing_df` attributes:

    ```python
    training_data = loader.training_df
    testing_data = loader.testing_df
    ```

4. Customize the class based on your requirements. You can modify the preprocessing methods, caching behavior, and file locations as needed.

### Notes

- The dataset file should be in cloudpickle format.
- Ensure that the required packages (NumPy, Pandas, Cloudpickle, Scikit-learn) are installed.


## Progress Reports

### 2023/06/30
Dataset loader has been created. Due to data unavailability as of now, preprocessing is delayed. The main goal for now is to load the data using pickling method and convert it into several pandas DataFrames.

### 2023/07/02
Started working on Deep AutoEncoder model to reduct the features of dataset. 
