# Indoor Localization Framework

## Introduction
This repository is a framework for Indoor Positioning with Machine Learning approach to achieve the smallest error rate possible. Based on a paper in [WiFi](https://ieeexplore.ieee.org/document/7275492) Indoor Positioning which introduced data collection using a Client-based fingerprinting method.

## Crawling Environment
RSSI Data crawling will be done in Dorm 1 in National Taiwan University of Science and Technology. Floor plan of the location can been seen below.
// Insert Image Here

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Dataset Loader

The Loader class is a data loading and preprocessing utility for indoor positioning datasets. It facilitates the loading of training and testing data from CSV files and performs preprocessing on the data using various methods such as StandardScaler, MinMaxScaler, and Normalizer.

### Class Initialization
```python
class Loader(object):
    def __init__(self,
                 path='../datas/',
                 frac=0.1,
                 preprocessor='standard_scaler',
                 prefix='IPS-LOADER',
                 no_val_rss=100,
                 floor=1
                 ):
```
Parameters:
1. `path` (str, optional): The path to the directory where the data files are located. Default is '../datas/'.
2. `frac` (float, optional): The fraction of data to be sampled. Should be a float between 0.0 and 1.0. Default is 0.1.
3. `preprocessor` (str, optional): The method to be used for preprocessing the data. Available options are 'standard_scaler', 'min_max_scaler', and 'normalization'. Default is 'standard_scaler'.
4. `prefix` (str, optional): A prefix string used for logging or identification purposes. Default is 'IPS-LOADER'.
5. `no_val_rss` (int, optional): The value used to indicate missing RSS (Received Signal Strength) values. Default is 100.
6. `floor` (int, optional): The floor number for which data should be loaded. Default is 1.

### Preprocessing Methods
The Loader class supports the following preprocessing methods:
1. Standard Scaler: Uses sklearn.preprocessing.StandardScaler for data standardization.
2. Min-Max Scaler: Uses sklearn.preprocessing.MinMaxScaler for data normalization to a given range.
3. Normalization: Uses sklearn.preprocessing.Normalizer for data normalization to unit norm.

The appropriate preprocessing method is selected based on the preprocessor parameter during object initialization.

### Example Usage
```python
# Create a Loader object with custom parameters
dataset = Loader(
    path='path_to_data_directory',
    frac=0.2,
    preprocessor='min_max_scaler',
    prefix='IPS-EXPERIMENT',
    no_val_rss=-90,
    floor=2
)

# Load and preprocess the data
dataset.load_data()
dataset.process_data()

# Access preprocessed training and testing data
print(dataset.training_data)
print(dataset.testing_data)
```

### Notes

- The dataset file should be in csv format.
- Ensure that the required packages (NumPy, Pandas, Scikit-learn) are installed

## Models

### Random Forest (random_forest.py)
implements a Random Forest Regression model with hyperparameter tuning using RandomizedSearchCV. This script includes functions for training the model, evaluating its performance, and saving the evaluation results, visualizations, and the trained model.

#### Main Functions
1. Hyperparameter tuning for the Random Forest model using RandomizedSearchCV.
2. Training the Random Forest model with the best hyperparameters.
3. Evaluating the model's performance using Mean Squared Error (MSE) and R-squared (R2) score on the testing set.
4. Creating visualizations of the true vs. predicted coordinates.
5. Saving the evaluation results and visualizations in a dedicated folder.

#### Training Model
Apply the Random Forest algorithm with hyperparameter tuning using RandomizedSearchCV.

Parameters:
1. `training_data` (namedtuple): Training data containing 'rss', 'rss_scaled', 'rss_scaler', and 'labels'.
2. `testing_data` (namedtuple): Testing data containing 'rss', 'rss_scaled', and 'labels'.
3. `random_state` (int, optional): Random seed for reproducibility. Default is None.

## Progress Reports

### 2023/07/17
Reworking on loader and preprocessing to match the characteristics of new dataset.

### 2023/07/02
Started working on Deep AutoEncoder model to reduct the features of dataset. 

### 2023/06/30
Dataset loader has been created. Due to data unavailability as of now, preprocessing is delayed. The main goal for now is to load the data using pickling method and convert it into several pandas DataFrames.


