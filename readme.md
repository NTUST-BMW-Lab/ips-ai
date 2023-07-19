# Indoor Localization Framework

## Introduction
This repository is a framework for Indoor Positioning with Machine Learning approach to achieve the smallest error rate possible. Based on a paper in [WiFi](https://ieeexplore.ieee.org/document/7275492) Indoor Positioning which introduced data collection using a Client-based fingerprinting method.

## Crawling Environment
RSSI Data crawling will be done in Dorm 1 in National Taiwan University of Science and Technology. Floor plan of the location can been seen below.
![1st Floor Plan](./assets/1f_plan.png)

*Green dots are pinpoint locations of additional APs to be placed*

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Pickle (File storing and loading)
- Matplotlib (Visualizations)
- Tensorflow (Deep Learning)

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

#### Hyperparameter Tuning
In machine learning, models have certain parameters that are learned during the training process, such as the weights in neural networks. However, hyperparameters are settings or configurations that are determined before the training process begins. These hyperparameters influence the learning process and performance of the model but are not directly learned from the data.

In this case, there are two parameters that will be tuned:
1. n_estimators: This hyperparameter determines the number of decision trees to be built in the Random Forest. Increasing the number of estimators generally improves performance, but it also increases computational complexity.
2. max_depth: The maximum depth allowed for each decision tree in the Random Forest. Limiting the depth helps prevent overfitting, as deeper trees can memorize noise in the training data.

These two parameters are crucial for the model performance, hence a hyperparemeter tuning method is used to evaluate each value performance in the provided dataset. `RandomizedSearchCV` is used and it is a process to search for the best set of hyperparameters for a machine learning model to achieve optimal performance. It works by performing a randomized search over a specified hyperparameter space to find the best combination of hyperparameters for a given machine learning model. 
It is a part of the scikit-learn library, and its main goal is to efficiently explore the hyperparameter space without trying out all possible combinations.

```python
param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30]
    }
```
Above is sets of parameters and their value combinations to be randomly searched.

Parameters of RandomizedSearchCV used:
```python
random_search = RandomizedSearchCV(
    estimator=rf_regressor,
    param_distributions=param_grid,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=random_state,
    n_jobs=-1
)
```

1. `estimator`: The base machine learning model (in this case, rf_regressor) that you want to tune. RandomizedSearchCV will create different instances of this model with various hyperparameter combinations and evaluate their performance.
2. `param_distributions`: A dictionary that maps hyperparameter names to their possible values or distributions. For example, {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]} indicates that n_estimators will be randomly selected from [50, 100, 150], and max_depth will be randomly selected from [None, 10, 20, 30]. RandomizedSearchCV will sample hyperparameters from these distributions during the search.
3. `n_iter`: The number of parameter settings that are sampled. It defines the number of random combinations of hyperparameters to try. The larger the value, the more combinations will be tested, potentially leading to better results. However, a larger value will also increase the computation time.
4. `scoring`: The scoring metric used to evaluate the performance of different hyperparameter combinations. In this case, we use 'neg_mean_squared_error', which means that the mean squared error (MSE) will be used as the evaluation metric. Since RandomizedSearchCV maximizes the score, we use negative MSE to find the combination with the lowest MSE.
5. `cv`: The number of folds in the cross-validation process. In this case, we use 3-fold cross-validation. Cross-validation helps to estimate the model's performance on unseen data and reduces overfitting.
6. `random_state`: The random seed for reproducibility. Setting a specific value ensures that the random sampling of hyperparameters is reproducible, meaning you will get the same results when running the tuning process multiple times.
7. `n_jobs`: The number of CPU cores to use for parallel processing. In this case, we set n_jobs=-1, which means to use all available cores. Parallel processing speeds up the hyperparameter tuning process, especially when there are multiple hyperparameter combinations to evaluate.

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


