# @file     svr.py
# @author   danielandrewr

import numpy as np
from sklearn.svm import SVR as ScikitSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mse, r2_score as r2

from .regression_model import RegressionModel

class SVR(RegressionModel):
    """
    A class representing a Support Vector Regression (SVR) model for predicting coordinates.

    Parameters:
        random_state (int, optional): Seed value for reproducibility. Default is None.

    Attributes:
        random_state (int, optional): Seed value for reproducibility.
        model (object): The SVR model.
        params (dict, optional): The hyperparameters for SVR (not used in this implementation).

    Methods:
        __init__(random_state=None): Constructor method that initializes the SVR object.
        train(x_train_scaled, train_labels): Trains the SVR model using the given training data and labels.
        predict(x_test_scaled, test_labels): Predicts the coordinates for the given test data.
        evaluate(test_labels, predicted_coords): Evaluates the model's performance using Mean Squared Error (MSE) 
                                                  and R-squared (R2) metrics and prints the results.

    Inherited Attributes:
        Inherits attributes from the RegressionModel class.

    Inherited Methods:
        Inherits methods from the RegressionModel class.

    Note:
        - The hyperparameter tuning using RandomizedSearchCV is commented out in this implementation.
        - The 'params' attribute is not used in this implementation and remains None.
        - The 'ScikitSVR' class is assumed to be a valid implementation of SVR from Scikit-learn.
        - The 'mse' and 'r2' functions are assumed to be available for calculating performance metrics.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None
        self.params = None
    
    def train(self, x_train_scaled, train_labels):
        np.random.seed(self.random_state)
        # Define the hyperparameters search space
        # param_grid = {
        #     'estimator': [20, 50, 100, 200, 300, 500, 700, 1000],
        #     'n_jobs': [1, 2, 3, 5, 10, 15, 20]
        # }

        svr_model = ScikitSVR()
        mtl_svr = MultiOutputRegressor(svr_model)
        
        # RandomSearch initialization for hyperparameter tuning
        # random_search = RandomizedSearchCV(
        #     estimator=mtl_svr,
        #     param_distributions=param_grid,
        #     n_iter=10,
        #     scoring='neg_mean_squared_error',  # neg_mean_squared_error for evaluation metric.
        #     cv=3,  # balanced value for cross-validation
        #     random_state=self.random_state,
        #     n_jobs=-1  # use all available cores
        # )

        # random_search.fit(x_train_scaled, train_labels.coords_scaled)
        
        # self.params = None
        self.model = MultiOutputRegressor(ScikitSVR())
        self.model.fit(x_train_scaled, train_labels.coords_scaled)

    def predict(self, x_test_scaled, test_labels):
        predictions = self.model.predict(x_test_scaled)
        predicted_coords = test_labels.coords_scaler.inverse_transform(predictions)
        return predicted_coords
    
    def evaluate(self, test_labels, predicted_coords):
        mse_val = mse(test_labels.coords, predicted_coords)
        r2_val = r2(test_labels.coords, predicted_coords)

        print('Summary of SVR Model: ')
        print('Parameters: ')
        if self.params is not None:
            for key, value in self.params.items():
                print(f'{key}: {value}')
        
        print('\nMetrics:')
        print(f'MSE: {mse_val}')
        print(f'R2 : {r2_val}')

        return mse_val, r2_val
