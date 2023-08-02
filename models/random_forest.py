# @file     randomforest.py
# @author   danielandrewr

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mse, r2_score as r2

from .regression_model import RegressionModel

class RandomForest(RegressionModel):
    """
    A class representing a RandomForest regression model for predicting coordinates using hyperparameter tuning.

    Parameters:
        random_state (int, optional): Seed value for reproducibility. Default is None.

    Attributes:
        random_state (int, optional): Seed value for reproducibility.
        model (object): The RandomForestRegressor model after hyperparameter tuning.
        params (dict): The best hyperparameters found during hyperparameter tuning.

    Methods:
        __init__(random_state=None): Constructor method that initializes the RandomForest object.
        train(x_train_scaled, train_labels): Trains the RandomForest model using hyperparameter tuning.
        predict(x_test_scaled, test_labels): Predicts the coordinates for the given test data.
        evaluate(test_labels, predicted_coords): Evaluates the model's performance using Mean Squared Error (MSE) 
                                                  and R-squared (R2) metrics and prints the results.

    Inherited Attributes:
        Inherits attributes from the RegressionModel class.

    Inherited Methods:
        Inherits methods from the RegressionModel class.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None
        self.params = None
    
    def train(self, x_train_scaled, train_labels):
        # Define the hyperparameters search space
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30]
        }
        rf_regressor = RandomForestRegressor(random_state=self.random_state)

        # RandomSearch initialization for hyperparameter tuning
        random_search = RandomizedSearchCV(
            estimator=rf_regressor,
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error', # neg_mean_squared_error for evaluation metric.
            cv=3, # balanced value for cross-validation
            random_state=self.random_state,
            n_jobs=-1 # use all available cores
        )

        # Perform hyperparameter tuning to the training data
        random_search.fit(x_train_scaled, train_labels.coords_scaled)

        self.params = random_search.best_params_ # Get the best parameters
        self.model = RandomForestRegressor(**self.params, random_state=self.random_state)
        self.model.fit(x_train_scaled, train_labels.coords_scaled)

    def predict(self, x_test_scaled, test_labels):
        predictions = self.model.predict(x_test_scaled)
        predicted_coords = test_labels.coords_scaler.inverse_transform(predictions)
        return predicted_coords
    
    def evaluate(self, test_labels, predicted_coords):
        mse_val = mse(test_labels.coords, predicted_coords)
        r2_val = r2(test_labels.coords, predicted_coords)
        
        print('Summary of Random Forest Model: ')
        print('Parameters: ')
        for key, value in self.params.items():
            print(f'{key}: {value}')
        
        print('\nMetrics:')
        print(f'MSE: {mse_val}')
        print(f'R2 : {r2_val}')

        return mse_val, r2_val