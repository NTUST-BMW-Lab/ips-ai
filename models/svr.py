# @file     svr.py
# @author   danielandrewr

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mse, r2_score as r2

from .regression_model import RegressionModel

class SVR(RegressionModel):
    def __init__(self, random_state):
        self.random_state = random_state
        self.model = None
        self.params = None
    
    def train(self, x_train_scaled, train_labels):
        # Define the hyperparameters search space
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        }

        svr_model = SVR(random_state=self.random_state)
        
        # RandomSearch initialization for hyperparameter tuning
        random_search = RandomizedSearchCV(
            estimator=svr_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',  # neg_mean_squared_error for evaluation metric.
            cv=3,  # balanced value for cross-validation
            random_state=self.random_state,
            n_jobs=-1  # use all available cores
        )

        random_search.fit(x_train_scaled, train_labels.coords_scaled)
        
        self.params = random_search.best_params_
        self.model = SVR(**self.params, random_state=self.random_state)
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
        for key, value in self.params.items():
            print(f'{key}: {value}')
        
        print('\nMetrics:')
        print(f'MSE: {mse_val}')
        print(f'R2 : {r2_val}')

        return mse_val, r2_val
