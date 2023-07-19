# @file     randomforest.py
# @author   danielandrewr

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def random_forest(
       training_data=None,
       testing_data=None,
       random_state=None
):
    """
    Apply Random Forest algorithm with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
        training_data (namedtuple): Training data containing 'rss', 'rss_scaled', 'rss_scaler', and 'labels'.
        testing_data (namedtuple): Testing data containing 'rss', 'rss_scaled', and 'labels'.
        random_state (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        RandomForestRegressor: Trained Random Forest model with the best hyperparameters.
        float: Mean squared error (MSE) on the testing set with the best hyperparameters.
    """
    # Values extraction from namedtuples
    x_train, x_train_scaled, rss_scaler, train_labels = training_data
    x_test, x_test_scaled, test_labels = testing_data

    # Define the hyperparameters search space
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30]
    }

    rf_regressor = RandomForestRegressor(random_state=random_state)

    # RandomSearch initialization for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=rf_regressor,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_mean_squared_error', # neg_mean_squared_error for evaluation metric.
        cv=3, # balanced value for cross-validation
        random_state=random_state,
        n_jobs=-1 # use all available cores
    )

    # Perform hyperparameter tuning to the training data
    random_search.fit(x_train_scaled, train_labels.coords_scaled)

    best_params = random_search.best_params_ # Get the best parameters
    
    # Train the model on top of the best parameters found
    rf_model = RandomForestRegressor(**best_params, random_state=random_state)
    rf_model.fit(x_train_scaled, train_labels.coords_scaled)

    predictions = rf_model.predict(x_test_scaled)

    # Inverse transform the predictictions to get the original value
    predicted_coords = test_labels.coords_scaler.inverse_transform(predictions)

    # Count the MSE of the inverse-transformed coords
    mse = mean_absolute_error(test_labels.coords, predicted_coords)
    r2_score_rf = r2_score(test_labels.coords, predicted_coords)

    return rf_model, mse, r2_score_rf, predicted_coords

def evaluate(
        model_name,
        model,
        mse,
        r2_score_rf,
        predicted_coords,
        folder_destination='../evaluation/'
    ):
    """
    Evaluate the model, create visualizations, and save them to the specified folder.

    Parameters:
        model_name (str): Name of the model for creating the folder.
        model: Trained machine learning model.
        mse (float): Mean squared error (MSE) on the testing set.
        r2_score_rf (float): R-squared (R2) score on the testing set.
        predicted_coords: Predicted coordinates on the testing set.
        folder_destination (str): Folder path to save the visualizations. Default is "evaluation".

    Returns:
        None
    """

    # Create the folder if it doesn't exist
    date_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(folder_destination, f"{model_name}_{date_now}")
    os.makedirs(folder_path, exist_ok=True)

    # Save the evaluation results to a CSV file
    evaluation_df = pd.DataFrame({'MSE': [mse], 'R2-Score': [r2_score_rf]})
    evaluation_df.to_csv(os.path.join(folder_destination), 'evaluation_results.csv')

    # Plotting
    plt.figure(figsize=(8,6))
    
    model_filename = os.path.join(folder_path, 'trained_model.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f'Evaluations and Model has been saved to ~/evaluation/{model_filename}')
