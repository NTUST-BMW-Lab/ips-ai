import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle

def save(model_name, model, mse_val, r2_val, predicted_coords, folder_dest='../evaluation'):
        # Create the folder if it doesn't exist
        date_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join(folder_dest, f"{model_name}_{date_now}")
        os.makedirs(folder_path, exist_ok=True)

        # Save the evaluation results to a CSV file
        evaluation_df = pd.DataFrame({'MSE': [mse_val], 'R2-Score': [r2_val]})
        evaluation_df.to_csv(os.path.join(folder_dest), 'evaluation_results.csv')

        # Plotting
        plt.figure(figsize=(8,6))
        #continue plotting here..
        
        model_filename = os.path.join(folder_path, f'{model_name}_{date_now}.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
               
        print(f'Evaluations and Model has been saved to ~/evaluation/{model_filename}')

def save_model_dnn(model_name, model, xr_mse, yr_mse, xr_r2, yr_r2, xr_test, yr_test, xr_pred, yr_pred):
    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join('../evaluation/', f"{model_name}_{time_now}")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # metrics: {
    #     mse_val: [float, float],
    #     r2_score: [float, float],
    # }

    # Creating DataFrame from Metric Values 
    res = pd.DataFrame({
         'coords': ['xr', 'yr'],
         'mse': [xr_mse, yr_mse],
         'r2': [xr_r2, yr_r2]
    })

    # Save DataFrame to evaluation in csv format
    res.to_csv(os.path.join(folder_path, '/evaluation_results.csv'))

    # Create Visualizations
    sample = np.arange(len(xr_test))
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.plot(sample, xr_test, color='r', label='Actual')
    plt.plot(sample, xr_pred, color='b', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('X Position (Relative)')
    plt.title('Actual vs Predicted of X Relative Coordinate')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sample, yr_test, color='r', label='Actual')
    plt.plot(sample, yr_pred, color='b', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Y Position (Relative)')
    plt.title('Actual vs Predicted of Y Relative Coordinate')
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(folder_path, f'/actual_predicted_xr.png'), format='png')
    plt.savefig(os.path.join(folder_path, f'/actual_predicted_yr.png'), format='png')

    plt.close()

    print(f'Evaluations and Model has been saved to {folder_path}.')