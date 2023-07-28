import os
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