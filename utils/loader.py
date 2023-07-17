# @file     loader.py
# @author   danielandrewr

import os
import sys
import numpy as np
import pandas as pd

class Loader(object):
    def __init__(self,
                 path='../datas/dataset.csv',
                 frac=0.1,
                 preprocessor='standard_scaler',
                 prefix='IPS-LOADER',
                 no_val_rss=100,
                 floor=1
                 ):
        self.path = path
        self.frac = frac
        self.prepocessor = preprocessor
        self.prefix = prefix
        self.no_val_rss = no_val_rss
        self.floor = floor

        if preprocessor == 'standard_scaler':
            from sklearn.preprocessing import StandardScaler
            self.rssi_scaler = StandardScaler()
            self.coords_preprocessing = StandardScaler()
        elif preprocessor == 'min_max_scaler':
            from sklearn.preprocessing import MinMaxScaler
            self.rssi_scaler = MinMaxScaler()
            self.coords_preprocessing = MinMaxScaler()
        elif preprocessor == 'normalization':
            from sklearn.preprocessing import Normalizer
            self.rssi_scaler = Normalizer()
            self.coords_preprocessing = Normalizer()
        else:
            print('{} - Preprocessing Method is not Supported!', self.prefix)
            sys.exit(0)
        
        self.training_fname = path + '/trainingData.csv'
        self.testing_fname = path + '/testingData.csv'
        self.num_aps = 0
        self.training_data = None
        self.training_df = None
        self.testing_data = None
        self.testing_df = None
        self.load_data() # Load the Data

    def load_data(self):
        self.training_df = pd.read_csv(self.training_fname, header=0, frac=self.frac)
        self.testing_df = pd.read_csv(self.testing_fname, header=0, frac=self.frac)

        self.training_df = self.training_df[self.training_df['floor'] == self.floor]
        self.testing_df = self.testing_df[self.testing_df['floor'] == self.floor]
        
        self.no_waps = [cols for cols in self.training_df.columns if 'AP' in cols]
        self.waps_size = len(self.no_waps)

        if self.frac < 1.0:
            self.training_df = self.training_df.sample(frac=self.frac)
            self.testing_df = self.testing_df.sample(frac=self.frac)
        
        print('Training Data Loaded: ')
        print(self.training_df)
        print('Testing Data Loaded: ')
        print(self.testing_df)

    def process_data(self):
        
        # Get min and max rss values of training dan testing data
        rss_min_train, rss_max_train = self.get_min_max_rss(self.training_df)
        rss_min_train, rss_max_test = self.get_min_max_rss(self.testing_df)

        # Fill missing values rssi values with 100
        no_waps = self.no_waps
        self.training_df[no_waps] = self.training_df[no_waps].fillna(self.no_val_rss)
        self.testing_df[no_waps] = self.testing_df[no_waps].fillna(self.no_val_rss)

        rss_training = np.assarray(self.training_df[no_waps])
        rss_testing = np.asarray(self.testing_df[no_waps])
        
        # Scale the flattened rssi data
        if self.rssi_scaler is not None:
            rss_training_scaled = (self.rssi_scaler.fit_transform(
                rss_training.reshape((-1, 1)))).reshape(rss_training.shape)
            rss_testing_scaled = (self.rssi_scaler.fit_transform(
                rss_testing.reshape((-1, 1)))).reshape(rss_testing.shape)
        else:
            rss_training_scaled = rss_training
            rss_testing_scaled = rss_testing
        
        # Process Coords
        training_coord_x = np.asarray(self.training_df['LONGITUDE'], dtype=float)
        training_coord_y = np.asarray(self.training_df['LATITUDE'], dtype=float)
        training_coords = np.column_stack((training_coord_x, training_coord_y))

        testing_coord_x = np.asarray(self.testing_df['LONGITUDE'], dtype=float)
        testing_coord_y = np.asarray(self.testing_df['LATITUDE'], dtype=float)
        testing_coords = np.column_stack((testing_coord_x, testing_coord_y))

        # Scale the stacked coords data
        if self.coords_preprocessing is not None:
            training_coords_scaled = self.coords_preprocessing.fit_transform(training_coords)
            testing_coords_scaled = self.coords_preprocessing.fit_transform(testing_coords)
        else:
            training_coords_scaled = training_coords
            training_coords_scaled = testing_coords
        
    def save_data(self):
        pass

    def get_min_max_rss(self, data):
        min_val = data[self.no_waps].min().min()
        max_val = data[self.no_waps].max().max()

        return min_val, max_val