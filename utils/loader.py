# @file     loader.py
# @author   danielandrewr

import os
import sys
import numpy as np
import pandas as pd
import cloudpickle as cp

class Loader(object):
    def __init__(self,
                 path='../datas/dataset.csv',
                 cache=True,
                 cache_fname=None,
                 frac=0.1,
                 preprocessor='standard_scaler',
                 unnamed_ap_val=-110,
                 prefix='IPS-LOADER'
                 ):
        self.path = path
        self.cache = cache
        self.cache_fname = cache_fname
        self.frac = frac
        self.prepocessor = preprocessor
        self.unnamed_ap_val = unnamed_ap_val
        self.prefix = prefix

        if preprocessor == 'standard_scaler':
            from sklearn.preprocessing import StandardScaler
            self.rssi_preprocessing = StandardScaler()
            self.coords_preprocessing = StandardScaler()
        elif preprocessor == 'min_max_scaler':
            from sklearn.preprocessing import MinMaxScaler
            self.rssi_preprocessing = MinMaxScaler()
            self.coords_preprocessing = MinMaxScaler()
        elif preprocessor == 'normalization':
            from sklearn.preprocessing import Normalizer
            self.rssi_preprocessing = Normalizer()
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
        self.cache_loaded = False
        self.load_data() # Load the Data
        if not self.cache_loaded:
            self.process_data()
            self.save_data()

    def load_data(self):
        with open(self.path, 'rb') as input_file:
            self.training_data = cp.load(input_file)
            self.training_df = cp.load(input_file)
            self.testing_data = cp.load(input_file)
            self.testing_df = cp.load(input_file)
        
        self.training_df = pd.read_csv(
            self.training_fname,
            header=0,
            frac=self.frac
        )
        self.testing_df = pd.read_csv(
            self.testing_fname,
            header=0,
            frac=self.frac
        )
        self.no_waps = [cols for cols in self.training_df.columns if 'WAP' in cols]
        # Count Number of APS
        if not self.cache_loaded:
            self.cache_loaded = True

    def process_data(self):
        pass

    def save_data(self):
        pass
