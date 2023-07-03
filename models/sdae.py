# @file     sdae.py
# @author   danielandrewr

import os
import sys
import tensorflow as tf
from tensorflow import keras

def sdae(
        dataset='uji',
        input_data=None,
        preprocessor='standard_scaler',
        batch_size=32,
        epochs=50,
        hidden_layer=[],
        optimizer='nadam',
        validation_split=0.0
):
    '''
    sdae (Stacked Denoising AutoEncoder)
        Parameters:
            - dataset: Dataset for training and testing
            - input_data: RSSI 2D Array
            - preprocessor: Data preprocessing technique for input data
            - batch_size: Batch size
            - epochs: Number of iterations
            - hidden_layer: A list of number of units for Deep AutoEncoder hidden layer
            - optimizer: Training optimizer method
            - validation_split: Fraction of training data for validation
    '''
    if (preprocessor == 'standard_scaler') or (preprocessor == 'normalizer'):
        loss = 'mean_squared_error'
    elif preprocessor == 'minmax_scaler':
        loss = 'binary_crossentropy'
    else:
        print('preprocessor not supported!')
        sys.exit()