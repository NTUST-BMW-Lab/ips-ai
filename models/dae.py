import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Model, Sequential, load_model

def dae(
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
    dae (Deep Autoencoder)
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

    model = Sequential()
    input_size = input_data.shape[1]
    model.add(
        Dense(
            hidden_layer[0],
            input_dim=input_size,
            activation='relu',
            name='dae_dense_1'
        )
    )
    n_hl = 1
    for units in hidden_layer[1]:
        n_hl += 1
        model.add(
            Dense(
                units,
                activation='relu',
                name='dae_dense'+str(n_hl)
            )
        )
    model.add(
        Dense(
            input_size,
            activation='sigmoid',
            name='dae_output'
        ))
    
    model.compile(
        optimizer=optimizer,
        loss=loss
    )

    history = model.fit(
        input_data,
        input_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=True
    )

    # Decoder Removing
    num_to_remove = (len(hidden_layer) + 1) // 2
    for i in range(num_to_remove): model.pop()

    return model

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument(
        '--P',
        '--preprocessor',
        help='Data preprocessing technique for input data',
        default='standard_scaler',
        type=str
    )
    arg_parse.add_argument(
        '--B',
        '--batch_size',
        help='Batch size',
        default=32,
        type=int
    )
    arg_parse.add_argument(
        '--O',
        '--optimizer',
        help='Training optimizer method',
        default='nadam',
        type=str
    )
    arg_parse.add_argument(
        '--E',
        '--epochs',
        help='Number of iterations',
        default=50,
        type=int
    )
    arg_parse.add_argument(
        '--V',
        '--validation',
        help='Fraction of training data for validation',
        default=0.0,
        type=float
    )
    arg_parse.add_argument(
        '--H',
        '--hidden_layer',
        help="Number of units of DAE Hidden Layer seperated by comma (default: '128,32,128')",
        default='128,32,128',
        type=str
    )
    arg_parse.add_argument(
        '--F',
        '--frac',
        help='Fraction of input data to load for training and validation (default: 1.0)',
        default=1.0,
        type=float
    )
    arg_parse.add_argument(
        '--R',
        '--random_seed',
        help='Random seed for modelling',
        default=0,
        type=int
    )

    preprocessor = arg_parse.preprocessor
    batch_size = arg_parse.batch_size
    optimizer = arg_parse.optimizer
    epochs = arg_parse.epochs
    validation = arg_parse.validation
    hidden_layer = arg_parse.hidden_layer
    frac = arg_parse.frac
    random_seed = arg_parse.random_seed

    # Library Configurations
    import random
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Load Data
    training_data = pd.DataFrame()
    # Build DAE Model
    model = dae(
        training_data,
        preprocessor=preprocessor,
        batch_size=batch_size,
        epochs=epochs,
        hidden_layer=hidden_layer,
        optimizer=optimizer,
        validation_split=validation
    )
    print(model.summary())

