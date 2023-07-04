# @file     sdae.py
# @author   danielandrewr

import os
import sys
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model, load_model

def masking_noise(x, corruption_level):
    x_corrupted = x
    x_corrupted[np.random.rand(len(x)) < corruption_level] = 0.0
    return x_corrupted

def sdae(
        dataset='uji',
        input_data=None,
        preprocessor='standard_scaler',
        batch_size=32,
        epochs=50,
        hidden_layer=[],
        corruption_level=0.1,
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
    
    input_dim = input_data.shape[1]
    input = Input(shape=(input_dim, ), name='sdae_input')

    encoded_input = []
    distorted_input = []
    encoded = []
    decoded = []
    autoencoder = []
    encoder = []
    x = input_data
    n_hl = len(hidden_layer)
    all_layers = [input_dim] + hidden_layer

    for i in range(n_hl):
        encoded_input.append(
            Input(
                shape=(all_layers[i], ),
                name='sdae_encoded_input' + str(i)
            )
        )
        encoded.append(
            Dense(
                all_layers[i+1],
                activation='sigmoid'
            )(encoded_input[i])
        )
        decoded.append(
            Dense(
                all_layers[i], activation='sigmoid'
            )(encoded[i])
        )
        autoencoder.append(
            Model(inputs=encoded_input[i], outputs=decoded[i])
        )
        encoder.append(Model(inputs=encoded_input[i], outputs=encoded[i]))
        autoencoder[i].compile(optimizer=optimizer, loss=loss)
        encoder[i].compile(optimizer=optimizer, loss=loss)
        autoencoder[i].fit(
            x=masking_noise(x, corruption_level),
            y=x,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            shuffle=True
        )
        x = encoder[i].predict(x)
    
    x = input
    for i in range(n_hl):
        x = encoder[i](x)
    
    output = x
    model = Model(inputs=input, output=output)
    model.compile(
        optimizer=optimizer,
        loss=loss
    )

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
    arg_parse.add_argument(
        '--CL',
        '--corruption_level',
        help='amount of noise intentionally added to the input data during the training phase.',
        default=0.1,
        type=float
    )

    preprocessor = arg_parse.preprocessor
    batch_size = arg_parse.batch_size
    optimizer = arg_parse.optimizer
    epochs = arg_parse.epochs
    validation = arg_parse.validation
    hidden_layer = arg_parse.hidden_layer
    frac = arg_parse.frac
    random_seed = arg_parse.random_seed
    corruption_level = arg_parse.corruption_level

    # Library Configurations
    import random
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Load the Data using loader.py
    training_data = pd.DataFrame()

    # Build the SDAE Model
    model = sdae(
        training_data,
        preprocessor=preprocessor,
        batch_size=batch_size,
        epochs=epochs,
        hidden_layer=hidden_layer,
        optimizer=optimizer,
        validation_split=validation,
        hidden_layer=hidden_layer,
        frac=frac,
        random_seed=random_seed,
        corruption_level=corruption_level
    )
    print(model.summary())