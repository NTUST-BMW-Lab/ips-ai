import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from keras.layers import Dense, Input
from keras.backend import backend as kback
from keras.activations import relu

from collections import namedtuple

def cae_classifier(
        dataset='',
        training_data=namedtuple,
        testing_data=namedtuple,
        preprocessing='',
        verbose=False,
        validation_split=0.2,
        hidden_layer=[],
        batch_size=32,
        random_state=42
):
    if (preprocessor == 'standard_scaler') or (preprocessor == 'normalizer'):
        loss = 'mean_squared_error'
    elif preprocessor == 'minmax_scaler':
        loss = 'binary_crossentropy'
    else:
        print('preprocessor not supported!')
        sys.exit()

    # initialize randoms
    np.random(random_state)
    tf.random.set_seed(random_state)

    # initialize scaled training data
    x_train_scaled = training_data.rss_scaled

    # initialize scaled labels data
    x_labels_scaled = training_data.labels.coords_scaled

    # define input layers
    input = Input(x_train_scaled)

    # define encoder layer
    def encoder():
        pass

    # define decoder layer
    def decoder():
        pass

    # build cae model
    def build_cae_model():
        pass

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument(
        '--p',
        '--preprocessor',
        help='processing method',
        dest='preprocessor',
        default='standard_scaler',
        type=str
    )
    arg_parse.add_argument(
        '--V',
        '--validation',
        help='Fraction of training data for validation',
        dest='validation',
        default=0.0,
        type=float
    )
    arg_parse.add_argument(
        '--H',
        '--hidden_layer',
        help="Number of units of DAE Hidden Layer seperated by comma (default: '128,32,128')",
        dest='hidden_layer',
        default='128,32,128',
        type=str
    )
    arg_parse.add_argument(
        '--R',
        '--random_seed',
        help='Random seed for modelling',
        default=0,
        type=int
    )

    preprocessor = arg_parse.preprocessor
    validation_split = arg_parse.validation
    hidden_layer = arg_parse.hidden_layer
    random_state = arg_parse.random_state
