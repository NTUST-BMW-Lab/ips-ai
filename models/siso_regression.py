import os
import sys
import argparse
import math
import random
import pandas as pd
import numpy as np
import multiprocessing as mp

import tensorflow as tf
num_cpu = mp.cpu_count()
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=num_cpu,
    inter_op_parallelism_threads=num_cpu
)
from tensorflow import keras
from keras.layers import Activation, Dense, Dropout, Input
from keras import backend as b
from keras.metrics import categorical_accuracy
from keras.models import Model

from dae import dae
from sdae import sdae

def siso_regression(
        dataset: str,
        preprocessor: str,
        batch_size: int,
        epochs: int,
        optimizer: str,
        valdiation_split: float,
        dropout: float,
        dae_hidden_layer: list,
        sdae_hidden_layer: list,
        reg_hidden_layer: list,
        verbose: int
):
    np.random.seed()
    random.seed()
    tf.set_random_seed(random.randint(0, 1000000))
    tf_sess = tf.Session(
        graph=tf.get_default_graph(),
        config=session_conf
    )
    b.set_session(tf_sess)

    # Load Dataset
    training_data = pd.DataFrame(columns=['rssi_scaled', 'floor_height']) # Dummy

    rssi = training_data.rssi_scaled
    coord = training_data.coord_3d_scaled
    coord_scaler = training_data.coord_3d_scaler # inverse transform
    labels = training_data.labels
    input = Input(shape=(rssi.shape[1], ), name='input')

    if dae_hidden_layer != '':
        print('-=- Initializing Deep Autoencoder Model -=-')
        model = dae(
            dataset=dataset,
            input_data=rssi,
            preprocessor=preprocessor,
            batch_size=batch_size,
            epochs=epochs,
            hidden_layer=dae_hidden_layer,
            optimizer=optimizer,
            validation_split=valdiation_split
        )
        x = model(input)
    elif sdae_hidden_layer != '':
        print('-=- Initializing Stacked Denoising Autoencoder Model -=-')
        model = sdae(
            dataset=dataset,
            input_data=rssi,
            preprocessor=preprocessor,
            batch_size=batch_size,
            epochs=epochs,
            hidden_layer=sdae_hidden_layer,
            optimizer=optimizer,
            validation_split=valdiation_split
        )
        x = model(input)
    
    
        
