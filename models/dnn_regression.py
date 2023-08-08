import os
import datetime
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from collections import namedtuple
from .dnn import DNN
from utils.save_model import save_model_dnn

class DNN_Regression(DNN):
    def __init__(
            self,
            training_data=None,
            testing_data=namedtuple,
            no_waps=8,
            random_state=42,
            preprocessor='standard_scaler',
            batch_size=8,
            epochs=10,
            optimizer=None,
            validation_split=None,
            dropout=0.2,
            tx_power=False,
            patience=10,
            checkpoint_filepath='../checkpoint/',
            save_model=True
        ):
        self.random_state = random_state
        self.training_data = training_data,
        self.testing_data = testing_data,
        self.no_waps = no_waps
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.dropout = dropout
        self.tx_power = tx_power
        self.patience = patience
        self.checkpoint_filepath = checkpoint_filepath
        self.save_model = save_model
        self.model = None

        self.training_data = training_data
        self.testing_data = testing_data

        time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f'dnn_{time_now}'

        self.rss_train_scaled = self.training_data.rss_scaled
        self.power_train = self.training_data.power
        self.xr_train_scaled = self.testing_data.labels.coords_scaled[:, 0]
        self.yr_train_scaled = self.testing_data.labels.coords_scaled[:, 1]

        self.rss_test_scaled = self.testing_data.rss_scaled
        self.power_test = self.testing_data.power
        self.xr_test_scaled = self.testing_data.labels.coords_scaled[:, 0]
        self.yr_test_scaled = self.testing_data.labels.coords_scaled[:, 1]

        print(self.xr_train_scaled)
        print(self.yr_train_scaled)
        print(self.xr_test_scaled)
        print(self.yr_test_scaled)

        print(len(self.rss_train_scaled))

        print('no waps ', self.no_waps)
        # initialize randoms
        if self.random_state != None:
            np.random(self.random_state)
            tf.random.set_seed(self.random_state)
        
    def build_model(self):
        
        if self.tx_power:
            # Input Layers
            input_rss = Input(shape=(self.no_waps,), name='input_rss')
            input_power = Input(shape=(self.no_waps,), name='input_power')

            # RSS Feature Extraction 
            hidden_rss = Dense(64, activation='relu')(input_rss)
            hidden_rss = Dropout(self.dropout)(hidden_rss)
            hidden_rss = Dense(32, activation='relu')(hidden_rss)

            # TX Power Feature Extraction
            hidden_power = Dense(64, activation='relu')(input_power)
            hidden_power = Dropout(self.dropout)(hidden_power)
            hidden_power = Dense(32, activation='relu')(hidden_power)

            # Concatenate to a merged feature
            merged = Concatenate()([hidden_rss, hidden_power])

            # Intermediate Layer
            hidden_merged = Dense(128, activation='relu')(merged)
            hidden_merged = Dropout(self.dropout)(hidden_merged)
            hidden_merged = Dense(64, activation='relu')(hidden_merged)

            # Output Layer
            output_xr = Dense(1, name='output_xr')(hidden_merged)
            output_yr = Dense(1, name='output_yr')(hidden_merged)

            model = Model(inputs=[input_rss, input_power], outputs=[output_xr, output_yr])

            model.compile(
                optimizer=self.optimizer,
                loss='mse',
                metrics=['mse']
            )
        else:
            # Input Layers
            input_rss = Input(shape=(self.no_waps,), name='input_rss')

            # RSS Feature Extraction 
            hidden_rss = Dense(64, activation='relu')(input_rss)
            hidden_rss = Dropout(self.dropout)(hidden_rss)
            hidden_rss = Dense(32, activation='relu')(hidden_rss)

            # Output Layer
            output_xr = Dense(1, name='output_xr')(hidden_rss)
            output_yr = Dense(1, name='output_yr')(hidden_rss)

            self.model = Model(inputs=[input_rss], outputs=[output_xr, output_yr])

            self.model.compile(
                optimizer=self.optimizer,
                loss='mse',
                metrics=['mse']
            )
    
    def train(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_filepath, self.model_name),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        earlystopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        if self.tx_power:
            self.model.fit(
                [self.rss_train_scaled, self.power_train], 
                [self.xr_train_scaled, self.yr_train_scaled],
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data =([self.rss_test_scaled, self.power_test], [self.xr_test_scaled, self.yr_test_scaled]),
                callbacks=[checkpoint_callback, earlystopping_callback]
            )
        else:
            self.model.fit(
                [self.rss_train_scaled],
                [self.xr_train_scaled, self.yr_train_scaled],
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=([self.rss_test_scaled], [self.xr_test_scaled, self.yr_test_scaled]),
                callbacks=[checkpoint_callback, earlystopping_callback]
            )
        
        print(self.model.summary())
    
    def predict(self, rss_predict, power_predict):
        pass

    def evaluate(self):
        if self.tx_power:
            xr_pred, yr_pred = self.model.predict([self.rss_test_scaled, self.power_test])
        else:
            xr_pred, yr_pred = self.model.predict([self.rss_test_scaled])
        
        xr_pred = self.preprocessor.inverse_transform(xr_pred)
        yr_pred = self.preprocessor.inverse_transform(yr_pred)
        xr_test = self.preprocessor.inverse_transform(self.xr_test_scaled)
        yr_test = self.preprocessor.inverse_transform(self.yr_test_scaled)

        xr_mse = mse(xr_test, xr_pred)
        yr_mse = mse(yr_test, yr_pred)
        xr_r2 = r2(xr_test, xr_pred)
        yr_r2 = r2(yr_test, yr_pred)

        print('-=-=-=-=-=-=-=-=- Metrics Evaluation -=-=-=-=-=-=-=-=-')
        print('Deep Neural Network (Regression)' )
        print('MSE for X Coordinates: ', xr_mse)
        print('MSE for Y Coordinates: ', yr_mse)
        print('\n')
        print('R2 Score for X Coordinates: ', xr_r2)
        print('R2 Score for Y Coordinates: ', yr_r2)
        print('-=-=-=-=-=-=-=-=- Metrics Evaluation -=-=-=-=-=-=-=-=-')

        if self.save_model:
            save_model_dnn(self.model_name, 
                           self.model, 
                           xr_mse=xr_mse,
                           yr_mse=yr_mse,
                           xr_r2=xr_r2,
                           yr_r2=yr_r2,
                           xr_test=xr_test,
                           yr_test=yr_test,
                           xr_pred=xr_pred,
                           yr_pred=yr_pred
            )
