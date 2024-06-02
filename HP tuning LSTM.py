# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:21:31 2024

@author: mathi
"""
# Standard libraries
import os
import json

# Data handling and numerical tools
import numpy as np
import pandas as pd
from numpy import hstack
import scipy.stats

# Machine Learning and Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Deep Learning
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Hyperparameter Tuning
from keras_tuner.tuners import RandomSearch, Hyperband

#%%

# Configure TensorFlow to optimize inter and intra operation parallelism
config = tf.compat.v1.ConfigProto()

# Sets the number of threads used for parallel operations
config.inter_op_parallelism_threads = 8  # Adjust based on your CPU's capability
config.intra_op_parallelism_threads = 16  # Adjust based on your CPU's capability

# Apply the configuration to TensorFlow session
tf.compat.v1.Session(config=config)

#%%
pd.options.mode.chained_assignment = None
scale = StandardScaler()
#scale=MinMaxScaler(feature_range=(-1, 1))
tf.random.set_seed(0)

# Inputs 
pathin= "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Initiation/part 2/"
path= "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Initiation/part 2/"
L1=3 ; L2=8 ; e='e1' ; m = {'e1':[2,6],'e5':[0,3]} ; 
tt='1H' ; perce=0.9 ; n_past = 48  ; n_future = 12 ; 
MAX_EPOCHS= 2 ; batchsize=18 ; baseline=False

""" variable to forecast is always the last one!!! because here the 'scale' is done for each column 
    one by one and it is the last one that is kept for the inverse_transform 
"""
input_group = {
    # 0: ['log'+e], 
    # 1: ['SYM/H','log'+e],
    # 2: ['SYM/H','AL-index','log'+e],
    # 3: ['LU', 'cosMLTU', 'sinMLTU', 'lat', 'log'+e],  
    4: ['log'+e, 'LU', 'cosMLTU', 'sinMLTU', 'SYM/H','lat'], 
    # 5: ['LU', 'cosMLTU','sinMLTU', 'WSpeed','FlowPressure','lat', 'log'+e],    
    # 6: ['LU', 'cosMLTU','sinMLTU', 'WSpeed','FlowPressure','SYM/H','lat', 'log'+e],    
    # 7: ['WSpeed', 'SYM/H', 'lat', 'log'+e],    
    }

# Functions ####################################################################################
def to_supervised(data, n_past, n_future):
    X, y = [], []
    in_start = 0
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_past
        out_end = in_end + n_future
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0]) # taking the first column 
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)

def prepare_data(dfg, test_size=0.2, random_state=42):
    # Split data first to avoid leakage
    train_df, test_df = dfg[0:int(n*perce)], dfg[int(n*perce):]

    # Fit and transform the training data
    scalers = {}
    for col in train_df.columns:
        scalers[col] = StandardScaler()
        scalers[col] = scalers[col].fit(train_df[col].values.reshape(-1, 1))
        train_df[col] = scalers[col].transform(train_df[col].values.reshape(-1, 1))
        test_df[col] = scalers[col].transform(test_df[col].values.reshape(-1, 1))
    
    # Prepare supervised data
    X_train, y_train = to_supervised(np.array(train_df), n_past, n_future)
    X_test, y_test = to_supervised(np.array(test_df), n_past, n_future)

    return X_train, y_train, X_test, y_test, scalers
    
    


# Define the model-building function
def build_model(hp):
    model = Sequential()
    
    # Middle LSTM layers if more than one
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 3)
    for i in range(1, num_lstm_layers):
        model.add(LSTM(
            units=hp.Int(f'units_{i}', 2, 512, step=32),
            return_sequences=True  # Only the last LSTM layer should have False
        ))
        model.add(Dropout(0.2))

    
    model.add(LSTM(
        units=hp.Int('units_last_lstm_layers', 2, 512, step=32),
        return_sequences=False  # False to match the Dense layer input
    ))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(n_future, activation='linear'))  # Ensure `n_future` matches your target shape

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4]),
            clipvalue=1.0
        ),
        loss='mse',
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    # Callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    mc = ModelCheckpoint(dire+'best_model.h5', save_best_only=True)
    
    # Model training
    model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=batchsize,
        validation_split=0.3,
        callbacks=[es, mc]
    )
    
    return model

# Program begins here

""" Data was obtained by merging 1-min data from:
1) EPT: https://swe.ssa.esa.int/space-radiation
2) Indices + SW from omniweb : https://omniweb.gsfc.nasa.gov/form/omni_min.html
"""
df = pd.read_csv(pathin+'df_'+e+'_L'+str(L1)+'-'+str(L2)+'+ind_nolog.csv') # Read the data files provided 
df.index = pd.to_datetime(df.Time)
df.loc[(df['MLTU'] == 24.0), 'MLTU'] = 0.0 # I found one !
print('fluxes read')

# Re-process data
if tt=='1H': df=df.resample('1H').mean() 
elif tt=='1D': df=df.resample('d').mean() 
df = df.dropna()
df = df.loc[df[e]>=0] ; df['log'+e]=np.log10(df[e]+1) # to have only positive values

n = len(df)-n_past
k=list(input_group.keys())[0]
dire = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/LSTM/" +e
dfg = df[input_group[k]] 


X_train, y_train, X_test, y_test, scalers = prepare_data(dfg)

#%%

tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=15,
    factor=3,
    hyperband_iterations= 3,
    directory="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/LSTM/" + e,
    project_name='lstm_tuning'
)

# Setup TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/LSTM/" + e + "/log",
    update_freq='epoch'  # Logs metrics at the end of each epoch
)

# Setup EarlyStopping callback
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',    # Monitor the validation loss
    patience=3,            # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

tuner.search(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[tensorboard_callback, early_stopping_callback]
)

#%%
# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

#%%
print("Best model summary:" )
best_model.summary()

"""Best model e5 
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 48, 112)           53312     
                                                                 
 dropout (Dropout)           (None, 48, 112)           0         
                                                                 
 lstm_1 (LSTM)               (None, 12)                6000      
                                                                 
 dropout_1 (Dropout)         (None, 12)                0         
                                                                 
 dense (Dense)               (None, 12)                156       
                                                                 
=================================================================
Total params: 59,468
Trainable params: 59,468
Non-trainable params: 0
_________________________________________________________________

Value             |Best Value So Far |Hyperparameter
1                 |2                 |num_lstm_layers
10                |12                |units_last_lstm_layers
0.0001            |0.001             |learning_rate
32                |112               |units_1
80                |88                |units_2
15                |15                |tuner/epochs
0                 |5                 |tuner/initial_epoch
0                 |2                 |tuner/bracket
0                 |2                 |tuner/round

"""






















