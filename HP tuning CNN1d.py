# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:34:30 2024

@author: mathi
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras_tuner.tuners import RandomSearch, Hyperband
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import scipy.stats, os, json
from numpy import hstack
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf


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
    
    # Conv1D layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):  # Number of Conv1D layers
        model.add(Conv1D(
            filters=hp.Int(f'filters_{i}', 32, 256, step=32),
            kernel_size=hp.Int(f'kernel_size_{i}', 2, 4),
            activation='relu'
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.2))
    
    model.add(Flatten())

    # Dense layers
    for j in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_{j}', 64, 512, step=64),
            activation='relu'
        ))
    
    model.add(Dense(n_future))  # Output layer
    model.add(Dropout(0.2))

    # Compile model
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3,1e-4]),
            clipvalue= 2.0
        ),
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
dire = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN1d/" +e
dfg = df[input_group[k]] 


X_train, y_train, X_test, y_test, scalers = prepare_data(dfg)
# Reshape X to be compatible with Conv2D (samples, height=1, width=n_steps_in, channels=n_features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, n_past, len(input_group[k])))
X_test = np.reshape(X_test, (X_test.shape[0], 1, n_past, len(input_group[k])))
#%%

tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=15,
    factor=3,
    hyperband_iterations= 3,
    directory="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN1d/" + e ,
    project_name='models_tuning'
)

# Setup TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN1d/" + e + '/log',
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
print("Best model summary:" )
best_model.summary()

""" Best model e5
Layer (type)                Output Shape              Param #      learning rate 1e-3
=================================================================
 conv1d (Conv1D)             (None, 1, 46, 192)        3648       kernel size = 3   
                                                                 
 batch_normalization (BatchN  (None, 1, 46, 192)       768       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 1, 23, 192)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 1, 23, 192)        0         
                                                                 
 flatten (Flatten)           (None, 4416)              0         
                                                                 
 dense (Dense)               (None, 128)               565376    
                                                                 
 dense_1 (Dense)             (None, 192)               24768     
                                                                 
 dense_2 (Dense)             (None, 12)                2316      
                                                                 
 dropout_1 (Dropout)         (None, 12)                0         
                                                                 
=================================================================
Total params: 596,876
Trainable params: 596,492
Non-trainable params: 384
_________________________________________________________________
"""


""" Best model e1
________________________________________________________________
 Layer (type)                Output Shape              Param #    learning rate 1e-4
=================================================================
 conv1d (Conv1D)             (None, 1, 46, 96)         1824      kernel size = 3  
                                                                 
 batch_normalization (BatchN  (None, 1, 46, 96)        384       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 1, 23, 96)        0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 1, 23, 96)         0         
                                                                 
 conv1d_1 (Conv1D)           (None, 1, 22, 224)        43232     kernel size = 2
                                                                 
 batch_normalization_1 (Batc  (None, 1, 22, 224)       896       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 1, 11, 224)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 1, 11, 224)        0         
                                                                 
 flatten (Flatten)           (None, 2464)              0         
                                                                 
 dense (Dense)               (None, 320)               788800    
                                                                 
 dense_1 (Dense)             (None, 384)               123264    
                                                                 
 dense_2 (Dense)             (None, 192)               73920     
                                                                 
 dense_3 (Dense)             (None, 12)                2316      
                                                                 
 dropout_2 (Dropout)         (None, 12)                0         
                                                                 
=================================================================
Total params: 1,034,636
Trainable params: 1,033,996
Non-trainable params: 640
_________________________________________________________________
"""



















