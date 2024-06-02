# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:29:23 2024

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    
    # Multiple Conv2D layers based on the tuner's decision
    for i in range(hp.Int('num_conv_layers', 1, 3)):  # Number of Conv2D layers
        model.add(Conv2D(
            filters=hp.Int(f'filters_{i}', 64, 512, step=64),
            kernel_size=(hp.Choice(f'kernel_rowsize_{i}', [1,2,3]), hp.Choice(f'kernel_colsize_{i}', [1,2,3,6])),
            activation='relu',
            padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.2))

    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten())

    # Adding Dense layers based on the tuner's decision
    for j in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            hp.Int(f'units_{j}', 64, 1024, step=64),
            activation='relu'
        ))
        model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(n_future))

    # Compile the model
    model.compile( 
        
        loss=tf.losses.MeanSquaredError(),
        optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4]),
            clipvalue=0.2
        ),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    # Early stopping and model checkpoint callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    mc = ModelCheckpoint(dire+'/best_model.h5', save_best_only=True)
    
    # Model training included within function for integration with Keras Tuner
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
dire = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN2d/" +e
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
    directory="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN2d/" + e,
    project_name='models_tuning'
)

# Setup TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN2d/" +e + "/log",
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


""" e5
Value             |Best Value So Far |Hyperparameter
3                 |1                 |num_conv_layers
192               |448               |filters_0
2                 |3                 |kernel_rowsize_0
1                 |3                 |kernel_colsize_0
2                 |1                 |num_dense_layers
1024              |640               |units_0
0.0001            |0.0001            |learning_rate
384               |64                |filters_1
2                 |3                 |kernel_rowsize_1
3                 |6                 |kernel_colsize_1
640               |192               |units_1
320               |128               |filters_2
1                 |1                 |kernel_rowsize_2
1                 |3                 |kernel_colsize_2
192               |832               |units_2
15                |15                |tuner/epochs
0                 |5                 |tuner/initial_epoch
0                 |2                 |tuner/bracket
0                 |2                 |tuner/round

Epoch 1/2
1180/1180 [==============================] - 51s 43ms/step - loss: 0.5070 - mean_absolute_error: 0.5403 - val_loss: 0.2841 - val_mean_absolute_error: 0.4171
Epoch 2/2
1180/1180 [==============================] - 51s 43ms/step - loss: 0.3473 - mean_absolute_error: 0.4469 - val_loss: 0.2491 - val_mean_absolute_error: 0.3798
Epoch 1/15
  5/759 [..............................] - ETA: 37s - loss: 0.2916 - mean_absolute_error: 0.4064WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0476s vs `on_train_batch_end` time: 0.0498s). Check your callbacks.
759/759 [==============================] - 39s 51ms/step - loss: 0.3008 - mean_absolute_error: 0.4114 - val_loss: 0.2137 - val_mean_absolute_error: 0.3596
Epoch 2/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2830 - mean_absolute_error: 0.3977 - val_loss: 0.2131 - val_mean_absolute_error: 0.3569
Epoch 3/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2696 - mean_absolute_error: 0.3868 - val_loss: 0.2094 - val_mean_absolute_error: 0.3529
Epoch 4/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2573 - mean_absolute_error: 0.3771 - val_loss: 0.2050 - val_mean_absolute_error: 0.3463
Epoch 5/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2482 - mean_absolute_error: 0.3698 - val_loss: 0.2177 - val_mean_absolute_error: 0.3621
Epoch 6/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2394 - mean_absolute_error: 0.3625 - val_loss: 0.2167 - val_mean_absolute_error: 0.3527
Epoch 7/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2320 - mean_absolute_error: 0.3567 - val_loss: 0.2035 - val_mean_absolute_error: 0.3479
Epoch 8/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2262 - mean_absolute_error: 0.3522 - val_loss: 0.2075 - val_mean_absolute_error: 0.3489
Epoch 9/15
759/759 [==============================] - 37s 49ms/step - loss: 0.2186 - mean_absolute_error: 0.3454 - val_loss: 0.2107 - val_mean_absolute_error: 0.3520
Epoch 10/15
759/759 [==============================] - 38s 50ms/step - loss: 0.2155 - mean_absolute_error: 0.3434 - val_loss: 0.2276 - val_mean_absolute_error: 0.3587
Trial 90 Complete [00h 08m 03s]
val_loss: 0.20354223251342773

Best val_loss So Far: 0.1677558422088623
Total elapsed time: 06h 26m 09s
Epoch 1/2
1180/1180 [==============================] - 61s 51ms/step - loss: 0.4098 - mean_absolute_error: 0.4780 - val_loss: 0.2242 - val_mean_absolute_error: 0.3600
Epoch 2/2
1180/1180 [==============================] - 60s 51ms/step - loss: 0.2950 - mean_absolute_error: 0.4094 - val_loss: 0.2174 - val_mean_absolute_error: 0.3466
Best model summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 1, 48, 448)        24640     
                                                                 
 batch_normalization (BatchN  (None, 1, 48, 448)       1792      
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 1, 24, 448)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 1, 24, 448)        0         
                                                                 
 flatten (Flatten)           (None, 10752)             0         
                                                                 
 dense (Dense)               (None, 640)               6881920   
                                                                 
 dropout_1 (Dropout)         (None, 640)               0         
                                                                 
 dense_1 (Dense)             (None, 12)                7692      
                                                                 
=================================================================
Total params: 6,916,044
Trainable params: 6,915,148
Non-trainable params: 896
_________________________________________________________________


"""















