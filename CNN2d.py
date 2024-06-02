# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:15:13 2024

@author: mathi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler # MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy.stats, os, json
from numpy import hstack
import matplotlib.dates as mdates
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



pd.options.mode.chained_assignment = None
#scale=MinMaxScaler(feature_range=(-1, 1))
tf.random.set_seed(0)

# Inputs 
L1=3 ; L2=8 ; e='e5' ; m = {'e1':[2,6],'e5':[0,3]} ; 
tt='1H' ; perce=0.9 ; n_past = 48  ; n_future = 12 ; 
MAX_EPOCHS= 50 ; batchsize=25 ; baseline=False
pathin= "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Initiation/part 2/"
path_out = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/models/CNN2d/" + 'L' + str(L1) + '-' + 'L' + str(L2) +' ' + e + '/'

""" variable to forecast is always the first one!!! because here the 'scale' is done for each column 
    one by one and it is the last one that is kept for the inverse_transform 
"""


input_group = {
    'physics': ['log'+e, 'LU', 'cosMLTU', 'sinMLTU', 'SYM/H','lat'], 
    'forward' : ['log'+e,'SYM/H','SYM/D','rad','LU','cosMLTU','sinMLTU','BU','kp_def','AE-index','ProtonTemperature','lat'],
    'lasso' : ['log'+e,'ProtonDensity', 'FlowPressure', 'WSpeed', 'AE-index','ProtonTemperature', 'lat', 'kp_def', 'sinMLTU'],
    'ridge' : ['log'+e,'ProtonDensity', 'FlowPressure', 'WSpeed', 'AE-index','ProtonTemperature', 'lat', 'kp_def'],
    'MI' :['log'+e,'BU', 'rad', 'lat', 'MLTU', 'Dst', 'cosMLTU', 'ProtonDensity','sinMLTU', 'WSpeed', 'SYM/H', 'FlowPressure', 'ProtonTemperature','AE-index', 'kp_def', 'AL-index'],
    'corr': ['log'+e,'SYM/H', 'Dst', 'rad', 'BU', 'LU', 'MLTU', 'cosMLTU', 'sinMLTU','kp_def', 'SYM/D', 'WSpeed', 'ProtonDensity', 'ProtonTemperature','FlowPressure', 'AL-index'],
    'base' : ['log'+e], 
    'all' : ['log'+e,'BU', 'rad', 'lat', 'MLTU', 'Dst', 'cosMLTU', 'ProtonDensity','sinMLTU', 'WSpeed', 'SYM/H', 'FlowPressure', 'ProtonTemperature','AE-index', 'kp_def', 'AL-index', 'SYM/D', 'LU']
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
    

def build_model(input_shape):
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Conv2D(filters=448, kernel_size=(2, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.2))  # Slightly reduce dropout

    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten())
    
    # Fully Connected Layer
    model.add(Dense(960, activation='relu'))
    model.add(Dropout(0.2))  # Slightly reduce dropout
    
    # Output Layer
    model.add(Dense(n_future))
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(clipvalue=1.0, learning_rate = 1e-4),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
    mc = tf.keras.callbacks.ModelCheckpoint(path_out+'best_model.h5')
    
    model.fit(X_train, y_train, epochs=MAX_EPOCHS,batch_size=batchsize,
                              validation_split=0.2, callbacks=[es,mc])
    return model

def make_forecasts(model, batchsize, X_test): # according to the model
    forecasts = model.predict(X_test, batch_size=batchsize)
    return forecasts


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
dfg = df[input_group['physics']] 



X_train, y_train, X_test, y_test, scalers = prepare_data(dfg)
# Reshape X to be compatible with Conv2D (samples, height=1, width=n_steps_in, channels=n_features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, n_past, len(input_group['physics'])))
X_test = np.reshape(X_test, (X_test.shape[0], 1, n_past, len(input_group['physics'])))


#%%

model_path = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN2d/e1/best_model.h5"
CNN2d_model = tf.keras.models.load_model(model_path)
CNN2d_model.summary()

#%%
# train model and save best
input_shape = (1, n_past, len(input_group['physics'])) # Adjust based on your dataset
CNN2d_model = build_model(input_shape) ; history = build_model

#%%
scale = scalers['log' + e]
# read saved model
model_path = "C:/Users/mathi/Documents/Université/Semestre 9/Mémoire/CODE/Propre/without leakage/Hyperparameter tuning/CNN2d/e1/best_model.h5"
CNN2d_model = tf.keras.models.load_model(model_path)
CNN2d_model.summary()
CNN2d_model = tf.keras.models.load_model(path_out+'best_model.h5')
forecasts = make_forecasts(CNN2d_model, batchsize, X_test)
forecasts = scale.inverse_transform(forecasts)
y_test = scale.inverse_transform(y_test)
dfpt = pd.DataFrame({'pred':forecasts.flatten(), 'test':y_test.flatten()})
# save forecasts and observations
#dfpt.to_csv(path_out+'PredTest.csv', sep=',')

#%%
df_final = dfpt

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
interval = 6
loc= mdates.HourLocator(interval=interval)
ax.plot(df_final.test,color='k', alpha = 1, label='Observations')
ax.plot(df_final.pred,color='red', alpha = 0.7, label='Prediction')
plt.xlim(df_final.index[0],df_final.index[-1])
plt.grid('both')
plt.legend(fontsize=14)
plt.title('Test Set Times serie',fontsize=16,fontweight='bold')
plt.ylabel('Log(' + str(e)+') [#/cm**2 s sr MeV]',fontsize=14,fontweight='bold')
plt.show()


#%%


def evaluation(actual_12_ahead, predicted_12_ahead):
    print('t, RMSE,  r, PE,  MSA, SSPB')
    for i in range(n_future):
        actual_values = [sublist[i] for sublist in actual_12_ahead]
        predicted_values = [sublist[i] for sublist in predicted_12_ahead]
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        
        # Calculate correlation coefficient (r)
        r_value, _ = scipy.stats.pearsonr(actual_values, predicted_values)
        
        actual = np.array(actual_values)
        predicted = np.array(predicted_values)
        
        # Calculate PE
        mean_actual = np.mean(actual)
        numerator = np.sum((actual - predicted) ** 2)
        denominator = np.sum((actual - mean_actual) ** 2)
        PE = 1 - (numerator / denominator)
    
        # Calculate Median Symetric Accuracy
        log_ratio = np.log(10**(predicted)/10**(actual)) 
        MSA = 100 * (np.exp(np.median(np.abs(log_ratio))) - 1)
        
        # Calculate Symmetric Signed Percentage Bias (SSPB)
        median_log_ratio = np.median(log_ratio)
        SSPB = 100 * np.sign(median_log_ratio) * (np.exp(np.abs(median_log_ratio)) - 1)
        print(str(i+1)+ ',' + str(rmse) +',' + str(r_value) + ',' + str(PE) + ',' +  str(MSA) + ','+  str(SSPB))

# Split DataFrame into chunks of 12 rows each
chunks = [df_final.iloc[i:i+n_future] for i in range(0, len(df_final), n_future)]

# Create lists for 'pred' and 'actual' columns
pred_lists = [chunk['pred'].tolist() for chunk in chunks]
actual_lists = [chunk['test'].tolist() for chunk in chunks]

evaluation(actual_lists, pred_lists)




































