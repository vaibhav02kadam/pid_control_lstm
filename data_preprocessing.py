import sys
sys.path.append(r'./PID_train_data.csv')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Glo_paras import *

def prepro(FILE_PATH):
    # IMPORT DATA
    data = pd.read_csv(FILE_PATH).drop('TIME',axis=1)
    
    
    # DATA GENERATOR
    data = np.array(data)
    X_data = []
    y_data = []
    for i in range(len(data)-WINDOW_SIZE):
        X = data[i:i+WINDOW_SIZE,:].flatten()
        y = data[i+WINDOW_SIZE,0]
        
        X_data.append(X)
        y_data.append(y)
    
    X_data = np.array(X_data)
    y_data = np.array(y_data).reshape((len(y_data),-1))
    
    # NORMALIZE
    scaler_x = StandardScaler()
    X_data = scaler_x.fit_transform(X_data)
    
    scaler_y = StandardScaler()
    y_data = scaler_y.fit_transform(y_data)
    
    # SPLIT DATA
    train_num = np.round(len(X_data)*TRAIN_RATIO).astype(int)
    val_num = np.round(len(X_data)*VAL_RATIO).astype(int)
    test_num = (len(X_data)-val_num-train_num).astype(int)
    
    # TRAIN SAMPLES
    X_train = X_data[:train_num,:].reshape((train_num,WINDOW_SIZE,data.shape[1]))
    y_train = y_data[:train_num].reshape((train_num,-1))
    
    # VAL SAMPLES
    X_val = X_data[train_num:train_num+val_num,:].reshape((val_num,WINDOW_SIZE,data.shape[1]))
    y_val = y_data[train_num:train_num+val_num].reshape((val_num,-1))
    
    # TEST SAMPLES
    X_test = X_data[train_num+val_num:,:].reshape((test_num,WINDOW_SIZE,data.shape[1]))
    y_test = y_data[train_num+val_num:].reshape((test_num,-1))
    
    return X_train,y_train,X_val,y_val, X_test,y_test,scaler_y

prepro(FILE_PATH)