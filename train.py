from sklearn.model_selection import KFold 

import pandas as pd
import numpy as np

from tqdm import tqdm

def training(model, train:list, test, val_size, K):
    x_train, y_train = train

    if K:
        y_pred, val_results = cross_validation(K, model, [x_train, y_train], test)
    else:
        y_pred, val_results = one_fold(model, [x_train, y_train], test, val_size)

    return y_pred, val_results

def one_fold(model, train:list, test, val_size):
    # data
    x_train, y_train = train
    x_test = test

    # build model
    model.build()

    # training
    val_results = model.fit(X=x_train, y=y_train, validation_size=val_size)
    
    # prediction
    model.fit(X=x_train, y=y_train)
    y_pred = model.predict(X=x_test)

    return y_pred, val_results

def cross_validation(K, model, train:list, test):
    # data
    x_train, y_train = train
    x_test = test
    y_pred_mean = np.zeros(len(x_test))
    # set K-fold
    cv = KFold(n_splits=K, random_state=model.random_state, shuffle=True)
    val_results_lst = []
    for i, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train)):
        # split train and validation set
        x_train_i, y_train_i = x_train.iloc[train_idx], y_train.iloc[train_idx]
        x_val_i, y_val_i = x_train.iloc[val_idx], y_train.iloc[val_idx]

        # build model 
        model.build()

        # training
        val_result = model.fit(X=x_train_i, y=y_train_i,validation_set=[x_val_i,y_val_i])
        val_results_lst.append(val_result)

        # prediction
        y_pred = model.predict(X=x_test)
        y_pred_mean += y_pred / K

    # evaluation
    val_results = np.mean(val_results_lst)

    return y_pred_mean, val_results






