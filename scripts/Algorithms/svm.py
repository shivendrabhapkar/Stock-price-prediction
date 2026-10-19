#!/usr/bin/env python
'''
    Running Support Vector Regression Model.
'''
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut

def convert_to_integer(dt_time):
    return 10000 * dt_time.year + 1000 * dt_time.month + dt_time.day

def preprocess(file_dataframe, cols=['date', 'open']):
    if 'date' in cols:
        file_dataframe['date'] = file_dataframe['date'].map(convert_to_integer)
    X = file_dataframe['open']
    y = file_dataframe['date']
    return X, y

def svm(file_dataframe, test_size=0.2, col='open'):
    '''
        Run Support Vector Regression
    '''
    print('Loading data for SVM...')
    df = file_dataframe.copy()
    if 'date' in df:
        df['date_dt'] = pd.to_datetime(df['date'])
        df['date_num'] = df['date_dt'].apply(lambda d: 10000 * d.year + 100 * d.month + d.day)

    X = df[['open']].values
    y = df['adj_close'].values if 'adj_close' in df else df['close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')

    svr_lin.fit(X_train, y_train)
    svr_rbf.fit(X_train, y_train)

    lin_score = svr_lin.score(X_test, y_test)
    rbf_score = svr_rbf.score(X_test, y_test)

    print(f"SVR Linear R2 Score: {lin_score:.4f}")
    print(f"SVR RBF R2 Score:    {rbf_score:.4f}")

    return {'linear': lin_score, 'rbf': rbf_score}

def main(dir_path):
    '''
        Run SVM on CSV files in dir_path.
    '''
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    for file_name in files:
        print(f"\nProcessing {file_name} with SVM...")
        file_dataframe = pd.read_csv(os.path.join(dir_path, file_name))
        scores = svm(file_dataframe, 0.2)
        print("Scores:", scores)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python svm.py <input-dir>")
