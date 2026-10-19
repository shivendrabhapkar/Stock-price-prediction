#!/usr/bin/env python
'''
    Data Normalization
'''
import pandas as pd
from sklearn import preprocessing

def normalize(file_dataframe, cols):
    '''
        Data Normalization using StandardScaler / MinMaxScaler.
    '''
    file_dataframe = file_dataframe.copy()
    for col in cols:
        if col in file_dataframe.columns:
            val = file_dataframe[[col]].values
            file_dataframe[col] = preprocessing.minmax_scale(val).flatten()

    return file_dataframe