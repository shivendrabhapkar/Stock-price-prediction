#!/usr/bin/env python
'''
    Data Interpolation
'''
import os
import sys
import pandas as pd

def interpolate(dataframe, cols_to_interpolate):
    dataframe = dataframe.copy()
    for col in cols_to_interpolate:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].interpolate(method='linear').bfill().ffill()
    return dataframe

def main(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    for file_name in files:
        dataframe = pd.read_csv(os.path.join(dir_path, file_name))
        dataframe = interpolate(dataframe, ['high', 'open', 'low', 'close', 'volume', 'adj_close'])
        print(dataframe.head())
        break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
