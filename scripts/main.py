#!/usr/bin/env python
'''
    Main Preprocessing Pipeline File.
'''
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from interpolation import interpolate
from normalization import normalize

def main(dir_path, output_dir):
    '''
        Run Pipeline of interpolation and normalization on CSV files.
    '''
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

    for file_name in files:
        file_dataframe = pd.read_csv(os.path.join(dir_path, file_name))
        cols = ['high', 'open', 'low', 'close', 'volume', 'adj_close']
        file_dataframe = interpolate(file_dataframe, cols)
        # Optional normalization
        out_path = os.path.join(output_dir, file_name)
        file_dataframe.to_csv(out_path, index=False, encoding='utf-8')
        print(f"Cleaned and saved: {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python main.py <input-dir> <output-dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
