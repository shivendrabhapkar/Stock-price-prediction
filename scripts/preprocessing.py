#!/usr/bin/env python
"""
    Preprocessing and Feature Extraction for Stock Data.
    Adds: prev_day_diff, 50_day_moving_avg, 10_day_volatility
"""
import os
import sys
import pandas as pd
import numpy as np

def preprocess_dataframe(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Calculate prev_day_diff: difference with previous day's adj_close
    df['prev_day_diff'] = df['adj_close'].diff().fillna(0.0).round(6)
    
    # Calculate 50_day_moving_avg: rolling mean of adj_close
    df['50_day_moving_avg'] = df['adj_close'].rolling(window=50, min_periods=1).mean().round(6)
    
    # Calculate 10_day_volatility: rolling standard deviation of adj_close
    df['10_day_volatility'] = df['adj_close'].rolling(window=10, min_periods=1).std(ddof=0).fillna(0.0).round(6)
    
    cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'prev_day_diff', '50_day_moving_avg', '10_day_volatility']
    return df[cols]

def main(dir_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv') and f != 'sp.csv']

    for file_name in files:
        in_path = os.path.join(dir_path, file_name)
        df = pd.read_csv(in_path)
        processed_df = preprocess_dataframe(df)
        out_path = os.path.join(output_dir, file_name)
        processed_df.to_csv(out_path, index=False)
        print(f"Preprocessed {file_name}: {len(processed_df)} rows saved to {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <input-dir> <output-dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])