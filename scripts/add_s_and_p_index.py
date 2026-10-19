#!/usr/bin/env python
"""
    Merge stock dataframe with S&P 500 index indicators on date.
"""
import os
import sys
import pandas as pd

def main(dir_path, sp_index_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sp_df = pd.read_csv(sp_index_file)
    sp_df['date'] = pd.to_datetime(sp_df['date']).dt.strftime('%Y-%m-%d')
    
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv') and f != os.path.basename(sp_index_file)]

    for file_name in files:
        stock_path = os.path.join(dir_path, file_name)
        stock_df = pd.read_csv(stock_path)
        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.strftime('%Y-%m-%d')

        merged = pd.merge(stock_df, sp_df, on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)

        out_path = os.path.join(output_dir, file_name)
        merged.to_csv(out_path, index=False)
        print(f"Merged {file_name} with S&P 500: {len(merged)} rows -> {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python add_s_and_p_index.py <stock-dir> <sp-index-file> <output-dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])