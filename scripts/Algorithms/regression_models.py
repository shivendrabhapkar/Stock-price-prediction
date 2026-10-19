# -*- coding: utf-8 -*-
"""
    Run Stock-Regression Algorithms
"""
from __future__ import print_function
import sys
import os
import pickle
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regression_helpers import (
    load_dataset, addFeatures, mergeDataframes, count_missing,
    applyTimeLag, performRegression
)

def main(dir_path, output_dir):
    '''
        Run Pipeline of processes on file one by one.
    '''
    os.makedirs(output_dir, exist_ok=True)
    scores = {}

    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    maxdelta = 30
    delta = list(range(8, maxdelta))
    print('Delta days accounted: ', max(delta))

    for file_name in files:
        try:
            symbol = file_name.split('.')[0]
            print(f"\n==================== Symbol: {symbol} ====================")
            
            datasets = load_dataset(dir_path, file_name)

            for i, dataset in enumerate(datasets):
                columns = dataset.columns
                adjclose = columns[-2]
                returns = columns[-1]
                for dele in delta:
                    addFeatures(dataset, adjclose, returns, dele)
                datasets[i] = dataset.iloc[max(delta):, :].copy()

            finance = mergeDataframes(datasets)

            high_value = 365
            high_value = min(high_value, max(1, finance.shape[0] // 3))
            print('Maximum time lag applied: ', high_value)

            if 'symbol' in finance.columns:
                finance.drop('symbol', axis=1, inplace=True)

            print('Size of data frame: ', finance.shape)
            print('Number of NaN after merging: ', count_missing(finance))

            finance = finance.interpolate(method='time')
            print('Number of NaN after time interpolation: ', count_missing(finance))

            finance = finance.fillna(finance.mean(numeric_only=True))
            print('Number of NaN after mean interpolation: ', count_missing(finance))

            finance.columns = [str(col).replace('&', '_and_') for col in finance.columns]

            # Move the Open Values behind by one step to predict future price
            finance['open'] = finance['open'].shift(-1)

            finance = applyTimeLag(finance, [high_value], delta)

            print('Number of NaN after temporal shifting: ', count_missing(finance))
            print('Size of data frame after feature creation: ', finance.shape)

            mean_squared_errors, r2_scores = performRegression(finance, 0.95, symbol, output_dir)

            scores[symbol] = [mean_squared_errors, r2_scores]
        except Exception as e:
            traceback.print_exc()
    
    with open(os.path.join(output_dir, 'scores.pickle'), 'wb') as handle:
        pickle.dump(scores, handle)
    print(f"\nAll models trained and evaluated. Artifacts saved in: {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python regression_models.py <input-dir> <output-dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
