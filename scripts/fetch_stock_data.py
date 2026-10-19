#!/usr/bin/env python
"""
    Fetch historical stock data using yfinance.
"""
import sys
import os
import yfinance as yf
import pandas as pd

TICKER_MAP = {
    'FB': 'META',
}

def fetch_symbol_data(symbol, start_date='2012-05-18', end_date='2016-03-09'):
    ticker = TICKER_MAP.get(symbol, symbol)
    print(f"Fetching data for {symbol} (ticker: {ticker}) from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    }, inplace=True)
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    df['symbol'] = symbol
    return df[cols]

def fetch_sp_index(start_date='2012-05-18', end_date='2016-03-09'):
    print(f"Fetching S&P 500 index (^GSPC) from {start_date} to {end_date}...")
    df = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df.rename(columns={
        'Date': 'date',
        'Open': 's&p_index_open',
        'High': 's&p_index_high',
        'Low': 's&p_index_low',
        'Close': 's&p_index_close',
        'Volume': 's&p_index_volume',
        'Adj Close': 's&p_index_adj_close'
    }, inplace=True)
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    cols = ['date', 's&p_index_open', 's&p_index_high', 's&p_index_low', 's&p_index_close', 's&p_index_volume', 's&p_index_adj_close']
    return df[cols]

def main(symbol_file, output_dir, start_date='2012-05-18', end_date='2016-03-09'):
    os.makedirs(output_dir, exist_ok=True)
    
    symbols = []
    if os.path.exists(symbol_file):
        with open(symbol_file, 'r') as fp:
            for line in fp:
                s = line.strip()
                if s:
                    symbols.append(s)
    else:
        symbols = ['FB', 'GOOG']

    for sym in symbols:
        try:
            df = fetch_symbol_data(sym, start_date, end_date)
            out_path = os.path.join(output_dir, f"{sym}.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved {sym} data ({len(df)} rows) to: {out_path}")
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

    try:
        sp_df = fetch_sp_index(start_date, end_date)
        sp_path = os.path.join(output_dir, "sp.csv")
        sp_df.to_csv(sp_path, index=False)
        print(f"Saved S&P 500 data ({len(sp_df)} rows) to: {sp_path}")
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1], 'output_data')
    else:
        main('input/symbols.txt', 'raw_data')
