import yfinance as yf
import pandas as pd

def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data['Close'].values
    except Exception as e:
        raise ValueError(f"Error loading data for ticker {ticker}: {e}")

if __name__ == "__main__":
    ticker = 'AAPL'  # Пример: Apple Inc.
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    data = load_data(ticker, start_date, end_date)
    print(data)