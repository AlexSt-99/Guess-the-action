import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    X, y = create_sequences(scaled_data, seq_length)
    return X, y, scaler

if __name__ == "__main__":
    from data_loader import load_data
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    data = load_data(ticker, start_date, end_date)
    seq_length = 60
    X, y, scaler = preprocess_data(data, seq_length)
    print(X.shape, y.shape)