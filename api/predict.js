import fetch from '@vercel/node-fetch';
import { MinMaxScaler } from 'sklearn/preprocessing';
import torch from 'torch';

// Загрузка модели
const model = new LSTMModel();
model.load_state_dict(torch.load('model.pth'));
model.eval();

async function load_data(ticker, start_date, end_date) {
    const response = await fetch(`https://query1.finance.yahoo.com/v7/finance/download/${ticker}?period1=${new Date(start_date).getTime() / 1000}&period2=${new Date(end_date).getTime() / 1000}&interval=1d&events=history`);
    if (!response.ok) {
        throw new Error(`No data found for ticker: ${ticker}`);
    }
    const text = await response.text();
    const rows = text.split('\n').slice(1, -1);
    return rows.map(row => parseFloat(row.split(',')[4]));
}

function create_sequences(data, seq_length) {
    const xs = [];
    const ys = [];
    for (let i = 0; i < data.length - seq_length; i++) {
        const x = data.slice(i, i + seq_length);
        const y = data[i + seq_length];
        xs.push(x);
        ys.push(y);
    }
    return [xs, ys];
}

function preprocess_data(data, seq_length) {
    const scaler = new MinMaxScaler({ feature_range: [-1, 1] });
    const scaled_data = scaler.fit_transform(data.map(d => [d])).map(d => d[0]);
    const [X, y] = create_sequences(scaled_data, seq_length);
    return [X, y, scaler];
}

export default async (req, res) => {
    const { ticker, start_date, end_date } = req.body;

    try {
        // Загрузка и предобработка данных
        const data = await load_data(ticker, start_date, end_date);
        const seq_length = 60;
        const [X, y, scaler] = preprocess_data(data, seq_length);

        if (X.length < seq_length) {
            throw new Error(`Not enough data for ticker ${ticker} to create sequences of length ${seq_length}.`);
        }

        const X_test = X.slice(-1);  // Используем последнюю последовательность для предсказания
        const X_test_tensor = torch.tensor(X_test, { dtype: torch.float32 }).unsqueeze(2);

        // Предсказание
        const prediction = model(X_test_tensor).squeeze();

        // Обратное преобразование предсказания
        const prediction_scaled = scaler.inverse_transform(prediction.numpy().reshape(-1, 1));
        res.status(200).json({ prediction: prediction_scaled[0][0] });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};