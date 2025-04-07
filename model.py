import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out)
        return predictions[-1]

def train_model(model, train_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    from preprocessing import preprocess_data
    from data_loader import load_data
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    data = load_data(ticker, start_date, end_date)
    seq_length = 60
    X, y, scaler = preprocess_data(data, seq_length)
    X_train, y_train = X[:int(0.8 * len(X))], y[:int(0.8 * len(y))]
    X_test, y_test = X[int(0.8 * len(X)):], y[int(0.8 * len(y)):]
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(2),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = LSTMModel()
    train_model(model, train_loader)
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")