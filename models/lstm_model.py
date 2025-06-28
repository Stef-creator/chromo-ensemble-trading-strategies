# lstm_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RollingWindowDataset(Dataset):
    def __init__(self, df, window_size, features=['RSI', 'Interval', 'Trend'], label_col='Label'):
        self.X = []
        self.y = []

        data = df[features].values.astype('float32')
        labels = df[label_col].values.astype('int64')

        for i in range(window_size, len(df)):
            window = data[i - window_size:i]  # shape: [window_size, features]
            self.X.append(window)
            self.y.append(labels[i])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = hn[-1]  # take the last hidden state
        out = self.fc(out)
        return out

def train_lstm_model(df, window_size=10, epochs=100, batch_size=64, lr=0.001, model_save_path="trained_lstm.pth"):
    dataset = RollingWindowDataset(df, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(input_size=len(['RSI', 'Interval', 'Trend']))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"\nLSTM model saved as {model_save_path}")

    return model
