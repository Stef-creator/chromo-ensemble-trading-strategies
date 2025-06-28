# cnn_model.py

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
            window = data[i - window_size:i].T  # shape: [features, window_size]
            self.X.append(window)
            self.y.append(labels[i])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class CNNClassifier(nn.Module):
    def __init__(self, input_channels, window_size, kernel_size=3, hidden_channels=32):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=64, kernel_size=kernel_size)
        conv_output_size = window_size - 2 * (kernel_size - 1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)  # 3 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn_model(df, window_size=10, epochs=100, batch_size=64, lr=0.001, model_save_path="trained_cnn.pth"):
    dataset = RollingWindowDataset(df, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNClassifier(input_channels=len(['RSI', 'Interval', 'Trend']), window_size=window_size)
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
    print(f"\nCNN model saved as {model_save_path}")

    return model
