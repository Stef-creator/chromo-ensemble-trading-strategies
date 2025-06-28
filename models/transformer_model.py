# transformer_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TransformerDataset(Dataset):
    def __init__(self, df, window_size, features=['RSI', 'Interval', 'Trend'], label_col='Label'):
        self.X = []
        self.y = []

        data = df[features].values.astype('float32')
        labels = df[label_col].values.astype('int64')

        for i in range(window_size, len(df)):
            window = data[i - window_size:i]
            self.X.append(window)
            self.y.append(labels[i])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, window_size, num_classes=3, dim_model=64, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.positional_encoding = nn.Parameter(torch.randn(window_size, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding  # [batch, seq_len, dim_model]
        x = x.permute(1, 0, 2)  # transformer expects [seq_len, batch, dim_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # global average pooling over sequence
        x = self.fc(x)
        return x

def train_transformer_model(df, window_size=10, epochs=100, batch_size=64, lr=0.001, model_save_path="trained_transformer.pth"):
    dataset = TransformerDataset(df, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(input_size=len(['RSI', 'Interval', 'Trend']), window_size=window_size)
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
    print(f"\nTransformer model saved as {model_save_path}")

    return model
