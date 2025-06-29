# tune_mlp.py
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from models.mlp_model import MLPClassifier_dynamic
from utils import create_labeled_dataset, plot_equity_curve, backtest_strategy

class RSIDataset(Dataset):
    def __init__(self, df):
        self.X = df[['RSI', 'Interval', 'Trend']].values.astype('float32')
        self.y = df['Label'].values.astype('int64')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_and_evaluate(df, best_chromosome, hidden_sizes, lr, batch_size, epochs=100):
    labeled_df = create_labeled_dataset(df, best_chromosome)
    dataset = RSIDataset(labeled_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build dynamic model
    model = MLPClassifier_dynamic(hidden_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0
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

    return total_loss / len(dataloader)


def run_tuning_mlp(ticker):
    # Load data
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
        'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Hyperparameter grid
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [64, 128, 256]
    hidden_layer_configs = [
        [20, 10, 8, 6, 5],
        [30, 20, 10],
        [40, 20]
    ]

    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for hidden_sizes in hidden_layer_configs:
                print(f"\nTesting LR={lr}, Batch Size={batch_size}, Hidden Sizes={hidden_sizes}")
                avg_loss = train_and_evaluate(df, best_chromosome, hidden_sizes, lr, batch_size)
                results.append({
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'hidden_sizes': hidden_sizes,
                    'avg_loss': avg_loss
                })

    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df.sort_values('avg_loss', inplace=True)
    print("\n=== Hyperparameter Tuning Results ===")
    print(results_df)
    results_df.to_csv("optimized_models/mlp_tuning_results.csv", index=False)

def best_mlp(ticker):
    # Load data
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
        'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Prepare dataset
    batch_size_val = pd.to_numeric(best_params.loc[0, 'batch_size'], errors='coerce')
    if isinstance(batch_size_val, complex):
        batch_size_val = batch_size_val.real
    if pd.isnull(batch_size_val):
        raise ValueError("batch_size is not a valid number")
    batch_size = int(batch_size_val)

    lr_val = pd.to_numeric(best_params.loc[0, 'learning_rate'], errors='coerce')
    if isinstance(lr_val, complex):
        lr_val = lr_val.real
    if pd.isnull(lr_val):
        raise ValueError("learning_rate is not a valid number")
    lr = float(lr_val)

    hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
    hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
    labeled_df = create_labeled_dataset(df, best_chromosome)
    dataset = RSIDataset(labeled_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build final optimal model
    model = MLPClassifier_dynamic(hidden_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epochs = 200
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

    # Save trained model
    torch.save(model.state_dict(), "optimized_models/trained_mlp_final.pth")
    print("\nModel saved as trained_mlp_final.pth")

    return None