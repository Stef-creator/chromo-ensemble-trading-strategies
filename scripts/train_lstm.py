# lstm_tuning_and_retraining.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models.lstm_model import LSTMClassifier, RollingWindowDataset
from utils import create_labeled_dataset, download_stock_data, calculate_RSI, calculate_SMA
import itertools

# === Load data ===

def tune_lstm_hyperparameters(ticker, window_size = 10):
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    df['Trend'] = (df['SMA_50'] > df['SMA_200']).astype(int)

    # Load best chromosome
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    df_labeled = create_labeled_dataset(df, best_chromosome)

    # === Define parameter grid ===

    hidden_sizes = [16, 32, 64]
    num_layers_list = [1, 2]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    epochs = 50

    results = []

    # === Grid search ===

    for hs, nl, lr, bs in itertools.product(hidden_sizes, num_layers_list, learning_rates, batch_sizes):
        print(f"Testing Hidden Size={hs}, Num Layers={nl}, LR={lr}, Batch Size={bs}")

        dataset = RollingWindowDataset(df_labeled, window_size)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        model = LSTMClassifier(input_size=3, hidden_size=hs, num_layers=nl)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
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
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        # Record final average loss
        avg_loss = total_loss / len(dataloader)
        results.append({
            'hidden_size': hs,
            'num_layers': nl,
            'learning_rate': lr,
            'batch_size': bs,
            'avg_loss': avg_loss
        })

    # === Save tuning results ===

    results_df = pd.DataFrame(results)
    results_df.sort_values('avg_loss', inplace=True)

    print("\n=== LSTM Tuning Results ===")
    print(results_df.to_string(index=False))

    results_df.to_csv('data/lstm_tuning_results.csv', index=False)

    return df_labeled, window_size, results_df

# === Step 2: Retrain final LSTM with best configuration ===

def best_lstm(ticker, window_size=10):
    results_df = pd.read_csv('data/lstm_tuning_results.csv')
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    ga_results = pd.read_csv('data/GA_final_results.csv')

    best = results_df.iloc[0]
    print(f"\nRetraining final LSTM with best config: Hidden Size={best['hidden_size']}, Num Layers={best['num_layers']}, LR={best['learning_rate']}, Batch Size={best['batch_size']}")

    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
            'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    df_labeled = create_labeled_dataset(df, best_chromosome)

    dataset = RollingWindowDataset(df_labeled, window_size)
    dataloader = DataLoader(dataset, batch_size=int(best['batch_size']), shuffle=True)

    final_model = LSTMClassifier(input_size=3, hidden_size=int(best['hidden_size']), num_layers=int(best['num_layers']))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best['learning_rate'])

    final_epochs = 100

    for epoch in range(final_epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[Final Retrain] Epoch [{epoch+1}/{final_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # === Save final retrained model ===

    torch.save(final_model.state_dict(), 'optimized_models/trained_lstm_best_tuned.pth')
    print("\nFinal tuned LSTM model saved as optimized_models/trained_lstm_best_tuned.pth")
    return final_model
