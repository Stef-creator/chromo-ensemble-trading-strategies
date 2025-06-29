import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models.cnn_model import CNNClassifier, RollingWindowDataset
from utils import create_labeled_dataset, download_stock_data, calculate_RSI, calculate_SMA
import itertools

# === Load data ===

def tune_cnn_hyperparameters(ticker, window_size = 10):
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

    kernel_sizes = [2, 3, 5]
    hidden_channels = [8, 16, 32]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    window_size = 10
    epochs = 50

    results = []

    # === Grid search ===

    for ks, hc, lr, bs in itertools.product(kernel_sizes, hidden_channels, learning_rates, batch_sizes):
        print(f"Testing Kernel={ks}, Hidden Channels={hc}, LR={lr}, Batch Size={bs}")

        dataset = RollingWindowDataset(df_labeled, window_size)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        model = CNNClassifier(input_channels=3, window_size=window_size, kernel_size=ks, hidden_channels=hc)
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
            'kernel_size': ks,
            'hidden_channels': hc,
            'learning_rate': lr,
            'batch_size': bs,
            'avg_loss': avg_loss
        })

    # === Save tuning results ===

    results_df = pd.DataFrame(results)
    results_df.sort_values('avg_loss', inplace=True)

    print("\n=== CNN Tuning Results ===")
    print(results_df.to_string(index=False))

    results_df.to_csv('data/cnn_tuning_results.csv', index=False)
    return results_df

def best_cnn(ticker, window_size = 10, final_epochs=100):
    
    results_df = pd.read_csv('data/cnn_tuning_results.csv')
    best = results_df.iloc[0]
    print(f"\nRetraining final CNN with best config: Kernel={best['kernel_size']}, Hidden Channels={best['hidden_channels']}, LR={best['learning_rate']}, Batch Size={best['batch_size']}")

    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    df_labeled = create_labeled_dataset(df, best_chromosome)
    dataset = RollingWindowDataset(df_labeled, window_size)
    dataloader = DataLoader(dataset, batch_size=int(best['batch_size']), shuffle=True)

    final_model = CNNClassifier(
        input_channels=3,
        window_size=window_size,
        kernel_size=int(best['kernel_size']),
        hidden_channels=int(best['hidden_channels'])
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best['learning_rate'])

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
    torch.save(final_model.state_dict(), 'optimized_models/trained_cnn_best_tuned.pth')
    print("\nFinal tuned CNN model saved as optimized_models/trained_cnn_best_tuned.pth")
    return final_model


