# tune_transformer.py

import torch
import pandas as pd
from models.transformer_model import TransformerClassifier, TransformerDataset
from utils import create_labeled_dataset, backtest_strategy_with_position_stoploss
from torch.utils.data import DataLoader


# Load data
def tune_transformer(ticker, window_size = 10):
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

    # Load best chromosome
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Create labeled dataset
    df_labeled = create_labeled_dataset(df, best_chromosome)

    dim_models = [64, 128]
    num_heads_list = [4, 8]
    num_layers_list = [1, 2]
    learning_rates = [0.001]
    batch_sizes = [64]
    epochs = 5

    # Prepare results storage
    tuning_results = []

    # Loop over grid combinations
    for dim_model in dim_models:
        for num_heads in num_heads_list:
            for num_layers in num_layers_list:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        print(f"\nTraining Transformer: dim_model={dim_model}, num_heads={num_heads}, num_layers={num_layers}, lr={lr}, batch_size={batch_size}")

                        # Prepare dataset and dataloader
                        dataset = TransformerDataset(df_labeled, window_size)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        # Initialize model
                        model = TransformerClassifier(
                            input_size=3,
                            window_size=window_size,
                            num_classes=3,
                            dim_model=dim_model,
                            num_heads=num_heads,
                            num_layers=num_layers
                        )
                        criterion = torch.nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        # Train model
                        epochs = 30
                        model.train()
                        for epoch in range(epochs):
                            total_loss = 0
                            for X_batch, y_batch in dataloader:
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs, y_batch)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                            if (epoch + 1) % 10 == 0 or epoch == 0:
                                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

                        # Evaluate model on full dataset
                        model.eval()
                        X_test = torch.stack([sample[0] for sample in dataset])
                        with torch.no_grad():
                            outputs = model(X_test)
                            _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
                            predictions = predicted.numpy()

                        # Backtest with risk management
                        results, portfolio, trade_returns, executed_trades = backtest_strategy_with_position_stoploss(
                            df_labeled.iloc[window_size:], predictions,
                            position_size=0.25, stop_loss=0.03, take_profit=0.10,
                            return_executed_trades=False
                        )

                        # Store results
                        tuning_results.append({
                            'dim_model': dim_model,
                            'num_heads': num_heads,
                            'num_layers': num_layers,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'total_return': results['total_return'],
                            'annualized_return': results['annualized_return'],
                            'sharpe_ratio': results['sharpe_ratio'],
                            'max_drawdown': results['max_drawdown'],
                            'num_trades': len(trade_returns)
                        })

    # Convert to DataFrame
    tuning_df = pd.DataFrame(tuning_results)

    # Sort by Sharpe ratio
    tuning_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)

    # Display results
    print("\n=== Transformer Tuning Results ===")
    print(tuning_df.to_string(index=False))

    # Save results
    tuning_df.to_csv("data/transformer_tuning_results.csv", index=False)

    return tuning_df


# Prepare dataset
def best_transformer(ticker, window_size=10, epochs=200):
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

    # Load best chromosome
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
            'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Initialize best model configuration
    best_params = pd.read_csv('data/transformer_tuning_results.csv')
    model = TransformerClassifier(
        input_size=3,
        window_size=window_size,
        num_classes=3,
        dim_model=int(best_params.iloc[0]['dim_model']),
        num_heads=int(best_params.iloc[0]['num_heads']),
        num_layers=int(best_params.iloc[0]['num_layers'])
    )
    batch_size = int(best_params.iloc[0]['batch_size'])
    lr = float(best_params.iloc[0]['learning_rate'])

    # Create labeled dataset
    df_labeled = create_labeled_dataset(df, best_chromosome)
    dataset = TransformerDataset(df_labeled, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train for specified epochs
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

    # Save model
    torch.save(model.state_dict(), "optimized_models/trained_transformer_best_config.pth")
    print("\nTransformer retrained and saved as trained_transformer_best_config.pth")

    return None