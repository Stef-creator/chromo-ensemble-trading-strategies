# ensemble_with_transformer.py

import pandas as pd
import numpy as np
import ast
import torch
from datetime import datetime
from models.mlp_model import MLPClassifier_dynamic
from models.cnn_model import CNNClassifier, RollingWindowDataset as CNNRollingDataset
from models.lstm_model import LSTMClassifier, RollingWindowDataset as LSTMRollingDataset
from models.transformer_model import TransformerClassifier, TransformerDataset
from utils import create_labeled_dataset, backtest_strategy_with_position_stoploss, plot_prediction_signals, download_stock_data, calculate_RSI, calculate_SMA, weighted_ensemble_vote, calculate_additional_metrics, backtest_strategy_with_kelly_position, plot_prediction_signals_kelly, plot_prediction_signals_dynamic

import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_ensemble_backtest(
    ticker,
    start_date="2017-02-01",
    end_date=None,
    window_size=10,
    position_size: float = 1, 
    stop_loss: float = 1, 
    take_profit: float = 1,
    mlp_model_path='optimized_models/trained_mlp_final.pth',
    cnn_model_path='optimized_models/trained_cnn_best_tuned.pth',
    lstm_model_path='optimized_models/trained_lstm_best_tuned.pth',
    transformer_model_path='optimized_models/trained_transformer_best_config.pth',
    plot_path="plots/ensemble_weighted_voting_prediction_signals.png"
):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Load data (true out-of-sample)
    df_oos = download_stock_data(ticker, start_date, end_date)
    if df_oos is None or df_oos.empty:
        raise ValueError("No data downloaded for out-of-sample period.")
    df_oos = calculate_RSI(df_oos)
    df_oos = calculate_SMA(df_oos)
    df_oos['Trend'] = (df_oos['SMA_50'] > df_oos['SMA_200']).astype(int)
    

    # Load best chromosome
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Create labeled dataset
    df_labeled = create_labeled_dataset(df_oos, best_chromosome)

    # === Generate predictions for each model ===

    # MLP predictions
    X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))
    best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
    hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
    hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
    model_mlp = MLPClassifier_dynamic(hidden_sizes)
    model_mlp.load_state_dict(torch.load(mlp_model_path))
    model_mlp.eval()
    with torch.no_grad():
        outputs_mlp = model_mlp(X_mlp)
        _, preds_mlp = torch.max(torch.nn.functional.softmax(outputs_mlp, dim=1), 1)
        predictions_mlp = preds_mlp.numpy()

    # CNN predictions
    cnn_dataset = CNNRollingDataset(df_labeled, window_size)
    X_cnn = torch.stack([sample[0] for sample in cnn_dataset])
    cnn_results = pd.read_csv('data/cnn_tuning_results.csv')
    best_cnn = cnn_results.iloc[0]
    best_kernel = int(best_cnn['kernel_size'])
    best_hidden_channels = int(best_cnn['hidden_channels'])

    model_cnn = CNNClassifier(
        input_channels=3,
        window_size=window_size,
        kernel_size=best_kernel,
        hidden_channels=best_hidden_channels
    )
    model_cnn.load_state_dict(torch.load(cnn_model_path))
    model_cnn.eval()
    with torch.no_grad():
        outputs_cnn = model_cnn(X_cnn)
        _, preds_cnn = torch.max(torch.nn.functional.softmax(outputs_cnn, dim=1), 1)
        predictions_cnn = preds_cnn.numpy()

    # LSTM predictions
    X_lstm = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32')).unsqueeze(1)
    lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
    best_lstm = lstm_results.iloc[0]
    best_hidden_size = int(best_lstm['hidden_size'])
    best_num_layers = int(best_lstm['num_layers'])

    model_lstm = LSTMClassifier(input_size=3, hidden_size=best_hidden_size, num_layers=best_num_layers)
    model_lstm.load_state_dict(torch.load(lstm_model_path))
    model_lstm.eval()
    with torch.no_grad():
        outputs_lstm = model_lstm(X_lstm)
        _, preds_lstm = torch.max(torch.nn.functional.softmax(outputs_lstm, dim=1), 1)
        predictions_lstm = preds_lstm.numpy()

    # Transformer predictions
    X_transformer = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32')).unsqueeze(1)
    transformer_results = pd.read_csv('data/transformer_tuning_results.csv')
    best_transformer = transformer_results.iloc[0]
    best_dim_model = int(best_transformer['dim_model'])
    best_num_heads = int(best_transformer['num_heads'])
    best_num_layers = int(best_transformer['num_layers'])

    model_transformer = TransformerClassifier(
        input_size=3,
        window_size=window_size,
        dim_model=best_dim_model,
        num_heads=best_num_heads,
        num_layers=best_num_layers
    )
    model_transformer.load_state_dict(torch.load(transformer_model_path))
    model_transformer.eval()
    with torch.no_grad():
        outputs_transformer = model_transformer(X_transformer)
        _, preds_transformer = torch.max(torch.nn.functional.softmax(outputs_transformer, dim=1), 1)
        predictions_transformer = preds_transformer.numpy()

    # === Align prediction lengths ===
    min_len = min(len(predictions_mlp), len(predictions_cnn), len(predictions_lstm), len(predictions_transformer))
    predictions_mlp = predictions_mlp[-min_len:]
    predictions_cnn = predictions_cnn[-min_len:]
    predictions_lstm = predictions_lstm[-min_len:]
    predictions_transformer = predictions_transformer[-min_len:]
    df_eval = df_labeled.iloc[-min_len:]

    # === Ensemble majority voting ===
    weights = [0.30, 0.20, 0.20, 0.20] 
    ensemble_predictions = []
    for p1, p2, p3, p4 in zip(predictions_mlp, predictions_cnn, predictions_lstm, predictions_transformer):
        votes = [p1, p2, p3, p4]
        final_decision = weighted_ensemble_vote(votes, weights)
        ensemble_predictions.append(final_decision)

    print("Prediction counts:", np.unique(ensemble_predictions, return_counts=True))

    # === Backtest ensemble ===
    results, portfolio, trade_returns, executed_trades = backtest_strategy_with_position_stoploss(
        df_eval, ensemble_predictions,
        position_size=position_size, stop_loss=stop_loss, take_profit=take_profit,
        return_executed_trades=True
    )

    # Print results
    print("\n=== Ensemble (MLP+CNN+LSTM+Transformer) Backtest Performance ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    metrics = calculate_additional_metrics(portfolio, trade_returns)

    print("\n=== Additional Trading Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot signals
    plot_prediction_signals(df_eval, ensemble_predictions, title="Ensemble Prediction Signals", save_path=plot_path)

    if any(x is None for x in [results, portfolio, trade_returns, ensemble_predictions, df_eval]):
        raise ValueError("One or more outputs from run_ensemble_backtest are None. Check your data and model paths.")
    
    return results, portfolio, trade_returns, ensemble_predictions, df_eval

def run_probabilistic_ensemble_backtest(
    ticker,
    start_date="2017-02-01",
    end_date=None,
    window_size=10,
    position_size : float = 0.25, 
    stop_loss: float= 0.03, 
    take_profit: float = 0.10,
    mlp_model_path='optimized_models/trained_mlp_final.pth',
    cnn_model_path='optimized_models/trained_cnn_best_tuned.pth',
    lstm_model_path='optimized_models/trained_lstm_best_tuned.pth',
    transformer_model_path='optimized_models/trained_transformer_best_config.pth',
):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Load data (true out-of-sample)
    df_oos = download_stock_data(ticker, start_date, end_date)
    if df_oos is None or df_oos.empty:
        raise ValueError("No data downloaded for out-of-sample period.")
    df_oos = calculate_RSI(df_oos)
    df_oos = calculate_SMA(df_oos)
    df_oos['Trend'] = (df_oos['SMA_50'] > df_oos['SMA_200']).astype(int)

    # Load best chromosome
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # Create labeled dataset
    df_labeled = create_labeled_dataset(df_oos, best_chromosome)

    # === Generate softmax probabilities for each model ===

    # MLP probabilities
    X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))
    best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
    hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
    hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
    model_mlp = MLPClassifier_dynamic(hidden_sizes)
    model_mlp.load_state_dict(torch.load(mlp_model_path))
    model_mlp.eval()
    with torch.no_grad():
        outputs_mlp = model_mlp(X_mlp)
        probs_mlp = torch.nn.functional.softmax(outputs_mlp, dim=1).numpy()

    # CNN probabilities
    cnn_results = pd.read_csv('data/cnn_tuning_results.csv')
    best_cnn = cnn_results.iloc[0]
    best_kernel = int(best_cnn['kernel_size'])
    best_hidden_channels = int(best_cnn['hidden_channels'])

    model_cnn = CNNClassifier(
        input_channels=3,
        window_size=window_size,
        kernel_size=best_kernel,
        hidden_channels=best_hidden_channels
    )
    cnn_dataset = CNNRollingDataset(df_labeled, window_size)
    X_cnn = torch.stack([sample[0] for sample in cnn_dataset])
    model_cnn.load_state_dict(torch.load(cnn_model_path))
    model_cnn.eval()
    with torch.no_grad():
        outputs_cnn = model_cnn(X_cnn)
        probs_cnn = torch.nn.functional.softmax(outputs_cnn, dim=1).numpy()

    # LSTM probabilities
    lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
    best_lstm = lstm_results.iloc[0]
    best_hidden_size = int(best_lstm['hidden_size'])
    best_num_layers = int(best_lstm['num_layers'])
    lstm_dataset = LSTMRollingDataset(df_labeled, window_size)
    X_lstm = torch.stack([sample[0] for sample in lstm_dataset])

    model_lstm = LSTMClassifier(
        input_size=3,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers
    )
    model_lstm.load_state_dict(torch.load(lstm_model_path))
    model_lstm.eval()
    with torch.no_grad():
        outputs_lstm = model_lstm(X_lstm)
        probs_lstm = torch.nn.functional.softmax(outputs_lstm, dim=1).numpy()

    # Transformer probabilities
    transformer_results = pd.read_csv('data/transformer_tuning_results.csv')
    best_transformer = transformer_results.iloc[0]
    best_dim_model = int(best_transformer['dim_model'])
    best_num_heads = int(best_transformer['num_heads'])
    best_num_layers = int(best_transformer['num_layers'])
    transformer_dataset = TransformerDataset(df_labeled, window_size)
    X_transformer = torch.stack([sample[0] for sample in transformer_dataset])

    model_transformer = TransformerClassifier(
        input_size=3,
        window_size=window_size,
        dim_model=best_dim_model,
        num_heads=best_num_heads,
        num_layers=best_num_layers
    )
    model_transformer.load_state_dict(torch.load(transformer_model_path))
    model_transformer.eval()
    with torch.no_grad():
        outputs_transformer = model_transformer(X_transformer)
        probs_transformer = torch.nn.functional.softmax(outputs_transformer, dim=1).numpy()

    # === Align probability array lengths ===
    min_len = min(len(probs_mlp), len(probs_cnn), len(probs_lstm), len(probs_transformer))
    probs_mlp = probs_mlp[-min_len:]
    probs_cnn = probs_cnn[-min_len:]
    probs_lstm = probs_lstm[-min_len:]
    probs_transformer = probs_transformer[-min_len:]
    df_eval = df_labeled.iloc[-min_len:]

    # === Probabilistic voting ensemble ===
    ensemble_predictions = []
    for p_mlp, p_cnn, p_lstm, p_trans in zip(probs_mlp, probs_cnn, probs_lstm, probs_transformer):
        all_probs = np.vstack([p_mlp, p_cnn, p_lstm, p_trans])
        avg_probs = np.mean(all_probs, axis=0)
        final_decision = np.argmax(avg_probs)
        ensemble_predictions.append(final_decision)

    print("Prediction counts:", np.unique(ensemble_predictions, return_counts=True))

    # === Backtest ensemble ===
    results, portfolio, trade_returns, executed_trades = backtest_strategy_with_position_stoploss(
        df_eval, ensemble_predictions,
        position_size = position_size, stop_loss = stop_loss, take_profit=take_profit,
        return_executed_trades=True
    )

    # Print results
    print("\n=== Ensemble (Probabilistic Voting) Backtest Performance ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    metrics = calculate_additional_metrics(portfolio, trade_returns)

    print("\n=== Additional Trading Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot signals
    plot_path=f"plots/ensemble_probabilistic_voting_signals_positionsize_{position_size}_sl_{stop_loss}.png"
    plot_prediction_signals_dynamic(df_eval, executed_trades, title="Prediction Signals with PnL", save_path=plot_path, initial_capital=10000)


    if any(x is None for x in [results, portfolio, trade_returns, ensemble_predictions, df_eval]):
        raise ValueError("One or more outputs from run_probabilistic_ensemble_backtest are None. Check your data and model paths.")

    return results, portfolio, trade_returns, ensemble_predictions, df_eval

def run_probabilistic_ensemble_backtest_with_kelly(
    ticker,
    start_date="2017-02-01",
    end_date=None,
    window_size=10,
    stop_loss: float = 0.5,
    take_profit: float = 0.5,
    mlp_model_path='optimized_models/trained_mlp_final.pth',
    cnn_model_path='optimized_models/trained_cnn_best_tuned.pth',
    lstm_model_path='optimized_models/trained_lstm_best_tuned.pth',
    transformer_model_path='optimized_models/trained_transformer_best_config.pth',
    plot_path="plots/ensemble_probabilistic_voting_signals_kelly.png"
):
    import torch
    import pandas as pd
    import numpy as np
    from datetime import datetime

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # === Load data (true out-of-sample) ===
    df_oos = download_stock_data(ticker, start_date, end_date)
    if df_oos is None or df_oos.empty:
        raise ValueError("No data downloaded for out-of-sample period.")
    df_oos = calculate_RSI(df_oos)
    df_oos = calculate_SMA(df_oos)
    df_oos['Trend'] = (df_oos['SMA_50'] > df_oos['SMA_200']).astype(int)

    # === Load best chromosome ===
    ga_results = pd.read_csv('data/GA_final_results.csv')
    best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
        ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
         'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
    ].values.tolist()

    # === Create labeled dataset ===
    df_labeled = create_labeled_dataset(df_oos, best_chromosome)

    # === Generate MLP probabilities ===
    X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))
    best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
    hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
    hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
    model_mlp = MLPClassifier_dynamic(hidden_sizes)
    model_mlp.load_state_dict(torch.load(mlp_model_path))
    model_mlp.eval()
    with torch.no_grad():
        outputs_mlp = model_mlp(X_mlp)
        probs_mlp = torch.nn.functional.softmax(outputs_mlp, dim=1).numpy()

    # === Generate CNN probabilities ===
    cnn_results = pd.read_csv('data/cnn_tuning_results.csv')
    best_cnn = cnn_results.iloc[0]
    best_kernel = int(best_cnn['kernel_size'])
    best_hidden_channels = int(best_cnn['hidden_channels'])

    model_cnn = CNNClassifier(
        input_channels=3,
        window_size=window_size,
        kernel_size=best_kernel,
        hidden_channels=best_hidden_channels
    )
    cnn_dataset = CNNRollingDataset(df_labeled, window_size)
    X_cnn = torch.stack([sample[0] for sample in cnn_dataset])
    model_cnn.load_state_dict(torch.load(cnn_model_path))
    model_cnn.eval()
    with torch.no_grad():
        outputs_cnn = model_cnn(X_cnn)
        probs_cnn = torch.nn.functional.softmax(outputs_cnn, dim=1).numpy()

    # === Generate LSTM probabilities ===
    lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
    best_lstm = lstm_results.iloc[0]
    best_hidden_size = int(best_lstm['hidden_size'])
    best_num_layers = int(best_lstm['num_layers'])
    lstm_dataset = LSTMRollingDataset(df_labeled, window_size)
    X_lstm = torch.stack([sample[0] for sample in lstm_dataset])

    model_lstm = LSTMClassifier(
        input_size=3,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers
    )
    model_lstm.load_state_dict(torch.load(lstm_model_path))
    model_lstm.eval()
    with torch.no_grad():
        outputs_lstm = model_lstm(X_lstm)
        probs_lstm = torch.nn.functional.softmax(outputs_lstm, dim=1).numpy()

    # === Generate Transformer probabilities ===
    transformer_results = pd.read_csv('data/transformer_tuning_results.csv')
    best_transformer = transformer_results.iloc[0]
    best_dim_model = int(best_transformer['dim_model'])
    best_num_heads = int(best_transformer['num_heads'])
    best_num_layers = int(best_transformer['num_layers'])
    transformer_dataset = TransformerDataset(df_labeled, window_size)
    X_transformer = torch.stack([sample[0] for sample in transformer_dataset])

    model_transformer = TransformerClassifier(
        input_size=3,
        window_size=window_size,
        dim_model=best_dim_model,
        num_heads=best_num_heads,
        num_layers=best_num_layers
    )
    model_transformer.load_state_dict(torch.load(transformer_model_path))
    model_transformer.eval()
    with torch.no_grad():
        outputs_transformer = model_transformer(X_transformer)
        probs_transformer = torch.nn.functional.softmax(outputs_transformer, dim=1).numpy()

    # === Align probability array lengths ===
    min_len = min(len(probs_mlp), len(probs_cnn), len(probs_lstm), len(probs_transformer))
    probs_mlp = probs_mlp[-min_len:]
    probs_cnn = probs_cnn[-min_len:]
    probs_lstm = probs_lstm[-min_len:]
    probs_transformer = probs_transformer[-min_len:]
    df_eval = df_labeled.iloc[-min_len:]

    # === Probabilistic voting ensemble ===
    ensemble_predictions = []
    for p_mlp, p_cnn, p_lstm, p_trans in zip(probs_mlp, probs_cnn, probs_lstm, probs_transformer):
        avg_probs = np.mean(np.vstack([p_mlp, p_cnn, p_lstm, p_trans]), axis=0)
        final_decision = np.argmax(avg_probs)
        ensemble_predictions.append(final_decision)

    print("Prediction counts:", np.unique(ensemble_predictions, return_counts=True))

    # === Backtest ensemble with Kelly position sizing ===
    results, portfolio, trade_returns, kelly_fractions = backtest_strategy_with_kelly_position(
        df_eval,
        ensemble_predictions,
        stop_loss=stop_loss,
        take_profit=take_profit,
        min_pos_size=0.01,
        max_pos_size=0.5,
        return_kelly_fraction=True
    )

    # === Print results ===
    print("\n=== Ensemble (Probabilistic Voting + Kelly) Backtest Performance ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # === Additional metrics ===
    metrics = calculate_additional_metrics(portfolio, trade_returns)
    print("\n=== Additional Trading Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # === Plot signals with Kelly sizing ===
    assert kelly_fractions is not None, "kelly_fractions should never be None here"

    plot_prediction_signals_kelly(
        df_eval,
        ensemble_predictions,
        title="Ensemble Probabilistic Voting Signals (Kelly)",
        save_path=plot_path,
        initial_capital=10000,
        kelly_fraction=kelly_fractions
    )

    # === Output validation ===
    if any(x is None for x in [results, portfolio, trade_returns, ensemble_predictions, df_eval]):
        raise ValueError("One or more outputs are None. Check data and model paths.")

    return results, portfolio, trade_returns, ensemble_predictions, df_eval