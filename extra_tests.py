# test_out_of_sample.py

import torch
import pandas as pd
from models.mlp_model import MLPClassifier
from utils import create_labeled_dataset



# Load full dataset
df_full = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")

# Define out-of-sample period (e.g. last 20%)
split_idx = int(len(df_full) * 0.8)
df_train = df_full.iloc[:split_idx]
df_test = df_full.iloc[split_idx:]

# Load best chromosome from GA results
ga_results = pd.read_csv('data/GA_final_results.csv')
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
        'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Prepare test dataset
labeled_test_df = create_labeled_dataset(df_test, best_chromosome)
X_test = torch.tensor(labeled_test_df[['RSI', 'Interval', 'Trend']].values.astype('float32'))

# Load trained model
hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
model = MLPClassifier_dynamic(hidden_sizes)
model.load_state_dict(torch.load('optimized_models/trained_mlp_final.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    predictions = predicted.numpy()

# Backtest on out-of-sample
results, portfolio, trade_returns = backtest_strategy(labeled_test_df, predictions)

# Print out-of-sample performance
print("\n=== Out-of-Sample Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot predictions and signals
plot_prediction_signals(labeled_test_df, predictions, title="Out-of-Sample Prediction Signals", save_path="plots/oos_prediction_signals.png")

# test_true_out_of_sample.py

import torch
import pandas as pd
from datetime import datetime
from utils import download_stock_data, calculate_RSI, calculate_SMA
from models.mlp_model import MLPClassifier_dynamic
from utils import create_labeled_dataset

# Define out-of-sample period
start_date = "2017-02-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download new out-of-sample data
ticker = "SPY"
df_oos = download_stock_data(ticker, start_date, end_date)
if df_oos is None or df_oos.empty:
    print("No data downloaded for out-of-sample period.")
    exit()

# Calculate indicators
df_oos = calculate_RSI(df_oos)
df_oos = calculate_SMA(df_oos)
df_oos['Trend'] = (df_oos['SMA_50'] > df_oos['SMA_200']).astype(int)

# Load best chromosome from GA results
ga_results = pd.read_csv('data/GA_final_results.csv')
best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
        'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Prepare labeled test dataset
labeled_oos_df = create_labeled_dataset(df_oos, best_chromosome)
X_test = torch.tensor(labeled_oos_df[['RSI', 'Interval', 'Trend']].values.astype('float32'))

# Load trained model
best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
model = MLPClassifier_dynamic(hidden_sizes)
model.load_state_dict(torch.load('optimized_models/trained_mlp_final.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    predictions = predicted.numpy()

# Backtest on true out-of-sample data
results, portfolio, trade_returns = backtest_strategy(labeled_oos_df, predictions)

# Print performance results
print("\n=== True Out-of-Sample Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot prediction signals with positions and PnL
plot_prediction_signals(labeled_oos_df, predictions, title="True OOS Prediction Signals", save_path="plots/true_oos_prediction_signals.png")

# test_cnn.py

import torch
import pandas as pd
from models.cnn_model import CNNClassifier, RollingWindowDataset
from utils import create_labeled_dataset, plot_prediction_signals, backtest_strategy

# Load data
df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

# Load best chromosome
ga_results = pd.read_csv('data/GA_final_results.csv')
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
     'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Create labeled dataset
from utils import create_labeled_dataset
df_labeled = create_labeled_dataset(df, best_chromosome)

# Prepare dataset
window_size = 10
dataset = RollingWindowDataset(df_labeled, window_size)
X_test = torch.stack([sample[0] for sample in dataset])
y_true = [sample[1].item() for sample in dataset]

# Load trained model
model = CNNClassifier(input_channels=3, window_size=window_size)
model.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    predictions = predicted.numpy()

# Backtest
results, portfolio, trade_returns = backtest_strategy(df_labeled.iloc[window_size:], predictions)

# Print results
print("\n=== CNN Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot signals
plot_prediction_signals(df_labeled.iloc[window_size:], predictions, title="CNN Prediction Signals", save_path="plots/cnn_prediction_signals.png")


# test_lstm.py

import torch
import pandas as pd
from models.lstm_model import LSTMClassifier, RollingWindowDataset
from utils import create_labeled_dataset, plot_prediction_signals, backtest_strategy

# Load data and prepare labeled dataset (similar to CNN)

# Prepare dataset
window_size = 10
dataset = RollingWindowDataset(df_labeled, window_size)
X_test = torch.stack([sample[0] for sample in dataset])
y_true = [sample[1].item() for sample in dataset]

# Load trained model
model = LSTMClassifier(input_size=3)
model.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    predictions = predicted.numpy()

# Backtest
results, portfolio, trade_returns = backtest_strategy(df_labeled.iloc[window_size:], predictions)

# Print results
print("\n=== LSTM Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot signals
plot_prediction_signals(df_labeled.iloc[window_size:], predictions, title="LSTM Prediction Signals", save_path="plots/lstm_prediction_signals.png")

# test_cnn_lstm_oos.py

import torch
import pandas as pd
from datetime import datetime
from utils import download_stock_data, calculate_RSI, calculate_SMA, create_labeled_dataset, backtest_strategy, plot_prediction_signals
from models.cnn_model import CNNClassifier, RollingWindowDataset as CNNRollingDataset
from models.lstm_model import LSTMClassifier, RollingWindowDataset as LSTMRollingDataset

# Define out-of-sample period
start_date = "2017-02-01"
end_date = datetime.today().strftime('%Y-%m-%d')
ticker = "SPY"

# Download out-of-sample data
df_oos = download_stock_data(ticker, start_date, end_date)
df_oos = calculate_RSI(df_oos)
df_oos = calculate_SMA(df_oos)
df_oos['Trend'] = (df_oos['SMA_50'] > df_oos['SMA_200']).astype(int)

# Load best chromosome from GA results
ga_results = pd.read_csv('data/GA_final_results.csv')
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
     'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Create labeled dataset
df_labeled = create_labeled_dataset(df_oos, best_chromosome)

# Define window size
window_size = 10

# CNN evaluation
print("\n=== CNN Out-of-Sample Evaluation ===")
cnn_results = pd.read_csv('data/cnn_tuning_results.csv')
best_cnn = cnn_results.iloc[0]
best_kernel = int(best_cnn['kernel_size'])
best_hidden_channels = int(best_cnn['hidden_channels'])

cnn_dataset = CNNRollingDataset(df_labeled, window_size)
X_cnn = torch.stack([sample[0] for sample in cnn_dataset])
model_cnn = CNNClassifier(
    input_channels=3,
    window_size=window_size,
    kernel_size=best_kernel,
    hidden_channels=best_hidden_channels
)
model_cnn.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model_cnn.eval()

with torch.no_grad():
    outputs_cnn = model_cnn(X_cnn)
    _, preds_cnn = torch.max(torch.nn.functional.softmax(outputs_cnn, dim=1), 1)
    predictions_cnn = preds_cnn.numpy()

results_cnn, portfolio_cnn, trade_returns_cnn = backtest_strategy(df_labeled.iloc[window_size:], predictions_cnn)

print("\n=== CNN OOS Backtest Performance ===")
for k, v in results_cnn.items():
    print(f"{k}: {v:.4f}")

plot_prediction_signals(df_labeled.iloc[window_size:], predictions_cnn, title="CNN OOS Signals", save_path="plots/cnn_oos_signals.png")

# LSTM evaluation
print("\n=== LSTM Out-of-Sample Evaluation ===")
lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
best_lstm = lstm_results.iloc[0]
best_hidden_size = int(best_lstm['hidden_size'])
best_num_layers = int(best_lstm['num_layers'])

model_lstm = LSTMClassifier(input_size=3, hidden_size=best_hidden_size, num_layers=best_num_layers)
model_lstm.load_state_dict(torch.load('optimized_models/trained_lstm_best_tuned.pth'))
model_lstm.eval()

with torch.no_grad():
    outputs_lstm = model_lstm(X_lstm)
    _, preds_lstm = torch.max(torch.nn.functional.softmax(outputs_lstm, dim=1), 1)
    predictions_lstm = preds_lstm.numpy()

results_lstm, portfolio_lstm, trade_returns_lstm = backtest_strategy(df_labeled.iloc[window_size:], predictions_lstm)

print("\n=== LSTM OOS Backtest Performance ===")
for k, v in results_lstm.items():
    print(f"{k}: {v:.4f}")

plot_prediction_signals(df_labeled.iloc[window_size:], predictions_lstm, title="LSTM OOS Signals", save_path="plots/lstm_oos_signals.png")

# ensemble_voting.py

import torch
import pandas as pd
import numpy as np
from models.mlp_model import MLPClassifier_dynamic
from models.cnn_model import CNNClassifier, RollingWindowDataset as CNNRollingDataset
from models.lstm_model import LSTMClassifier, RollingWindowDataset as LSTMRollingDataset
from utils import create_labeled_dataset, backtest_strategy, plot_prediction_signals

# Load data
df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

# Load best chromosome
ga_results = pd.read_csv('data/GA_final_results.csv')
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
     'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Create labeled dataset
df_labeled = create_labeled_dataset(df, best_chromosome)
window_size = 10

# Prepare CNN inputs
cnn_dataset = CNNRollingDataset(df_labeled, window_size)
X_cnn = torch.stack([sample[0] for sample in cnn_dataset])

# Prepare LSTM inputs
lstm_dataset = LSTMRollingDataset(df_labeled, window_size)
X_lstm = torch.stack([sample[0] for sample in lstm_dataset])

# Prepare MLP inputs
X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))

# Load models
# CNN
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
model_cnn.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model_cnn.eval()
# LSTM
lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
best_lstm = lstm_results.iloc[0]
best_hidden_size = int(best_lstm['hidden_size'])
best_num_layers = int(best_lstm['num_layers'])

model_lstm = LSTMClassifier(input_size=3, hidden_size=best_hidden_size, num_layers=best_num_layers)
model_lstm.load_state_dict(torch.load('optimized_models/trained_lstm_best_tuned.pth'))
model_lstm.eval()

# MLP
best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
model_mlp = MLPClassifier_dynamic(hidden_sizes)
model_mlp.load_state_dict(torch.load('optimized_models/trained_mlp_final.pth'))
model_mlp.eval()

# Make predictions
with torch.no_grad():
    outputs_cnn = model_cnn(X_cnn)
    _, preds_cnn = torch.max(torch.nn.functional.softmax(outputs_cnn, dim=1), 1)
    predictions_cnn = preds_cnn.numpy()

    outputs_lstm = model_lstm(X_lstm)
    _, preds_lstm = torch.max(torch.nn.functional.softmax(outputs_lstm, dim=1), 1)
    predictions_lstm = preds_lstm.numpy()

    outputs_mlp = model_mlp(X_mlp)
    _, preds_mlp = torch.max(torch.nn.functional.softmax(outputs_mlp, dim=1), 1)
    predictions_mlp = preds_mlp.numpy()

# Align lengths (due to rolling window offsets)
min_len = min(len(predictions_cnn), len(predictions_lstm), len(predictions_mlp))
predictions_cnn = predictions_cnn[-min_len:]
predictions_lstm = predictions_lstm[-min_len:]
predictions_mlp = predictions_mlp[-min_len:]
df_eval = df_labeled.iloc[-min_len:]

# Ensemble voting
ensemble_predictions = []
for p1, p2, p3 in zip(predictions_mlp, predictions_cnn, predictions_lstm):
    votes = [p1, p2, p3]
    majority = max(set(votes), key=votes.count)
    ensemble_predictions.append(majority)

# Backtest
results, portfolio, trade_returns = backtest_strategy(df_eval, ensemble_predictions)

# Print results
print("\n=== Ensemble Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot signals
plot_prediction_signals(df_eval, ensemble_predictions, title="Ensemble Prediction Signals", save_path="plots/ensemble_prediction_signals.png")


# ensemble_voting_oos.py

import torch
import pandas as pd
from datetime import datetime
from models.mlp_model import MLPClassifier_dynamic
from models.cnn_model import CNNClassifier, RollingWindowDataset as CNNRollingDataset
from models.lstm_model import LSTMClassifier, RollingWindowDataset as LSTMRollingDataset
from utils import create_labeled_dataset, backtest_strategy, plot_prediction_signals, download_stock_data, calculate_RSI, calculate_SMA

# Define OOS period
start_date = "2017-02-01"
end_date = datetime.today().strftime('%Y-%m-%d')
ticker = "SPY"

# Download OOS data
df_oos = download_stock_data(ticker, start_date, end_date)
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
window_size = 10

# Prepare inputs
cnn_dataset = CNNRollingDataset(df_labeled, window_size)
X_cnn = torch.stack([sample[0] for sample in cnn_dataset])

lstm_dataset = LSTMRollingDataset(df_labeled, window_size)
X_lstm = torch.stack([sample[0] for sample in lstm_dataset])

X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))

# Load models
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
model_cnn.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model_cnn.eval()

model_lstm = LSTMClassifier(input_size=3)
model_lstm.load_state_dict(torch.load('optimized_models/trained_lstm_best_tuned.pth'))
model_lstm.eval()

best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
model_mlp = MLPClassifier_dynamic(hidden_sizes)
model_mlp.load_state_dict(torch.load('optimized_models/trained_mlp_final.pth'))
model_mlp.eval()

# Make predictions
with torch.no_grad():
    outputs_cnn = model_cnn(X_cnn)
    _, preds_cnn = torch.max(torch.nn.functional.softmax(outputs_cnn, dim=1), 1)
    predictions_cnn = preds_cnn.numpy()

    outputs_lstm = model_lstm(X_lstm)
    _, preds_lstm = torch.max(torch.nn.functional.softmax(outputs_lstm, dim=1), 1)
    predictions_lstm = preds_lstm.numpy()

    outputs_mlp = model_mlp(X_mlp)
    _, preds_mlp = torch.max(torch.nn.functional.softmax(outputs_mlp, dim=1), 1)
    predictions_mlp = preds_mlp.numpy()

# Align lengths
min_len = min(len(predictions_cnn), len(predictions_lstm), len(predictions_mlp))
predictions_cnn = predictions_cnn[-min_len:]
predictions_lstm = predictions_lstm[-min_len:]
predictions_mlp = predictions_mlp[-min_len:]
df_eval = df_labeled.iloc[-min_len:]

# Ensemble majority voting
ensemble_predictions = []
for p1, p2, p3 in zip(predictions_mlp, predictions_cnn, predictions_lstm):
    votes = [p1, p2, p3]
    majority = max(set(votes), key=votes.count)
    ensemble_predictions.append(majority)

# Backtest
results, portfolio, trade_returns = backtest_strategy(df_eval, ensemble_predictions)

# Print results
print("\n=== Ensemble OOS Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot signals
plot_prediction_signals(df_eval, ensemble_predictions, title="Ensemble OOS Prediction Signals", save_path="plots/ensemble_oos_prediction_signals.png")

# test_transformer.py

import torch
import pandas as pd
from models.transformer_model import TransformerClassifier, TransformerDataset
from utils import create_labeled_dataset, backtest_strategy_with_position_stoploss, plot_prediction_signals

# Load data
df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

# Load best chromosome
ga_results = pd.read_csv('data/GA_final_results.csv')
best_chromosome = ga_results.sort_values('fitness', ascending=False).iloc[0][
    ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
     'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int']
].values.tolist()

# Create labeled dataset
df_labeled = create_labeled_dataset(df, best_chromosome)
window_size = 10

# Prepare dataset
dataset = TransformerDataset(df_labeled, window_size)
X_test = torch.stack([sample[0] for sample in dataset])
y_true = [sample[1].item() for sample in dataset]

# Load trained model
model = TransformerClassifier(input_size=3, window_size=window_size)
model.load_state_dict(torch.load('optimized_models/trained_transformer_window10.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    predictions = predicted.numpy()

# Backtest using risk-managed strategy
results, portfolio, trade_returns = backtest_strategy_with_position_stoploss(
    df_labeled.iloc[window_size:], predictions,
    position_size=0.25, stop_loss=0.03, take_profit=0.10
)

# Print results
print("\n=== Transformer Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot prediction signals
plot_prediction_signals(df_labeled.iloc[window_size:], predictions, title="Transformer Prediction Signals", save_path="plots/transformer_prediction_signals.png")



# ensemble_with_cnn_lstm_prob.py

import torch
import pandas as pd
from datetime import datetime
import numpy as np
from models.mlp_model import MLPClassifier_dynamic
from models.cnn_model import CNNClassifier, RollingWindowDataset as CNNRollingDataset
from models.lstm_model import LSTMClassifier, RollingWindowDataset as LSTMRollingDataset
from models.transformer_model import TransformerClassifier, TransformerDataset
from utils import create_labeled_dataset, backtest_strategy_with_position_stoploss, plot_prediction_signals, download_stock_data, calculate_RSI, calculate_SMA

# Load data (true out-of-sample)
start_date = "2017-02-01"
end_date = datetime.today().strftime('%Y-%m-%d')
ticker = "SPY"

df_oos = download_stock_data(ticker, start_date, end_date)
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
window_size = 10

# === Generate softmax probabilities for each model ===

# MLP probabilities
X_mlp = torch.tensor(df_labeled[['RSI', 'Interval', 'Trend']].values.astype('float32'))
best_params = pd.read_csv("optimized_models/mlp_tuning_results.csv")
hidden_sizes_str = best_params.loc[0, 'hidden_sizes']
hidden_sizes = ast.literal_eval(str(hidden_sizes_str))
model_mlp = MLPClassifier_dynamic(hidden_sizes)
model_mlp.load_state_dict(torch.load('optimized_models/trained_mlp_final.pth'))
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
model_cnn.load_state_dict(torch.load('optimized_models/trained_cnn_best_tuned.pth'))
model_cnn.eval()
with torch.no_grad():
    outputs_cnn = model_cnn(X_cnn)
    probs_cnn = torch.nn.functional.softmax(outputs_cnn, dim=1).numpy()

# LSTM probabilities
lstm_results = pd.read_csv('data/lstm_tuning_results.csv')
best_lstm = lstm_results.iloc[0]
best_hidden_size = int(best_lstm['hidden_size'])
best_num_layers = int(best_lstm['num_layers'])

model_lstm = LSTMClassifier(input_size=3, hidden_size=best_hidden_size, num_layers=best_num_layers)
model_lstm.load_state_dict(torch.load('optimized_models/trained_lstm_best_tuned.pth'))
model_lstm.eval()
with torch.no_grad():
    outputs_lstm = model_lstm(X_lstm)
    probs_lstm = torch.nn.functional.softmax(outputs_lstm, dim=1).numpy()


# === Align probability array lengths ===
min_len = min(len(probs_mlp), len(probs_cnn), len(probs_lstm))
probs_mlp = probs_mlp[-min_len:]
probs_cnn = probs_cnn[-min_len:]
probs_lstm = probs_lstm[-min_len:]
df_eval = df_labeled.iloc[-min_len:]

# === Probabilistic voting ensemble ===
ensemble_predictions = []
for p_mlp, p_cnn, p_lstm in zip(probs_mlp, probs_cnn, probs_lstm):
    all_probs = np.vstack([p_mlp, p_cnn, p_lstm])
    avg_probs = np.mean(all_probs, axis=0)
    final_decision = np.argmax(avg_probs)
    ensemble_predictions.append(final_decision)

print("Prediction counts:", np.unique(ensemble_predictions, return_counts=True))

# === Backtest ensemble ===
results, portfolio, trade_returns = backtest_strategy_with_position_stoploss(
    df_eval, ensemble_predictions,
    position_size=0.25, stop_loss=0.03, take_profit=0.10
)

# Print results
print("\n=== Ensemble (Probabilistic Voting) Backtest Performance ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Plot signals
plot_prediction_signals(df_eval, ensemble_predictions, title="Ensemble Probabilistic Voting Signals", save_path="plots/ensemble_probabilistic_voting_signals.png")