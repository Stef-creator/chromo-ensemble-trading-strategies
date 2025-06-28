# stock_data.py

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import torch.nn.functional as F
from collections import defaultdict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data using Adjusted Close only.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df is not None and not df.empty:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.reset_index(inplace=True)
    else:
        print("df is None or empty")
        return None

    return df

def calculate_RSI(df, intervals=range(1, 21)):
    for period in intervals:
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df[f"RSI_{period}"] = rsi
    return df

def calculate_SMA(df, periods=[50, 200]):
    for period in periods:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
    return df

def add_technical_indicators(df, ticker):
    df = calculate_RSI(df)
    df = calculate_SMA(df)

    # Determine trend direction (1=uptrend, 0=downtrend)
    df["Trend"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    
    os.makedirs("data", exist_ok=True)

    # Save to CSV for inspection
    df.to_csv(f"data/{ticker}_technical_indicators.csv", index=False)
    print(f"Data saved: {ticker}_technical_indicators.csv")
    return df

def create_labeled_dataset(df, best_chromosome):
    labels = []
    rsi_values = []
    intervals = []
    trends = []
    closes = []

    for _, row in df.iterrows():
        trend = row['Trend']

        if trend == 0:
            buy_val, buy_int, sell_val, sell_int = best_chromosome[:4]
        else:
            buy_val, buy_int, sell_val, sell_int = best_chromosome[4:]

        rsi_col = f'RSI_{int(buy_int)}'
        if rsi_col in row:
            rsi_value = row[rsi_col]
            if rsi_value < buy_val:
                label = 1
            elif rsi_value > sell_val:
                label = 2
            else:
                label = 0

            labels.append(label)
            rsi_values.append(rsi_value)
            intervals.append(buy_int)
            trends.append(trend)
            closes.append(row['Close'])  

    labeled_df = pd.DataFrame({
        'RSI': rsi_values,
        'Interval': intervals,
        'Trend': trends,
        'Close': closes,
        'Label': labels
    })

    labeled_df.dropna(inplace=True)
    return labeled_df

def plot_equity_curve(capital_series, title="Equity Curve", save_path=None):
    """
    Plots the equity curve of the strategy.

    Args:
        capital_series (pd.Series): Series of portfolio value over time.
        title (str): Plot title.
        save_path (str): If provided, saves the plot to this path.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(capital_series.index, capital_series.values, label='Equity Curve')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=["Hold", "Buy", "Sell"], title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix for classification results.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names in order.
        title (str): Plot title.
        save_path (str): If provided, saves the plot to this path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(6, 6))
    cm_display.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title(title)
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def calculate_max_consecutive_streaks(trade_returns):
    """
    Calculates the maximum consecutive wins and losses.

    Args:
        trade_returns (list or pd.Series): List of trade profit/loss values.

    Returns:
        max_wins (int): Maximum consecutive profitable trades.
        max_losses (int): Maximum consecutive losing trades.
    """
    max_wins = max_losses = current_wins = current_losses = 0

    for ret in trade_returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
        elif ret < 0:
            current_losses += 1
            current_wins = 0
        else:
            # treat zero return as neutral, reset both
            current_wins = 0
            current_losses = 0

        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)

    return max_wins, max_losses

def backtest_strategy(df, predictions):
    capital = 10000
    position = 0
    entry_price = None
    capital_series = []
    trade_returns = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = df.iloc[i]['Close']

        if pred == 1 and position == 0:
            position = capital / price
            entry_price = price  # record entry price
            capital = 0

        elif pred == 2 and position > 0:
            exit_price = price
            capital = position * exit_price

            # Calculate trade return
            trade_pnl = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_pnl)

            position = 0
            entry_price = None

        total_value = capital if position == 0 else position * price
        capital_series.append(total_value)

    portfolio = pd.Series(capital_series)
    daily_returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()

    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

    return results, portfolio, trade_returns

def plot_prediction_signals(df, predictions, title="Prediction Signals with PnL", save_path=None):
    """
    Plots buy/sell/hold signals on price and calculates cumulative PnL.

    Args:
        df (pd.DataFrame): DataFrame containing 'Close' prices.
        predictions (array-like): Predicted labels (0=Hold, 1=Buy, 2=Sell).
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
    """
    prices = df['Close'].values
    position = 0
    entry_price = None
    trade_pnls = []
    position_series = []

    buy_signals = []
    sell_signals = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]

        if pred == 1 and position == 0:
            position = 1
            entry_price = price
            buy_signals.append((i, price))
        elif pred == 2 and position == 1:
            position = 0
            pnl = (price - entry_price) / entry_price
            trade_pnls.append(pnl)
            sell_signals.append((i, price))
            entry_price = None

        position_series.append(position)

    # Calculate cumulative PnL
    cumulative_pnl = [0]
    total_pnl = 0
    for pnl in trade_pnls:
        total_pnl += pnl
        cumulative_pnl.append(total_pnl)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, prices, label='Price', color='black')

    # Plot buy signals
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        buy_indices = list(buy_indices)  # ✅ convert to list
        plt.scatter(df.index[buy_indices], buy_prices, marker='^', color='green', label='Buy Signal')

    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        sell_indices = list(sell_indices)  # ✅ convert to list
        plt.scatter(df.index[sell_indices], sell_prices, marker='v', color='red', label='Sell Signal')

    plt.title(f"{title} | Total PnL: {total_pnl:.2%}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Prediction signal plot saved to {save_path}")
    else:
        plt.show()

    return trade_pnls, total_pnl

def backtest_strategy_with_position_stoploss(df, predictions, position_size=0.25, stop_loss=0.05, take_profit=0.10):
    """
    Backtest strategy with position sizing and stop-loss/take-profit.

    Args:
        df (pd.DataFrame): Data with 'Close' prices.
        predictions (array-like): Predicted labels.
        position_size (float): Fraction of capital per trade (e.g. 0.25 for 25%).
        stop_loss (float): Stop-loss threshold as decimal (e.g. 0.05 for 5% loss).
        take_profit (float): Take-profit threshold as decimal (e.g. 0.10 for 10% profit).

    Returns:
        results (dict): Performance metrics.
        portfolio (pd.Series): Equity curve.
        trade_returns (list): List of trade PnLs.
    """
    capital = 10000
    position = 0
    entry_price = None
    trade_pnls = []
    capital_series = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = df.iloc[i]['Close']

        # Check stop-loss or take-profit if position is held
        if position > 0 and entry_price is not None:
            pnl = (price - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                capital += position * price
                trade_pnls.append(pnl)
                position = 0
                entry_price = None

        # Buy signal
        if pred == 1 and position == 0:
            alloc = capital * position_size
            position = alloc / price
            capital -= alloc
            entry_price = price

        # Sell signal
        elif pred == 2 and position > 0:
            pnl = (price - entry_price) / entry_price
            capital += position * price
            trade_pnls.append(pnl)
            position = 0
            entry_price = None

        # Update portfolio value
        total_value = capital if position == 0 else capital + position * price
        capital_series.append(total_value)

    # Close remaining position at end
    if position > 0 and entry_price is not None:
        price = df.iloc[-1]['Close']
        pnl = (price - entry_price) / entry_price
        capital += position * price
        trade_pnls.append(pnl)

    portfolio = pd.Series(capital_series)
    daily_returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()

    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

    return results, portfolio, trade_pnls

def backtest_strategy_with_dynamic_position(df, predictions, stop_loss=0.05, take_profit=0.10, vol_window=20, base_risk=0.01, max_pos_size=0.5, min_pos_size=0.01):
    """
    Backtest strategy with dynamic position sizing based on volatility.

    Args:
        df (pd.DataFrame): Data with 'Close' prices.
        predictions (array-like): Predicted labels.
        stop_loss (float): Stop-loss threshold as decimal.
        take_profit (float): Take-profit threshold as decimal.
        vol_window (int): Rolling window for volatility calculation.
        base_risk (float): Base risk fraction of capital per unit volatility.
        max_pos_size (float): Maximum position size fraction.
        min_pos_size (float): Minimum position size fraction.

    Returns:
        results (dict): Performance metrics.
        portfolio (pd.Series): Equity curve.
        trade_pnls (list): List of trade PnLs.
    """

    capital = 10000
    position = 0
    entry_price = None
    trade_pnls = []
    capital_series = []

    # Calculate rolling volatility (daily returns std)
    df = df.copy()  # avoid modifying original DataFrame
    df.loc[:, 'Return'] = df['Close'].pct_change()
    df.loc[:, 'Volatility'] = df['Return'].rolling(window=vol_window).std().bfill()

    for i in range(len(predictions)):
        pred = predictions[i]
        price = df.iloc[i]['Close']
        vol = df.iloc[i]['Volatility']

        # Robust position size calculation
        if vol <= 0 or pd.isna(vol):
            pos_size = min_pos_size
        else:
            pos_size = base_risk / vol
            pos_size = max(min(pos_size, max_pos_size), min_pos_size)

        alloc = capital * pos_size

        # Check stop-loss or take-profit
        if position > 0 and entry_price is not None:
            pnl = (price - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                capital += position * price
                trade_pnls.append(pnl)
                position = 0
                entry_price = None

        # Buy signal
        if pred == 1 and position == 0:
            position = alloc / price
            capital -= alloc
            entry_price = price

        # Sell signal
        elif pred == 2 and position > 0:
            pnl = (price - entry_price) / entry_price
            capital += position * price
            trade_pnls.append(pnl)
            position = 0
            entry_price = None

        # Update portfolio value
        total_value = capital if position == 0 else capital + position * price
        capital_series.append(total_value)

    # Close remaining position at end
    if position > 0 and entry_price is not None:
        price = df.iloc[-1]['Close']
        pnl = (price - entry_price) / entry_price
        capital += position * price
        trade_pnls.append(pnl)

    portfolio = pd.Series(capital_series)
    daily_returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()

    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

    return results, portfolio, trade_pnls

def backtest_strategy_with_kelly_position(df, predictions, stop_loss=0.05, take_profit=0.10, min_pos_size=0.01, max_pos_size=0.5):
    """
    Backtest strategy with position sizing based on Kelly criterion.

    Args:
        df (pd.DataFrame): Data with 'Close' prices.
        predictions (array-like): Predicted labels.
        stop_loss (float): Stop-loss threshold as decimal.
        take_profit (float): Take-profit threshold as decimal.
        min_pos_size (float): Minimum position size fraction.
        max_pos_size (float): Maximum position size fraction.

    Returns:
        results (dict): Performance metrics.
        portfolio (pd.Series): Equity curve.
        trade_pnls (list): List of trade PnLs.
    """

    capital = 10000
    position = 0
    entry_price = None
    trade_pnls = []
    wins = []
    losses = []
    capital_series = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = df.iloc[i]['Close']

        # Calculate Kelly position size from historical trades
        if wins and losses:
            p = len(wins) / (len(wins) + len(losses))
            avg_win = np.mean(wins)
            avg_loss = -np.mean(losses)  # losses are negative, so negate for magnitude
            b = avg_win / avg_loss if avg_loss != 0 else 1

            # Calculate full Kelly fraction
            f_star = (b * p - (1 - p)) / b

            # Use half Kelly for stability
            f_star *= 0.5

            # Avoid betting if negative edge
            if f_star <= 0:
                pos_size = 0
            else:
                pos_size = max(min(float(f_star), max_pos_size), min_pos_size)
        else:
            pos_size = min_pos_size  # Default early position size

        alloc = capital * pos_size

        # Check stop-loss or take-profit
        if position > 0 and entry_price is not None:
            pnl = (price - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                capital += position * price
                trade_pnls.append(pnl)
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(pnl)
                position = 0
                entry_price = None

        # Buy signal
        if pred == 1 and position == 0 and pos_size > 0:
            position = alloc / price
            capital -= alloc
            entry_price = price

        # Sell signal
        elif pred == 2 and position > 0:
            pnl = (price - entry_price) / entry_price
            capital += position * price
            trade_pnls.append(pnl)
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
            position = 0
            entry_price = None

        # Update portfolio value
        total_value = capital if position == 0 else capital + position * price
        capital_series.append(total_value)

    # Close remaining position at end
    if position > 0 and entry_price is not None:
        price = df.iloc[-1]['Close']
        pnl = (price - entry_price) / entry_price
        capital += position * price
        trade_pnls.append(pnl)
        if pnl > 0:
            wins.append(pnl)
        else:
            losses.append(pnl)

    portfolio = pd.Series(capital_series)
    daily_returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()

    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

    return results, portfolio, trade_pnls

def comparative_evaluation_summary(results_list, sort_by='Sharpe Ratio', save_path=None):
    """
    Generate a comparative evaluation summary DataFrame from a list of results.

    Args:
        results_list (list of dict): Each dict contains model results with keys:
            - 'Model'
            - 'Total Return'
            - 'Annualized Return'
            - 'Sharpe Ratio'
            - 'Max Drawdown'
        sort_by (str): Metric to sort by, default 'Sharpe Ratio'.
        save_path (str): Optional path to save the summary CSV.

    Returns:
        pd.DataFrame: Sorted summary DataFrame.
    """

    df_summary = pd.DataFrame(results_list)
    df_summary.sort_values(by=sort_by, ascending=False, inplace=True)

    print("\n=== Comparative Evaluation Summary ===")
    print(df_summary.to_string(index=False))

    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"Summary saved to {save_path}")

    return df_summary

def weighted_ensemble_vote(votes, weights):
    """
    Perform weighted voting for an ensemble.
    
    Args:
        votes (list): List of model votes (0, 1, or 2).
        weights (list): List of weights for each model, same length as votes.
    
    Returns:
        int: Final ensemble decision (0, 1, or 2).
    """
    assert len(votes) == len(weights), "Votes and weights must be same length."
    
    vote_weights = defaultdict(float)
    
    for vote, weight in zip(votes, weights):
        vote_weights[vote] += weight
    
    # Determine the class with the highest weighted sum
    max_vote = max(vote_weights.items(), key=lambda x: x[1])[0]
    
    return max_vote