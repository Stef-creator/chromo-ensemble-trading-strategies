# stock_data.py

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Sequence

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

def plot_prediction_signals_dynamic(df, executed_trades, title="Prediction Signals with PnL", save_path=None, initial_capital=10000):
    """
    Plots buy/sell signals on price using executed trades for exact alignment with backtest.

    Args:
        df (pd.DataFrame): DataFrame containing 'Close' prices.
        executed_trades (dict): Dict containing 'buy_indices', 'buy_prices', 'sell_indices', 'sell_prices', 'capital_series'.
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
        initial_capital (float): Starting portfolio capital.

    Returns:
        compounded_return (float): Final portfolio return relative to initial capital.
    """
    prices = df['Close'].values
    capital_series = executed_trades['capital_series']
    buy_indices = executed_trades['buy_indices']
    buy_prices = executed_trades['buy_prices']
    sell_indices = executed_trades['sell_indices']
    sell_prices = executed_trades['sell_prices']

    # Calculate compounded return
    compounded_return = (capital_series[-1] / initial_capital) - 1 if capital_series else 0.0

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, prices, label='Price', color='black')

    if buy_indices:
        plt.scatter(df.index[buy_indices], buy_prices, marker='^', color='green', label='Buy Signal')

    if sell_indices:
        plt.scatter(df.index[sell_indices], sell_prices, marker='v', color='red', label='Sell Signal')

    plt.title(f"{title} | Portfolio Return: {compounded_return:.2%}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Prediction signal plot saved to {save_path}")
    else:
        plt.show()

    return compounded_return

def plot_prediction_signals(df, predictions, title="Prediction Signals with PnL", save_path=None, initial_capital=10000):
    """
    Plots buy/sell/hold signals on price and calculates cumulative PnL and portfolio return.

    Args:
        df (pd.DataFrame): DataFrame containing 'Close' prices.
        predictions (array-like): Predicted labels (0=Hold, 1=Buy, 2=Sell).
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
        initial_capital (float): Starting portfolio capital.
    """
    prices = df['Close'].values
    position = 0
    entry_price = None
    trade_pnls = []
    portfolio = initial_capital
    capital_series = []

    buy_signals = []
    sell_signals = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]

        if pred == 1 and position == 0:
            position = portfolio / price  # buy with full portfolio value
            entry_price = price
            buy_signals.append((i, price))
            portfolio = 0  # fully invested

        elif pred == 2 and position > 0:
            portfolio = position * price  # sell and convert back to cash
            pnl = (price - entry_price) / entry_price
            trade_pnls.append(pnl)
            sell_signals.append((i, price))
            position = 0
            entry_price = None

        # Calculate portfolio value at each step
        total_value = portfolio if position == 0 else position * price
        capital_series.append(total_value)

    # Close open position at the end if any
    if position > 0:
        portfolio = position * prices[-1]
        pnl = (prices[-1] - entry_price) / entry_price
        trade_pnls.append(pnl)
        position = 0

    # Calculate metrics
    total_pnl = sum(trade_pnls)
    compounded_return = (capital_series[-1] / initial_capital) - 1

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, prices, label='Price', color='black')

    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        buy_indices = list(buy_indices)
        plt.scatter(df.index[buy_indices], buy_prices, marker='^', color='green', label='Buy Signal')

    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        sell_indices = list(sell_indices)
        plt.scatter(df.index[sell_indices], sell_prices, marker='v', color='red', label='Sell Signal')

    plt.title(f"{title} | Summed PnL: {total_pnl:.2%} | Portfolio Return: {compounded_return:.2%}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Prediction signal plot saved to {save_path}")
    else:
        plt.show()

    return trade_pnls, total_pnl, compounded_return

def plot_prediction_signals_kelly(
    df,
    predictions,
    title="Prediction Signals with Kelly Sizing",
    save_path=None,
    initial_capital=10000,
    kelly_fraction: Union[float, Sequence[float], np.ndarray] = 0.1
):
    """
    Plots buy/sell/hold signals on price and calculates portfolio return with Kelly position sizing.

    Args:
        df (pd.DataFrame): DataFrame containing 'Close' prices.
        predictions (array-like): Predicted labels (0=Hold, 1=Buy, 2=Sell).
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
        initial_capital (float): Starting portfolio capital.
        kelly_fraction (float or array-like): Kelly position size fraction (0-1). Can be scalar or precomputed array per trade.

    Returns:
        trade_pnls (list): List of individual trade returns.
        summed_pnl (float): Sum of trade returns.
        portfolio_return (float): Total portfolio return relative to initial capital.
    """
    prices = df['Close'].values
    capital = initial_capital
    position = 0
    entry_price = None
    trade_pnls = []
    capital_series = []

    buy_signals = []
    sell_signals = []

    # Convert kelly_fraction to np.array for indexing consistency
    if not isinstance(kelly_fraction, (list, tuple, np.ndarray)):
        kelly_fraction = np.full(len(predictions), kelly_fraction)
    else:
        kelly_fraction = np.array(kelly_fraction)

    # Check length consistency
    if len(kelly_fraction) != len(predictions):
        raise ValueError(f"kelly_fraction length {len(kelly_fraction)} does not match predictions length {len(predictions)}")

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]
        kelly_size = kelly_fraction[i]

        # Buy
        if pred == 1 and position == 0:
            alloc = capital * kelly_size
            position = alloc / price
            entry_price = price
            capital -= alloc
            buy_signals.append((i, price))

        # Sell
        elif pred == 2 and position > 0:
            proceeds = position * price
            pnl = (price - entry_price) / entry_price
            trade_pnls.append(pnl)
            capital += proceeds
            position = 0
            entry_price = None
            sell_signals.append((i, price))

        # Update portfolio value
        total_value = capital if position == 0 else capital + position * price
        capital_series.append(total_value)

    # Close any open position at the end
    if position > 0:
        proceeds = position * prices[-1]
        pnl = (prices[-1] - entry_price) / entry_price
        trade_pnls.append(pnl)
        capital += proceeds
        position = 0

    # Calculate metrics
    summed_pnl = sum(trade_pnls)
    portfolio_return = (capital_series[-1] / initial_capital) - 1 if capital_series else 0.0

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, prices, label='Price', color='black')

    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        plt.scatter(df.index[list(buy_indices)], buy_prices, marker='^', color='green', label='Buy Signal')

    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        plt.scatter(df.index[list(sell_indices)], sell_prices, marker='v', color='red', label='Sell Signal')

    plt.title(f"{title} | Summed PnL: {summed_pnl:.2%} | Portfolio Return: {portfolio_return:.2%}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Prediction signal plot saved to {save_path}")
    else:
        plt.show()

    return trade_pnls, summed_pnl, portfolio_return

def backtest_strategy_with_position_stoploss(df, predictions, position_size: float = 1, stop_loss: float = 1, take_profit: float = 1000, return_executed_trades=False):
    """
    Backtest strategy with position sizing and stop-loss/take-profit.

    Args:
        df (pd.DataFrame): Data with 'Close' prices.
        predictions (array-like): Predicted labels.
        position_size (float): Fraction of capital per trade (e.g. 0.25 for 25%).
        stop_loss (float): Stop-loss threshold as decimal.
        take_profit (float): Take-profit threshold as decimal.
        return_executed_trades (bool): Whether to return executed trades dict for plotting.

    Returns:
        results (dict): Performance metrics.
        portfolio (pd.Series): Equity curve.
        trade_returns (list): List of trade PnLs.
        executed_trades (dict, optional): Dict with buy/sell indices and prices, capital_series.
    """
    capital = 10000
    position = 0
    entry_price = None
    trade_pnls = []
    capital_series = []

    buy_indices = []
    buy_prices = []
    sell_indices = []
    sell_prices = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = df.iloc[i]['Close']

        # Check stop-loss or take-profit if position is held
        if position > 0 and entry_price is not None:
            pnl = (price - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                capital += position * price
                trade_pnls.append(pnl)
                sell_indices.append(i)
                sell_prices.append(price)
                position = 0
                entry_price = None

        # Buy signal
        if pred == 1 and position == 0:
            alloc = capital * position_size
            position = alloc / price
            capital -= alloc
            entry_price = price
            buy_indices.append(i)
            buy_prices.append(price)

        # Sell signal
        elif pred == 2 and position > 0:
            pnl = (price - entry_price) / entry_price
            capital += position * price
            trade_pnls.append(pnl)
            sell_indices.append(i)
            sell_prices.append(price)
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
        sell_indices.append(len(df) - 1)
        sell_prices.append(price)

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

    # Return executed trades dict if requested
    if return_executed_trades:
        executed_trades = {
            'buy_indices': buy_indices,
            'buy_prices': buy_prices,
            'sell_indices': sell_indices,
            'sell_prices': sell_prices,
            'capital_series': capital_series
        }
        return results, portfolio, trade_pnls, executed_trades
    else:
        return results, portfolio, trade_pnls, None

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

def backtest_strategy_with_kelly_position(df, predictions, stop_loss=0.05, take_profit=0.10, min_pos_size=0.01, max_pos_size=0.5, return_kelly_fraction=False):
    """
    Backtest strategy with position sizing based on Kelly criterion.

    Args:
        df (pd.DataFrame): Data with 'Close' prices.
        predictions (array-like): Predicted labels.
        stop_loss (float): Stop-loss threshold as decimal.
        take_profit (float): Take-profit threshold as decimal.
        min_pos_size (float): Minimum position size fraction.
        max_pos_size (float): Maximum position size fraction.
        return_kelly_fraction (bool): Whether to return per-trade Kelly fractions.

    Returns:
        results (dict): Performance metrics.
        portfolio (pd.Series): Equity curve.
        trade_pnls (list): List of trade PnLs.
        kelly_fractions (np.array, optional): Per-trade Kelly fractions if return_kelly_fraction=True.
    """
    import numpy as np
    import pandas as pd

    capital = 10000
    position = 0
    entry_price = None
    trade_pnls = []
    wins = []
    losses = []
    capital_series = []
    kelly_fractions = []

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

        kelly_fractions.append(pos_size)  # save Kelly fraction for this trade

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

    kelly_fractions = np.array(kelly_fractions)

    if return_kelly_fraction:
        return results, portfolio, trade_pnls, kelly_fractions
    else:
        return results, portfolio, trade_pnls, None

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

def calculate_additional_metrics(portfolio, trade_returns):
    """
    Calculates advanced trading evaluation metrics.

    Args:
        portfolio (pd.Series): Portfolio equity curve.
        trade_returns (list or np.array): List of individual trade returns.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    daily_returns = portfolio.pct_change().dropna()

    # Maximum Drawdown
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()

    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino = daily_returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else np.nan

    # Calmar Ratio (annual return / max drawdown)
    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Omega Ratio (threshold 0)
    gains = daily_returns[daily_returns > 0].sum()
    losses = -daily_returns[daily_returns < 0].sum()
    omega = gains / losses if losses != 0 else np.nan

    # Gain to Pain Ratio (sum of returns / sum of absolute losses)
    gain_to_pain = daily_returns.sum() / losses if losses != 0 else np.nan

    # Consistency metrics
    profitable_trades = len([r for r in trade_returns if r > 0])
    total_trades = len(trade_returns)
    win_rate = profitable_trades / total_trades if total_trades > 0 else np.nan
    profit_factor = (sum([r for r in trade_returns if r > 0]) / -sum([r for r in trade_returns if r < 0])) if sum([r for r in trade_returns if r < 0]) != 0 else np.nan

    # Longest winning and losing streaks
    streaks = []
    current_streak = 0
    current_sign = None
    for r in trade_returns:
        sign = r > 0
        if sign == current_sign:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append((current_sign, current_streak))
            current_sign = sign
            current_streak = 1
    if current_streak > 0:
        streaks.append((current_sign, current_streak))

    longest_win_streak = max([s[1] for s in streaks if s[0]], default=0)
    longest_loss_streak = max([s[1] for s in streaks if not s[0]], default=0)

    # Compile metrics
    metrics = {
        'Sortino Ratio': float(sortino),
        'Calmar Ratio': float(calmar),
        'Omega Ratio': float(omega),
        'Max Drawdown': float(max_drawdown),
        'Gain to Pain Ratio': float(gain_to_pain),
        'Win Rate': float(win_rate),
        'Profit Factor': float(profit_factor),
        'Longest Win Streak': float(longest_win_streak),
        'Longest Loss Streak': float(longest_loss_streak)
    }

    return metrics

def ensemble_comparison_summary(metrics_weighted_majority, metrics_kelly, metrics_prob_voting_uncon, metrics_prob_voting_con_min_pos, metrics_prob_voting_tot_con, save_path=None):
    def safe_get(d, key):
        # Check both lowercase and title case variants
        for variant in [key, key.title(), key.lower()]:
            if variant in d:
                return d[variant]
        return np.nan

    results_list = []
    for name, metrics in [
        ('Weighted Majority Voting', metrics_weighted_majority),
        ('Probabilistic Voting (Unconstrained)', metrics_prob_voting_uncon),
        ('Probabilistic Voting (Constrained, Min Pos)', metrics_prob_voting_con_min_pos),
        ('Probabilistic Voting (Totally Constrained)', metrics_prob_voting_tot_con),
        ('Probabilistic Voting + Kelly Sizing', metrics_kelly),
    ]:
        results_list.append({
            'Ensemble Method': name,
            'Total Return': safe_get(metrics, 'total_return'),
            'Annualized Return': safe_get(metrics, 'annualized_return'),
            'Sharpe Ratio': safe_get(metrics, 'sharpe_ratio'),
            'Sortino Ratio': safe_get(metrics, 'Sortino Ratio'),
            'Max Drawdown': safe_get(metrics, 'max_drawdown'),
            'Win Rate': safe_get(metrics, 'Win Rate'),
            'Profit Factor': safe_get(metrics, 'Profit Factor')
        })

    results_df = pd.DataFrame(results_list)
    columns_order = ['Ensemble Method', 'Total Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    results_df = results_df[columns_order]

    pd.set_option('display.precision', 4)
    print("\n=== Ensemble Model Comparison Summary ===")
    print(results_df.to_string(index=False))

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Saved ensemble comparison table to {save_path}")

    return results_df