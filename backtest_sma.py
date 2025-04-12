import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import setup_logger


def backtest_sma_crossover(
        timeframe: str = 'H1',
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        fast_ma: int = 9,
        slow_ma: int = 21
) -> Tuple[Dict, pd.DataFrame, List[Dict]]:
    """
    Backtest a simple SMA crossover strategy as a sanity check.

    Args:
        timeframe: Timeframe to use for backtesting (e.g., 'H1')
        initial_balance: Initial account balance
        risk_per_trade: Fraction of balance to risk per trade
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period

    Returns:
        Tuple of (metrics, results_dataframe, trades_list)
    """
    logger = setup_logger("SMABacktest")
    logger.info(f"Starting SMA crossover backtest for {timeframe}")
    logger.info(f"Parameters: initial_balance={initial_balance}, risk_per_trade={risk_per_trade}")
    logger.info(f"MA Periods: fast={fast_ma}, slow={slow_ma}")

    # Load data
    from data.storage import DataStorage
    storage = DataStorage()
    latest_data_paths = storage.find_latest_data(timeframe=timeframe)

    if not latest_data_paths:
        logger.error(f"No data found for timeframe {timeframe}")
        return {}, pd.DataFrame(), []

    data_path = latest_data_paths[timeframe]
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded data shape: {df.shape}")

    # Calculate moving averages
    df['fast_ma'] = df['close'].rolling(window=fast_ma).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_ma).mean()
    df['ma_signal'] = np.where(df['fast_ma'] > df['slow_ma'], 1, -1)

    # Drop NaN values from start
    df.dropna(inplace=True)

    # Backtest variables
    balance = initial_balance
    position = 0  # 0: no position, 1: long, -1: short
    position_size = 0.0
    entry_price = 0.0
    trades = []

    # Create results DataFrame
    results = df.copy()
    results['balance'] = initial_balance
    results['position'] = 0
    results['trade_action'] = None
    results['profit_loss'] = 0.0

    # Set spread and slippage
    spread_points = 30
    slippage_points = 5
    point_value = 0.01

    # Run backtest
    logger.info("Running backtest...")
    prev_signal = 0

    for idx, row in results.iterrows():
        current_price = row['close']
        current_signal = row['ma_signal']

        # Generate trade action on signal change
        action = 'HOLD'

        # Signal change from -1 to 1 = BUY
        if prev_signal == -1 and current_signal == 1 and position <= 0:
            action = 'BUY'
        # Signal change from 1 to -1 = SELL
        elif prev_signal == 1 and current_signal == -1 and position >= 0:
            action = 'SELL'

        # Execute trading action
        results.loc[idx, 'trade_action'] = action

        if action == 'BUY':
            # Close short position if exists
            if position == -1:
                # Calculate profit/loss
                exit_price = current_price + (spread_points + slippage_points) * point_value
                profit_loss = (entry_price - exit_price) * position_size

                # Update balance
                balance += profit_loss
                results.loc[idx, 'profit_loss'] = profit_loss

                trades.append({
                    'time': idx,
                    'action': 'CLOSE_SHORT',
                    'price': exit_price,
                    'profit_loss': profit_loss,
                    'balance': balance
                })

            # Open long position
            position_size = (balance * risk_per_trade) / (current_price * 0.01)
            entry_price = current_price + (spread_points + slippage_points) * point_value
            position = 1

            trades.append({
                'time': idx,
                'action': 'BUY',
                'price': entry_price,
                'size': position_size,
                'balance': balance
            })

            logger.debug(f"BUY at {idx}: price={entry_price:.2f}, size={position_size:.4f}")

        elif action == 'SELL':
            # Close long position if exists
            if position == 1:
                # Calculate profit/loss
                exit_price = current_price - (spread_points + slippage_points) * point_value
                profit_loss = (exit_price - entry_price) * position_size

                # Update balance
                balance += profit_loss
                results.loc[idx, 'profit_loss'] = profit_loss

                trades.append({
                    'time': idx,
                    'action': 'CLOSE_LONG',
                    'price': exit_price,
                    'profit_loss': profit_loss,
                    'balance': balance
                })

            # Open short position
            position_size = (balance * risk_per_trade) / (current_price * 0.01)
            entry_price = current_price - (spread_points + slippage_points) * point_value
            position = -1

            trades.append({
                'time': idx,
                'action': 'SELL',
                'price': entry_price,
                'size': position_size,
                'balance': balance
            })

            logger.debug(f"SELL at {idx}: price={entry_price:.2f}, size={position_size:.4f}")

        # Update tracking variables
        results.loc[idx, 'balance'] = balance
        results.loc[idx, 'position'] = position
        prev_signal = current_signal

    # Close final position if open
    if position != 0:
        last_idx = results.index[-1]
        last_price = results.loc[last_idx, 'close']

        if position == 1:
            exit_price = last_price - (spread_points + slippage_points) * point_value
            profit_loss = (exit_price - entry_price) * position_size
        else:
            exit_price = last_price + (spread_points + slippage_points) * point_value
            profit_loss = (entry_price - exit_price) * position_size

        balance += profit_loss
        results.loc[last_idx, 'profit_loss'] += profit_loss
        results.loc[last_idx, 'balance'] = balance

        trades.append({
            'time': last_idx,
            'action': 'CLOSE_FINAL',
            'price': exit_price,
            'profit_loss': profit_loss,
            'balance': balance
        })

    # Calculate cumulative P&L
    results['cum_profit_loss'] = results['profit_loss'].cumsum()

    # Calculate metrics
    logger.info("Calculating performance metrics...")

    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'absolute_return': balance - initial_balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'n_trades': len(trades)
    }

    if metrics['n_trades'] > 0:
        # Calculate win rate, profit factor, etc.
        profit_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        loss_trades = [t for t in trades if t.get('profit_loss', 0) < 0]

        metrics['n_winning_trades'] = len(profit_trades)
        metrics['n_losing_trades'] = len(loss_trades)
        metrics['win_rate'] = len(profit_trades) / len(trades)

        # Profit factor
        total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

        # Max drawdown
        peak = initial_balance
        drawdown = 0
        max_drawdown = 0

        for idx, row in results.iterrows():
            current_balance = row['balance']
            if current_balance > peak:
                peak = current_balance

            drawdown = (peak - current_balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        metrics['max_drawdown_pct'] = max_drawdown

        # Calculate Sharpe Ratio (simplified)
        daily_returns = results['profit_loss'] / results['balance'].shift(1)
        metrics['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * (
                    252 ** 0.5) if daily_returns.std() > 0 else 0
    else:
        # Default values if no trades
        metrics.update({
            'n_winning_trades': 0,
            'n_losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0
        })

    # Log metrics
    logger.info(f"Backtest results:")
    logger.info(f"  Initial balance: ${metrics['initial_balance']:.2f}")
    logger.info(f"  Final balance: ${metrics['final_balance']:.2f}")
    logger.info(f"  Return: {metrics['return_pct']:.2f}%")
    logger.info(f"  Number of trades: {metrics['n_trades']}")

    if metrics['n_trades'] > 0:
        logger.info(f"  Win rate: {metrics['win_rate']:.2f}")
        logger.info(f"  Profit factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

    return metrics, results, trades


if __name__ == "__main__":
    metrics, results, trades = backtest_sma_crossover(timeframe='H1')

    # Save results
    os.makedirs('analysis', exist_ok=True)
    results.to_csv('analysis/sma_backtest_results.csv')

    # Save trades
    import json

    with open('analysis/sma_backtest_trades.json', 'w') as f:
        json.dump([{str(t['time']): t} for t in trades], f, indent=2, default=str)

    print(f"Backtest complete. Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")