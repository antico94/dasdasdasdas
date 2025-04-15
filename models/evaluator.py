import os
import time
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from strategy.strategies import Strategy, StrategyFactory
from models.factory import ModelFactory
from data.processor import DataProcessor
from data.storage import DataStorage
from utils.logger import setup_logger


def backtest_model_with_strategy(
        config: Dict,
        model_path: str,
        timeframe: str,
        strategy: Strategy,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        include_spread: bool = True,
        include_slippage: bool = True
) -> Tuple[Dict, pd.DataFrame, List[Dict]]:
    # Setup enhanced logging
    logger = setup_logger(name="BacktestLogger")
    logger.info(f"Starting backtest for model: {model_path}, timeframe: {timeframe}")
    logger.info(f"Using strategy: {strategy.name}")
    logger.info(f"Parameters: initial_balance={initial_balance}, risk_per_trade={risk_per_trade}")
    logger.info(f"Include spread: {include_spread}, Include slippage: {include_slippage}")

    # Load the trained model
    logger.info(f"Loading model from {model_path}")
    model = ModelFactory.load_model(model_path)
    logger.info(f"Model type: {type(model).__name__}")

    # Extract model metadata if available
    model_metadata = {}
    metadata_path = model_path.replace(".joblib", "_metadata.pkl")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "rb") as f:
                model_metadata = pickle.load(f)
            logger.info(f"Loaded model metadata: {list(model_metadata.keys())}")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")

    # Load and prepare test data
    storage = DataStorage()
    latest_processed_data = storage.find_latest_processed_data()
    if timeframe not in latest_processed_data:
        logger.error(f"No processed data found for timeframe {timeframe}")
        return {}, pd.DataFrame(), []

    logger.info(f"Loading test data from {latest_processed_data[timeframe]}")
    try:
        df = pd.read_csv(latest_processed_data[timeframe], index_col=0, parse_dates=True)
        logger.info(f"Loaded data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return {}, pd.DataFrame(), []

    # Use the last 20% of data for testing (or load specific test split if available)
    test_data = df.iloc[int(len(df) * 0.8):].copy()
    logger.info(f"Test data shape: {test_data.shape}")

    # Get horizon from model path or metadata
    horizon = 1  # Default value
    try:
        model_filename = os.path.basename(model_path)
        if '_' in model_filename:
            parts = model_filename.split('_')
            for part in parts:
                if part.isdigit():
                    horizon = int(part)
                    break
        if horizon == 1 and model_metadata and 'prediction_horizon' in model_metadata:
            horizon = model_metadata['prediction_horizon']
        logger.info(f"Using prediction horizon: {horizon}")
    except Exception as e:
        logger.warning(f"Could not extract horizon from model path or metadata: {str(e)}")
        logger.info(f"Using default horizon: {horizon}")

    # Prepare features and target using the DataProcessor
    processor = DataProcessor()
    X_test, y_test = processor.prepare_ml_features(test_data, horizon=horizon)
    logger.info(f"Prepared features shape: {X_test.shape}, target shape: {y_test.shape}")

    # Filter features to match what the model was trained on
    if model_metadata.get('features'):
        fitted_features = model_metadata['features']
        logger.info(f"Filtering test features to fitted features: {fitted_features}")
        # Print warning if there are extra columns
        extra_features = [f for f in X_test.columns if f not in fitted_features]
        if extra_features:
            logger.warning(f"Extra features in test data (will be dropped): {extra_features}")
        missing_features = [f for f in fitted_features if f not in X_test.columns]
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
        X_test = X_test[fitted_features]
    else:
        # Fall back to model.get_feature_names if available
        if hasattr(model, 'get_feature_names'):
            model_features = model.get_feature_names()
            extra_features = [f for f in X_test.columns if f not in model_features]
            if extra_features:
                logger.warning(f"Extra features in test data (will be dropped): {extra_features}")
            X_test = X_test[model_features]

    logger.info(f"Filtered features shape: {X_test.shape}")
    logger.info(f"Feature statistics: min={X_test.min().min():.4f}, max={X_test.max().max():.4f}")

    # Generate model predictions
    logger.info("Generating predictions...")
    model_predictions = pd.DataFrame(index=X_test.index)

    is_classifier = hasattr(model, 'predict_proba')
    if is_classifier:
        try:
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:
                positive_probs = y_pred_proba[:, 1]  # Probability for class 1 (UP)
                model_predictions['pred_probability_up'] = positive_probs
                model_predictions['pred_probability_down'] = 1 - positive_probs

                # Log probability distribution
                prob_bins = np.linspace(0, 1, 11)
                hist, _ = np.histogram(positive_probs, bins=prob_bins)
                bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

                logger.info("Prediction probability distribution:")
                for center, count in zip(bin_centers, hist):
                    logger.info(f"  {center:.1f}: {count} predictions ({count / len(positive_probs) * 100:.1f}%)")
                logger.info(f"Min probability: {positive_probs.min():.4f}")
                logger.info(f"Max probability: {positive_probs.max():.4f}")
                logger.info(f"Mean probability: {positive_probs.mean():.4f}")
        except Exception as e:
            logger.error(f"Error generating prediction probabilities: {str(e)}")
    else:
        y_pred = model.predict(X_test)
        model_predictions['predicted_value'] = y_pred
        logger.info(
            f"Prediction statistics: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")

    # Generate trading signals using the strategy
    test_results = test_data.loc[X_test.index].copy()
    test_results = strategy.generate_signals(test_results, model_predictions)

    logger.info(f"Generated signals using strategy {strategy.name}")
    logger.info(f"Buy signals: {(test_results['signal'] == 'BUY').sum()}")
    logger.info(f"Sell signals: {(test_results['signal'] == 'SELL').sum()}")

    # Initialize backtest variables
    balance = initial_balance
    position = 0  # 0: no position, 1: long, -1: short
    position_size = 0.0
    entry_price = 0.0
    entry_time = None  # Track when we entered a position
    highest_price = 0.0  # Track highest price since entry for trailing stops
    lowest_price = float('inf')  # Track lowest price since entry for trailing stops
    stop_loss_price = 0.0
    take_profit_price = 0.0
    position_data = {}  # Additional data about the current position
    trades = []

    # Initialize tracking columns
    test_results['balance'] = initial_balance
    test_results['position'] = 0
    test_results['trade_action'] = None
    test_results['entry_price'] = None
    test_results['stop_loss'] = None
    test_results['take_profit'] = None
    test_results['profit_loss'] = 0.0
    test_results['cum_profit_loss'] = 0.0
    test_results['position_duration'] = 0.0

    # Define trading costs
    SPREAD_TYPICAL = config.get('risk', {}).get('spread', 0.5)
    SLIPPAGE_TYPICAL = config.get('risk', {}).get('slippage', 0.1)
    POINT_VALUE = 0.01  # For XAU/USD

    spread_points = SPREAD_TYPICAL if include_spread else 0
    slippage_points = SLIPPAGE_TYPICAL if include_slippage else 0
    point_value = POINT_VALUE

    logger.info(f"Using spread: {spread_points} points, slippage: {slippage_points} points")
    logger.info(f"Point value: {point_value}")

    # Run the simulation
    logger.info("Running backtest simulation...")
    for i, (idx, row) in enumerate(test_results.iterrows()):
        if i < horizon:
            continue

        current_price = row['close']
        current_signal = row['signal']

        # Update highest/lowest prices if in a position
        if position == 1:  # Long position
            highest_price = max(highest_price, current_price)
            position_data['highest_price'] = highest_price
        elif position == -1:  # Short position
            lowest_price = min(lowest_price, current_price)
            position_data['lowest_price'] = lowest_price

        # Calculate position duration if in a position
        if position != 0 and entry_time is not None:
            position_duration = (idx - entry_time).total_seconds() / 3600  # hours
            test_results.at[idx, 'position_duration'] = position_duration
            position_data['duration'] = position_duration

        # Update position data with current information
        position_data['current_signal'] = current_signal
        position_data['current_price'] = current_price

        # Determine action based on current position and strategy
        action = "HOLD"
        exit_reason = None

        if position == 0:  # No position - check for entry
            if current_signal == "BUY":
                action = "BUY"
            elif current_signal == "SELL":
                action = "SELL"
        else:  # In a position - check for exit
            should_exit, reason = strategy.should_exit_position(
                position, entry_price, current_price, entry_time, idx, position_data
            )

            if should_exit:
                action = "CLOSE"
                exit_reason = reason.value if reason else "unknown"
            else:
                # Update stop loss and take profit
                stop_loss_price, take_profit_price = strategy.update_stops(
                    position, entry_price, current_price, highest_price, lowest_price,
                    stop_loss_price, take_profit_price
                )

                test_results.at[idx, 'stop_loss'] = stop_loss_price
                test_results.at[idx, 'take_profit'] = take_profit_price

        test_results.at[idx, 'trade_action'] = action

        # Execute trade action
        if action == "BUY":
            # Calculate entry price with costs
            adjusted_price = current_price + slippage_points * point_value

            # Calculate position size
            stop_loss_value = adjusted_price * (1 - strategy.stop_loss_pct / 100)
            position_size = strategy.calculate_position_size(
                balance, adjusted_price, risk_per_trade, stop_loss_value
            )

            # Set up position tracking
            position = 1
            entry_price = adjusted_price
            entry_time = idx
            highest_price = adjusted_price
            stop_loss_price = stop_loss_value
            take_profit_price = adjusted_price * (1 + strategy.take_profit_pct / 100)

            # Record trade
            trade = {
                'time': idx,
                'action': 'BUY',
                'price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'balance': balance
            }
            trades.append(trade)

            logger.debug(
                f"BUY at {idx}: price={entry_price:.2f}, size={position_size:.4f}, "
                f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}"
            )

        elif action == "SELL":
            # Calculate entry price with costs
            adjusted_price = current_price - slippage_points * point_value

            # Calculate position size
            stop_loss_value = adjusted_price * (1 + strategy.stop_loss_pct / 100)
            position_size = strategy.calculate_position_size(
                balance, adjusted_price, risk_per_trade, stop_loss_value
            )

            # Set up position tracking
            position = -1
            entry_price = adjusted_price
            entry_time = idx
            lowest_price = adjusted_price
            stop_loss_price = stop_loss_value
            take_profit_price = adjusted_price * (1 - strategy.take_profit_pct / 100)

            # Record trade
            trade = {
                'time': idx,
                'action': 'SELL',
                'price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'balance': balance
            }
            trades.append(trade)

            logger.debug(
                f"SELL at {idx}: price={entry_price:.2f}, size={position_size:.4f}, "
                f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}"
            )

        elif action == "CLOSE":
            # Calculate exit price with costs
            if position == 1:  # Long position
                exit_price = current_price - slippage_points * point_value
                profit_loss = (exit_price - entry_price) * position_size
                # Deduct spread cost
                profit_loss -= spread_points * point_value * position_size
            else:  # Short position
                exit_price = current_price + slippage_points * point_value
                profit_loss = (entry_price - exit_price) * position_size
                # Deduct spread cost
                profit_loss -= spread_points * point_value * position_size

            # Apply stop-loss risk limit
            max_loss = -initial_balance * 0.05  # Limit loss to 5% per trade
            if profit_loss < max_loss:
                logger.warning(
                    f"Limiting loss at {idx} from {profit_loss:.2f} to {max_loss:.2f}"
                )
                profit_loss = max_loss

            # Update account balance
            balance += profit_loss
            test_results.at[idx, 'profit_loss'] = profit_loss

            # Calculate position duration
            position_duration = (idx - entry_time).total_seconds() / 3600 if entry_time else 0

            # Record trade
            trade = {
                'time': idx,
                'action': 'CLOSE',
                'reason': exit_reason,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'profit_loss': profit_loss,
                'balance': balance,
                'duration_hours': position_duration
            }
            trades.append(trade)

            logger.debug(
                f"CLOSE at {idx}: reason={exit_reason}, price={exit_price:.2f}, "
                f"P/L={profit_loss:.2f}, duration={position_duration:.1f}h"
            )

            # Reset position tracking
            position = 0
            position_size = 0
            entry_price = 0
            entry_time = None
            highest_price = 0
            lowest_price = float('inf')
            stop_loss_price = 0
            take_profit_price = 0
            position_data = {}

        # Update tracking columns
        test_results.at[idx, 'balance'] = balance
        test_results.at[idx, 'position'] = position
        test_results.at[idx, 'entry_price'] = entry_price if position != 0 else None

    # Close any open position at the end of the test
    if position != 0:
        last_idx = test_results.index[-1]
        last_price = test_results.loc[last_idx, 'close']

        # Calculate exit price with costs
        if position == 1:  # Long position
            exit_price = last_price - slippage_points * point_value
            profit_loss = (exit_price - entry_price) * position_size
            profit_loss -= spread_points * point_value * position_size
        else:  # Short position
            exit_price = last_price + slippage_points * point_value
            profit_loss = (entry_price - exit_price) * position_size
            profit_loss -= spread_points * point_value * position_size

        # Calculate position duration
        position_duration = (last_idx - entry_time).total_seconds() / 3600 if entry_time else 0

        logger.info(
            f"Closing final position at {last_idx}: position={position}, "
            f"entry_price={entry_price:.2f}, exit_price={exit_price:.2f}, "
            f"profit_loss={profit_loss:.2f}, duration={position_duration:.1f}h"
        )

        # Add to balance and update results
        balance += profit_loss
        test_results.at[last_idx, 'profit_loss'] = profit_loss
        test_results.at[last_idx, 'balance'] = balance

        # Record trade
        trades.append({
            'time': last_idx,
            'action': 'CLOSE_FINAL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': position_size,
            'profit_loss': profit_loss,
            'balance': balance,
            'duration_hours': position_duration
        })

    # Calculate cumulative profit/loss
    test_results['cum_profit_loss'] = test_results['profit_loss'].fillna(0).cumsum()

    # Calculate metrics
    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'absolute_return': balance - initial_balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'n_trades': len([t for t in trades if t['action'] in ['BUY', 'SELL']]),
        'n_closed_trades': len([t for t in trades if t['action'] in ['CLOSE', 'CLOSE_FINAL']])
    }

    if metrics['n_closed_trades'] > 0:
        # Extract closed trades with P/L
        closed_trades = [t for t in trades if 'profit_loss' in t and t.get('profit_loss', 0) != 0]

        # Calculate winning/losing trades
        profit_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
        loss_trades = [t for t in closed_trades if t.get('profit_loss', 0) < 0]

        metrics['n_winning_trades'] = len(profit_trades)
        metrics['n_losing_trades'] = len(loss_trades)
        metrics['win_rate'] = len(profit_trades) / len(closed_trades) if closed_trades else 0

        # Calculate profit factor
        total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate max drawdown
        peak = initial_balance
        max_drawdown = 0

        for balance in test_results['balance']:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        metrics['max_drawdown_pct'] = max_drawdown

        # Calculate Sharpe ratio (annualized)
        daily_returns = test_results['profit_loss'] / test_results['balance'].shift(1)
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(daily_returns) > 1:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0

        # Additional metrics
        if closed_trades:
            # Average trade duration
            durations = [t.get('duration_hours', 0) for t in closed_trades if 'duration_hours' in t]
            metrics['avg_trade_duration_hours'] = sum(durations) / len(durations) if durations else 0

            # Average profit/loss per trade
            metrics['avg_profit_per_trade'] = sum(t.get('profit_loss', 0) for t in closed_trades) / len(
                closed_trades)

            # Average win and loss
            metrics['avg_win'] = sum(t.get('profit_loss', 0) for t in profit_trades) / len(
                profit_trades) if profit_trades else 0
            metrics['avg_loss'] = sum(t.get('profit_loss', 0) for t in loss_trades) / len(
                loss_trades) if loss_trades else 0

            # Largest win and loss
            metrics['largest_win'] = max(t.get('profit_loss', 0) for t in profit_trades) if profit_trades else 0
            metrics['largest_loss'] = min(t.get('profit_loss', 0) for t in loss_trades) if loss_trades else 0
    else:
        # Default values if no closed trades
        metrics.update({
            'n_winning_trades': 0,
            'n_losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'avg_trade_duration_hours': 0,
            'avg_profit_per_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        })

    # Log results
    logger.info(f"Backtest Results:")
    logger.info(f"  Initial Balance: ${metrics['initial_balance']:.2f}")
    logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
    logger.info(f"  Return: {metrics['return_pct']:.2f}%")
    logger.info(f"  Number of Trades: {metrics['n_trades']}")

    if metrics['n_closed_trades'] > 0:
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Avg Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours")

    # Save backtest results
    storage = DataStorage()
    model_name = os.path.basename(model_path).replace('.joblib', '')
    backtest_results = {
        'metrics': metrics,
        'trades': trades,
        'strategy_stats': strategy.get_stats(),
        'balance_curve': test_results[['balance']].copy(),
        'signals': test_results[['signal', 'trade_action']].copy() if 'signal' in test_results.columns else None
    }

    results_path = storage.save_results(
        backtest_results,
        f"{model_name}_backtest_{timeframe}",
        include_timestamp=True
    )
    logger.info(f"Saved backtest results to {results_path}")

    return metrics, test_results, trades


@dataclass
class BacktestResult:
    """Data class to hold backtest results."""
    metrics: Dict[str, Any]
    results_df: pd.DataFrame
    trades: List[Dict[str, Any]]
    strategy_stats: Dict[str, Any]
    execution_time: float


class Backtest:
    """Backtesting engine for evaluating trading strategies with ML models."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = setup_logger("Backtest")
        self.logger.info("Initializing backtest engine")

        # Extract backtest config parameters
        self.include_spread = config.get('backtest', {}).get('include_spread', True)
        self.include_slippage = config.get('backtest', {}).get('include_slippage', True)
        self.initial_balance = config.get('backtest', {}).get('initial_balance', 10000.0)

        # Trading costs
        self.spread_points = config.get('risk', {}).get('spread', 0.5)
        self.slippage_points = config.get('risk', {}).get('slippage', 0.1)
        self.point_value = config.get('backtest', {}).get('point_value', 0.01)  # For XAU/USD

        # Apply spread/slippage based on config
        self.spread_cost = self.spread_points * self.point_value if self.include_spread else 0
        self.slippage_cost = self.slippage_points * self.point_value if self.include_slippage else 0

        # Store all results
        self.results = {}

        # Data storage reference
        self.storage = DataStorage()

        self.logger.debug(
            f"Backtest config: initial_balance={self.initial_balance}, "
            f"spread={self.spread_cost}, slippage={self.slippage_cost}"
        )

    def run(self, model_path: str, strategy_name: str, timeframe: str) -> BacktestResult:
        start_time = time.time()
        self.logger.info(f"Starting backtest for model: {model_path}")
        self.logger.info(f"Strategy: {strategy_name}, Timeframe: {timeframe}")

        # Create strategy
        strategy = StrategyFactory.create_strategy(strategy_name, self.config)
        self.logger.info(f"Created strategy: {strategy.name}")

        # Load and prepare data
        model, test_data, model_predictions = self._load_and_prepare_data(model_path, timeframe)
        if model is None or test_data is None:
            self.logger.error("Failed to load model or data. Aborting backtest.")
            return BacktestResult({}, pd.DataFrame(), [], {}, 0.0)

        # Generate signals using the strategy
        test_results = strategy.generate_signals(test_data, model_predictions)

        # Run simulation
        metrics, results_df, trades = self._run_simulation(test_results, strategy)

        # Get strategy statistics
        strategy_stats = strategy.get_stats()

        execution_time = time.time() - start_time
        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")

        # Create result object
        result = BacktestResult(
            metrics=metrics,
            results_df=results_df,
            trades=trades,
            strategy_stats=strategy_stats,
            execution_time=execution_time
        )

        # Save results
        model_name = os.path.basename(model_path).replace('.joblib', '')
        result_key = f"{model_name}_{strategy_name}_{timeframe}"
        self.results[result_key] = result

        # Save to disk
        self._save_results(result, model_name, strategy_name, timeframe)

        return result

    def _load_and_prepare_data(self, model_path: str, timeframe: str) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
        try:
            model = ModelFactory.load_model(model_path)
            self.logger.info(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None, None, None

        # Load model metadata
        model_metadata = self._load_model_metadata(model_path)

        # Get prediction horizon
        horizon = self._get_prediction_horizon(model_path, model_metadata)
        self.logger.info(f"Using prediction horizon: {horizon}")

        # Load test data
        test_data = self._load_test_data(timeframe)
        if test_data is None:
            return None, None, None

        # Prepare features and get model predictions
        X_test, model_predictions = self._prepare_features_and_predict(model, test_data, model_metadata, horizon)
        if X_test is None:
            return None, None, None

        # Return only the test data rows that match our feature timeframe
        test_data = test_data.loc[X_test.index].copy()

        return model, test_data, model_predictions

    def _load_model_metadata(self, model_path: str) -> Dict:
        """Load model metadata if available."""
        model_metadata = {}
        metadata_path = model_path.replace(".joblib", "_metadata.pkl")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "rb") as f:
                    model_metadata = pickle.load(f)
                self.logger.info(f"Loaded model metadata: {list(model_metadata.keys())}")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {str(e)}")

        return model_metadata

    def _get_prediction_horizon(self, model_path: str, model_metadata: Dict) -> int:
        """Determine prediction horizon from model path or metadata."""
        horizon = 1  # Default value

        # First try to extract from filename
        model_filename = os.path.basename(model_path)
        if '_' in model_filename:
            parts = model_filename.split('_')
            for part in parts:
                if part.isdigit():
                    horizon = int(part)
                    break

        # If not found in filename, try metadata
        if horizon == 1 and model_metadata and 'prediction_horizon' in model_metadata:
            horizon = model_metadata['prediction_horizon']

        return horizon

    def _load_test_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Load test data for the specified timeframe."""
        latest_processed_data = self.storage.find_latest_processed_data()

        if timeframe not in latest_processed_data:
            self.logger.error(f"No processed data found for timeframe {timeframe}")
            return None

        try:
            df = pd.read_csv(latest_processed_data[timeframe], index_col=0, parse_dates=True)
            self.logger.info(f"Loaded data shape: {df.shape}")

            # Use the last 20% of data for testing
            test_data = df.iloc[int(len(df) * 0.8):].copy()
            self.logger.info(f"Test data shape: {test_data.shape}")

            return test_data

        except Exception as e:
            self.logger.error(f"Error loading test data: {str(e)}")
            return None

    def _prepare_features_and_predict(self, model, test_data: pd.DataFrame,
                                      model_metadata: Dict, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and generate predictions."""
        # Process features
        processor = DataProcessor()
        X_test, y_test = processor.prepare_ml_features(test_data, horizon=horizon)
        self.logger.info(f"Prepared features shape: {X_test.shape}, target shape: {y_test.shape}")

        # Filter features to match what the model was trained on
        X_test = self._filter_features(X_test, model, model_metadata)
        if X_test is None:
            return None, None

        # Generate predictions
        model_predictions = pd.DataFrame(index=X_test.index)

        try:
            is_classifier = hasattr(model, 'predict_proba')

            if is_classifier:
                y_pred_proba = model.predict_proba(X_test)

                if y_pred_proba.shape[1] == 2:
                    model_predictions['pred_probability_up'] = y_pred_proba[:, 1]
                    model_predictions['pred_probability_down'] = 1 - y_pred_proba[:, 1]

                    # Log distribution statistics
                    self._log_prediction_distribution(model_predictions['pred_probability_up'])
                else:
                    self.logger.warning(f"Unexpected prediction shape: {y_pred_proba.shape}")
            else:
                y_pred = model.predict(X_test)
                model_predictions['predicted_value'] = y_pred
                self.logger.info(
                    f"Prediction statistics: min={y_pred.min():.4f}, "
                    f"max={y_pred.max():.4f}, mean={y_pred.mean():.4f}"
                )

            return X_test, model_predictions

        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return None, None

    def _filter_features(self, X_test: pd.DataFrame, model, model_metadata: Dict) -> Optional[pd.DataFrame]:
        """Filter features to match what the model was trained on."""
        try:
            if model_metadata.get('features'):
                fitted_features = model_metadata['features']
                self.logger.info(f"Filtering test features to fitted features from metadata")

                # Warning for extra or missing features
                extra_features = [f for f in X_test.columns if f not in fitted_features]
                if extra_features:
                    self.logger.warning(f"Extra features in test data (will be dropped): {extra_features}")

                missing_features = [f for f in fitted_features if f not in X_test.columns]
                if missing_features:
                    self.logger.error(f"Missing features in test data: {missing_features}")
                    return None

                # Filter to only include the fitted features
                return X_test[fitted_features]

            elif hasattr(model, 'get_feature_names'):
                model_features = model.get_feature_names()
                self.logger.info(f"Filtering test features to fitted features from model")

                extra_features = [f for f in X_test.columns if f not in model_features]
                if extra_features:
                    self.logger.warning(f"Extra features in test data (will be dropped): {extra_features}")

                missing_features = [f for f in model_features if f not in X_test.columns]
                if missing_features:
                    self.logger.error(f"Missing features in test data: {missing_features}")
                    return None

                return X_test[model_features]
            else:
                self.logger.warning("No feature metadata available, using all features")
                return X_test

        except Exception as e:
            self.logger.error(f"Error filtering features: {str(e)}")
            return None

    def _log_prediction_distribution(self, predictions: pd.Series) -> None:
        """Log distribution of prediction probabilities."""
        prob_bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(predictions, bins=prob_bins)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

        self.logger.info("Prediction probability distribution:")
        for center, count in zip(bin_centers, hist):
            percentage = count / len(predictions) * 100 if len(predictions) > 0 else 0
            self.logger.info(f"  {center:.1f}: {count} predictions ({percentage:.1f}%)")

        self.logger.info(f"Min probability: {predictions.min():.4f}")
        self.logger.info(f"Max probability: {predictions.max():.4f}")
        self.logger.info(f"Mean probability: {predictions.mean():.4f}")

    def _run_simulation(self, test_results: pd.DataFrame,
                        strategy: Strategy) -> Tuple[Dict[str, Any], pd.DataFrame, List[Dict]]:
        """
        Run trading simulation on the test data using the specified strategy.

        Args:
            test_results: DataFrame with test data and signals
            strategy: Strategy instance to use for the simulation

        Returns:
            Tuple of (metrics dictionary, results dataframe, trades list)
        """
        self.logger.info("Running trading simulation...")

        # Initialize simulation variables
        balance = self.initial_balance
        position = 0  # 0: no position, 1: long, -1: short
        position_size = 0.0
        entry_price = 0.0
        entry_time = None
        highest_price = 0.0
        lowest_price = float('inf')
        stop_loss_price = 0.0
        take_profit_price = 0.0
        position_data = {}
        trades = []

        # Initialize tracking columns
        test_results['balance'] = self.initial_balance
        test_results['position'] = 0
        test_results['entry_price'] = None
        test_results['stop_loss'] = None
        test_results['take_profit'] = None
        test_results['profit_loss'] = 0.0
        test_results['cum_profit_loss'] = 0.0
        test_results['position_duration'] = 0.0

        # Log simulation parameters
        self.logger.info(
            f"Simulation parameters: initial_balance={self.initial_balance}, "
            f"spread={self.spread_cost}, slippage={self.slippage_cost}"
        )

        # Run simulation bar by bar
        for i, (idx, row) in enumerate(test_results.iterrows()):
            current_price = row['close']
            current_signal = row['signal']

            # Update highest/lowest prices if in a position
            if position == 1:  # Long position
                highest_price = max(highest_price, current_price)
                position_data['highest_price'] = highest_price
            elif position == -1:  # Short position
                lowest_price = min(lowest_price, current_price)
                position_data['lowest_price'] = lowest_price

            # Calculate position duration if in a position
            if position != 0 and entry_time is not None:
                position_duration = (idx - entry_time).total_seconds() / 3600  # hours
                test_results.at[idx, 'position_duration'] = position_duration
                position_data['duration'] = position_duration

            # Update position data with current information
            position_data['current_signal'] = current_signal
            position_data['current_price'] = current_price

            # Determine action based on current position and strategy
            action = "HOLD"
            exit_reason = None

            if position == 0:  # No position - check for entry
                if current_signal == "BUY":
                    action = "BUY"
                elif current_signal == "SELL":
                    action = "SELL"
            else:  # In a position - check for exit
                should_exit, reason = strategy.should_exit_position(
                    position, entry_price, current_price, entry_time, idx, position_data
                )

                if should_exit:
                    action = "CLOSE"
                    exit_reason = reason.value if reason else "unknown"
                else:
                    # Update stop loss and take profit
                    stop_loss_price, take_profit_price = strategy.update_stops(
                        position, entry_price, current_price, highest_price, lowest_price,
                        stop_loss_price, take_profit_price
                    )

                    test_results.at[idx, 'stop_loss'] = stop_loss_price
                    test_results.at[idx, 'take_profit'] = take_profit_price

            # Execute trade action
            if action == "BUY":
                # Calculate entry price with costs
                adjusted_price = current_price + self.slippage_cost

                # Calculate position size
                stop_loss_value = adjusted_price * (1 - strategy.stop_loss_pct / 100)
                position_size = strategy.calculate_position_size(
                    balance, adjusted_price, None, stop_loss_value
                )

                # Set up position tracking
                position = 1
                entry_price = adjusted_price
                entry_time = idx
                highest_price = adjusted_price
                stop_loss_price = stop_loss_value
                take_profit_price = adjusted_price * (1 + strategy.take_profit_pct / 100)

                # Record trade
                trade = {
                    'time': idx,
                    'action': 'BUY',
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'balance': balance
                }
                trades.append(trade)

                self.logger.debug(
                    f"BUY at {idx}: price={entry_price:.2f}, size={position_size:.4f}, "
                    f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}"
                )

            elif action == "SELL":
                # Calculate entry price with costs
                adjusted_price = current_price - self.slippage_cost

                # Calculate position size
                stop_loss_value = adjusted_price * (1 + strategy.stop_loss_pct / 100)
                position_size = strategy.calculate_position_size(
                    balance, adjusted_price, None, stop_loss_value
                )

                # Set up position tracking
                position = -1
                entry_price = adjusted_price
                entry_time = idx
                lowest_price = adjusted_price
                stop_loss_price = stop_loss_value
                take_profit_price = adjusted_price * (1 - strategy.take_profit_pct / 100)

                # Record trade
                trade = {
                    'time': idx,
                    'action': 'SELL',
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'balance': balance
                }
                trades.append(trade)

                self.logger.debug(
                    f"SELL at {idx}: price={entry_price:.2f}, size={position_size:.4f}, "
                    f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}"
                )

            elif action == "CLOSE":
                # Calculate exit price with costs
                if position == 1:  # Long position
                    exit_price = current_price - self.slippage_cost
                    profit_loss = (exit_price - entry_price) * position_size
                    # Deduct spread cost
                    profit_loss -= self.spread_cost * position_size
                else:  # Short position
                    exit_price = current_price + self.slippage_cost
                    profit_loss = (entry_price - exit_price) * position_size
                    # Deduct spread cost
                    profit_loss -= self.spread_cost * position_size

                # Apply stop-loss risk limit
                max_loss = -self.initial_balance * 0.05  # Limit loss to 5% per trade
                if profit_loss < max_loss:
                    self.logger.warning(
                        f"Limiting loss at {idx} from {profit_loss:.2f} to {max_loss:.2f}"
                    )
                    profit_loss = max_loss

                # Update account balance
                balance += profit_loss
                test_results.at[idx, 'profit_loss'] = profit_loss

                # Calculate position duration
                position_duration = (idx - entry_time).total_seconds() / 3600 if entry_time else 0

                # Record trade
                trade = {
                    'time': idx,
                    'action': 'CLOSE',
                    'reason': exit_reason,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'profit_loss': profit_loss,
                    'balance': balance,
                    'duration_hours': position_duration
                }
                trades.append(trade)

                self.logger.debug(
                    f"CLOSE at {idx}: reason={exit_reason}, price={exit_price:.2f}, "
                    f"P/L={profit_loss:.2f}, duration={position_duration:.1f}h"
                )

                # Reset position tracking
                position = 0
                position_size = 0
                entry_price = 0
                entry_time = None
                highest_price = 0
                lowest_price = float('inf')
                stop_loss_price = 0
                take_profit_price = 0
                position_data = {}

            # Update tracking columns
            test_results.at[idx, 'balance'] = balance
            test_results.at[idx, 'position'] = position
            test_results.at[idx, 'entry_price'] = entry_price if position != 0 else None

        # Close any open position at the end of the test
        if position != 0:
            last_idx = test_results.index[-1]
            last_price = test_results.loc[last_idx, 'close']

            # Calculate exit price with costs
            if position == 1:  # Long position
                exit_price = last_price - self.slippage_cost
                profit_loss = (exit_price - entry_price) * position_size
                profit_loss -= self.spread_cost * position_size
            else:  # Short position
                exit_price = last_price + self.slippage_cost
                profit_loss = (entry_price - exit_price) * position_size
                profit_loss -= self.spread_cost * position_size

            # Calculate position duration
            position_duration = (last_idx - entry_time).total_seconds() / 3600 if entry_time else 0

            self.logger.info(
                f"Closing final position at {last_idx}: position={position}, "
                f"entry_price={entry_price:.2f}, exit_price={exit_price:.2f}, "
                f"profit_loss={profit_loss:.2f}, duration={position_duration:.1f}h"
            )

            # Add to balance and update results
            balance += profit_loss
            test_results.at[last_idx, 'profit_loss'] = profit_loss
            test_results.at[last_idx, 'balance'] = balance

            # Record trade
            trades.append({
                'time': last_idx,
                'action': 'CLOSE_FINAL',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'profit_loss': profit_loss,
                'balance': balance,
                'duration_hours': position_duration
            })

        # Calculate cumulative profit/loss
        test_results['cum_profit_loss'] = test_results['profit_loss'].fillna(0).cumsum()

        # Calculate basic metrics
        metrics = self._calculate_performance_metrics(test_results, trades)

        return metrics, test_results, trades

    def _calculate_performance_metrics(self, results: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results."""
        self.logger.info("Calculating performance metrics...")

        metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': results['balance'].iloc[-1],
            'absolute_return': results['balance'].iloc[-1] - self.initial_balance,
            'return_pct': ((results['balance'].iloc[-1] / self.initial_balance) - 1) * 100,
            'n_trades': len([t for t in trades if t['action'] in ['BUY', 'SELL']]),
            'n_closed_trades': len([t for t in trades if t['action'] == 'CLOSE']),
        }

        if metrics['n_closed_trades'] > 0:
            # Extract closed trades with P/L
            closed_trades = [t for t in trades if 'profit_loss' in t and t.get('profit_loss', 0) != 0]

            # Calculate winning/losing trades
            profit_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
            loss_trades = [t for t in closed_trades if t.get('profit_loss', 0) < 0]

            metrics['n_winning_trades'] = len(profit_trades)
            metrics['n_losing_trades'] = len(loss_trades)
            metrics['win_rate'] = len(profit_trades) / len(closed_trades) if closed_trades else 0

            # Calculate profit factor
            total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
            total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

            # Calculate max drawdown
            peak = self.initial_balance
            max_drawdown = 0

            for balance in results['balance']:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)

            metrics['max_drawdown_pct'] = max_drawdown

            # Calculate Sharpe ratio (annualized)
            daily_returns = results['profit_loss'] / results['balance'].shift(1)
            daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

            if len(daily_returns) > 1:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
                metrics['sharpe_ratio'] = sharpe_ratio
            else:
                metrics['sharpe_ratio'] = 0

            # Additional metrics
            if closed_trades:
                # Average trade duration
                durations = [t.get('duration_hours', 0) for t in closed_trades if 'duration_hours' in t]
                metrics['avg_trade_duration_hours'] = sum(durations) / len(durations) if durations else 0

                # Average profit/loss per trade
                metrics['avg_profit_per_trade'] = sum(t.get('profit_loss', 0) for t in closed_trades) / len(
                    closed_trades)

                # Average win and loss
                metrics['avg_win'] = sum(t.get('profit_loss', 0) for t in profit_trades) / len(
                    profit_trades) if profit_trades else 0
                metrics['avg_loss'] = sum(t.get('profit_loss', 0) for t in loss_trades) / len(
                    loss_trades) if loss_trades else 0

                # Largest win and loss
                metrics['largest_win'] = max(t.get('profit_loss', 0) for t in profit_trades) if profit_trades else 0
                metrics['largest_loss'] = min(t.get('profit_loss', 0) for t in loss_trades) if loss_trades else 0
        else:
            # Default values if no closed trades
            metrics.update({
                'n_winning_trades': 0,
                'n_losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration_hours': 0,
                'avg_profit_per_trade': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            })

        self.logger.info(f"Backtest Results:")
        self.logger.info(f"  Initial Balance: ${metrics['initial_balance']:.2f}")
        self.logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
        self.logger.info(f"  Return: {metrics['return_pct']:.2f}%")
        self.logger.info(f"  Number of Trades: {metrics['n_trades']}")

        if metrics['n_closed_trades'] > 0:
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
            self.logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            self.logger.info(f"  Avg Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours")

        return metrics

    def _calculate_additional_metrics(self, metrics: Dict, results_df: pd.DataFrame, trades: List[Dict]) -> None:
        """Calculate additional performance metrics and statistics."""
        # Exit reason distribution
        if 'reason' in trades[0] if trades else False:
            reason_counts = {}
            for trade in trades:
                if 'reason' in trade and trade['reason']:
                    reason = trade['reason']
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

            metrics['exit_reasons'] = reason_counts
            self.logger.info(f"  Exit reasons: {reason_counts}")

        # Consecutive wins/losses
        if trades:
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0

            for trade in trades:
                if 'profit_loss' not in trade:
                    continue

                pl = trade.get('profit_loss', 0)

                if pl > 0:  # Winning trade
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    max_win_streak = max(max_win_streak, current_streak)
                elif pl < 0:  # Losing trade
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))

            metrics['max_win_streak'] = max_win_streak
            metrics['max_loss_streak'] = max_loss_streak

            # Monthly returns if data spans multiple months
            if len(results_df) > 0:
                # Resample to monthly and calculate returns
                monthly_returns = results_df.resample('M')['balance'].last().pct_change()
                if len(monthly_returns) > 1:
                    metrics['monthly_returns'] = monthly_returns.dropna().to_dict()
                    metrics['avg_monthly_return'] = monthly_returns.mean()
                    metrics['best_month_return'] = monthly_returns.max()
                    metrics['worst_month_return'] = monthly_returns.min()

    def _save_results(self, result: BacktestResult, model_name: str, strategy_name: str, timeframe: str) -> None:
        """Save backtest results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{strategy_name}_{timeframe}_backtest_{timestamp}"

            # Save metrics and trades to pickle
            backtest_data = {
                'metrics': result.metrics,
                'trades': result.trades,
                'strategy_stats': result.strategy_stats,
                'execution_time': result.execution_time
            }

            results_path = self.storage.save_results(backtest_data, filename)
            self.logger.info(f"Saved backtest results to {results_path}")

            # Save results DataFrame to CSV for further analysis
            csv_path = os.path.join(os.path.dirname(results_path), f"{filename}_results.csv")
            result.results_df.to_csv(csv_path)
            self.logger.info(f"Saved detailed results to {csv_path}")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")

    def plot_equity_curve(self, result: BacktestResult, title: str = None) -> Figure:
        """
        Plot equity curve and drawdown from backtest results.

        Args:
            result: BacktestResult object
            title: Optional title for the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = "Equity Curve"

        results = result.results_df

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot equity curve
        ax1.plot(results.index, results['balance'], label='Balance', color='blue')

        # Add trade markers
        for trade in result.trades:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['time'], trade['balance'], marker='^', color='green', s=100)
            elif trade['action'] == 'SELL':
                ax1.scatter(trade['time'], trade['balance'], marker='v', color='red', s=100)

        ax1.set_title(title)
        ax1.set_ylabel('Account Value')
        ax1.legend()
        ax1.grid(True)

        # Plot drawdown
        equity = results['balance']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100

        ax2.fill_between(results.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def plot_trade_distribution(self, result: BacktestResult, title: str = None) -> Figure:
        """
        Plot trade duration and profit distribution.

        Args:
            result: BacktestResult object
            title: Optional title for the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = "Trade Analysis"

        # Filter trades to only include those with profit/loss
        trades = [t for t in result.trades if 'profit_loss' in t and 'duration_hours' in t]

        if not trades:
            self.logger.warning("No closed trades to plot distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No closed trades to analyze",
                    horizontalalignment='center', verticalalignment='center')
            return fig

        # Extract data
        profits = [t['profit_loss'] for t in trades]
        durations = [t['duration_hours'] for t in trades]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot profit distribution
        ax1.hist(profits, bins=20, color='skyblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_title('Profit/Loss Distribution')
        ax1.set_xlabel('Profit/Loss')
        ax1.set_ylabel('Frequency')

        # Plot duration distribution
        ax2.hist(durations, bins=20, color='lightgreen', edgecolor='black')
        ax2.set_title('Trade Duration Distribution')
        ax2.set_xlabel('Duration (hours)')
        ax2.set_ylabel('Frequency')

        plt.suptitle(title)
        plt.tight_layout()
        return fig

    def compare_strategies(self, model_path: str, strategy_names: List[str],
                           timeframe: str) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies on the same model and timeframe.

        Args:
            model_path: Path to the model file
            strategy_names: List of strategy names to compare
            timeframe: Timeframe to use

        Returns:
            Dictionary of strategy_name -> BacktestResult
        """
        self.logger.info(f"Comparing {len(strategy_names)} strategies on {model_path}")

        results = {}
        for strategy_name in strategy_names:
            self.logger.info(f"Running backtest with strategy: {strategy_name}")
            result = self.run(model_path, strategy_name, timeframe)
            results[strategy_name] = result

        # Log comparison summary
        self.logger.info("\nStrategy Comparison Summary:")
        self.logger.info("-" * 80)
        self.logger.info(
            f"{'Strategy':<20} {'Return %':<10} {'Win Rate':<10} {'# Trades':<10} {'Profit Factor':<15} {'Max DD %':<10}")
        self.logger.info("-" * 80)

        for name, result in results.items():
            metrics = result.metrics
            self.logger.info(
                f"{name:<20} "
                f"{metrics['return_pct']:<10.2f} "
                f"{metrics.get('win_rate', 0):<10.2f} "
                f"{metrics['n_trades']:<10} "
                f"{metrics.get('profit_factor', 0):<15.2f} "
                f"{metrics.get('max_drawdown_pct', 0):<10.2f}"
            )

        self.logger.info("-" * 80)
        return results

    def optimize_strategy(self, model_path: str, strategy_name: str, timeframe: str,
                          param_grid: Dict[str, List]) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters using grid search.

        Args:
            model_path: Path to the model file
            strategy_name: Strategy name to optimize
            timeframe: Timeframe to use
            param_grid: Dictionary of parameter names -> list of values to test

        Returns:
            Tuple of (best_params, best_result)
        """
        self.logger.info(f"Optimizing strategy {strategy_name} on {model_path}")
        self.logger.info(f"Parameter grid: {param_grid}")

        # Generate all parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")

        best_result = None
        best_params = None
        best_metric = -float('inf')  # For maximizing return

        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            params = {key: value for key, value in zip(param_keys, combination)}

            # Update config with new parameters
            temp_config = self.config.copy()
            for key, value in params.items():
                key_parts = key.split('.')

                # Navigate to the right level in the config
                current = temp_config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value
                current[key_parts[-1]] = value

            # Create strategy with updated parameters
            strategy = StrategyFactory.create_strategy(strategy_name, temp_config)

            # Run backtest
            self.logger.info(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")
            result = self.run(model_path, strategy_name, timeframe)

            # Check if this is the best result
            current_metric = result.metrics['return_pct']
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params
                best_result = result

                self.logger.info(
                    f"New best result: Return={best_metric:.2f}%, "
                    f"Win Rate={result.metrics.get('win_rate', 0):.2f}, "
                    f"Trades={result.metrics['n_trades']}"
                )

        self.logger.info(f"Optimization complete.")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(
            f"Best result: Return={best_result.metrics['return_pct']:.2f}%, "
            f"Win Rate={best_result.metrics.get('win_rate', 0):.2f}, "
            f"Profit Factor={best_result.metrics.get('profit_factor', 0):.2f}"
        )

        return best_params, best_result

    def run_backtest(config_path: str, model_path: str, strategy_name: str, timeframe: str) -> None:
        """
        Run a backtest from command line.

        Args:
            config_path: Path to config file
            model_path: Path to model file
            strategy_name: Name of strategy to use
            timeframe: Timeframe to use
        """
        # Load config
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create backtest engine
        backtest = Backtest(config)

        # Run backtest
        result = backtest.run(model_path, strategy_name, timeframe)

        # Create and save plots
        fig1 = backtest.plot_equity_curve(result)
        fig2 = backtest.plot_trade_distribution(result)

        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).replace('.joblib', '')

        fig1.savefig(f"{model_name}_{strategy_name}_{timeframe}_equity_{timestamp}.png")
        fig2.savefig(f"{model_name}_{strategy_name}_{timeframe}_trades_{timestamp}.png")

        # Print summary
        print("\n" + "=" * 50)
        print("Backtest Summary")
        print("=" * 50)
        print(f"Model: {model_path}")
        print(f"Strategy: {strategy_name}")
        print(f"Timeframe: {timeframe}")
        print(f"Initial Balance: ${result.metrics['initial_balance']:.2f}")
        print(f"Final Balance: ${result.metrics['final_balance']:.2f}")
        print(f"Total Return: {result.metrics['return_pct']:.2f}%")
        print(f"Number of Trades: {result.metrics['n_trades']}")

        if result.metrics.get('win_rate') is not None:
            print(f"Win Rate: {result.metrics['win_rate']:.2f}")
            print(f"Profit Factor: {result.metrics.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {result.metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")

        print("=" * 50)
