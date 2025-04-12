import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from config.constants import PredictionTarget, TradeAction
from data.processor import DataProcessor
from data.storage import DataStorage
from models.base import BaseModel
from utils.logger import setup_logger

# Set up a module-level logger for the evaluator/backtest module
logger = setup_logger("BacktestEvaluator")


class ModelEvaluator:
    """Evaluate model performance and generate trading signals."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.prediction_horizon = config.get('model', {}).get('prediction_horizon', 12)
        self.confidence_threshold = config.get('strategy', {}).get('min_confidence', 0.65)
        self.is_classifier = self.target_type in [
            PredictionTarget.DIRECTION.value,
            PredictionTarget.VOLATILITY.value
        ]
        # Use the module-level logger for class-level logging
        self.logger = logger
        self.logger.debug(
            "Initialized ModelEvaluator with target_type: %s, prediction_horizon: %d, confidence_threshold: %.2f",
            self.target_type, self.prediction_horizon, self.confidence_threshold)

    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model and return performance metrics."""
        self.logger.info("Evaluating model performance.")
        self.logger.debug("Evaluation features: %s", X.columns.tolist())
        self.logger.debug("Evaluation data shape: %s", X.shape)
        y_pred = model.predict(X)
        metrics = {}

        # Remove NaN values
        mask = ~np.isnan(y_pred)
        if not all(mask):
            y_pred = y_pred[mask]
            y_true = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
        else:
            y_true = y

        if self.is_classifier:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
            self.logger.debug("Classifier metrics computed: %s", metrics)

            try:
                y_proba = model.predict_proba(X)
                if mask is not None and not all(mask):
                    y_proba = y_proba[mask]
                metrics['mean_probability'] = np.mean(np.max(y_proba, axis=1))
                confidence_bins = np.linspace(0.5, 1, 6)
                confidence_hist = np.histogram(np.max(y_proba, axis=1), bins=confidence_bins)[0]
                confidence_hist = confidence_hist / confidence_hist.sum()
                metrics['confidence_distribution'] = dict(zip(
                    [f"{b:.1f}" for b in confidence_bins[:-1]],
                    confidence_hist
                ))
                self.logger.debug("Added probability metrics: %s", metrics.get('confidence_distribution'))
            except (AttributeError, ValueError) as e:
                self.logger.warning("Failed to compute prediction probabilities: %s", str(e))
        else:
            metrics = {
                'mae': np.mean(np.abs(y_true - y_pred)),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mean_error': np.mean(y_true - y_pred),
                'std_error': np.std(y_true - y_pred)
            }
            self.logger.debug("Regression metrics computed: %s", metrics)

        return metrics

    def generate_signals(self, model: BaseModel, X: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from model predictions."""
        self.logger.info("Generating trading signals.")
        results = price_data.copy()

        # Get predictions and add to results
        predictions = model.predict(X)
        results['prediction'] = np.nan
        valid_idx = ~np.isnan(predictions)
        results.loc[X.index[valid_idx], 'prediction'] = predictions[valid_idx]

        if self.is_classifier:
            try:
                probas = model.predict_proba(X)
                results['confidence'] = np.nan
                max_probas = np.max(probas, axis=1)
                results.loc[X.index[valid_idx], 'confidence'] = max_probas[valid_idx]

                results['signal'] = TradeAction.HOLD.value
                buy_mask = ((results['prediction'] == 1) & (results['confidence'] >= self.confidence_threshold))
                results.loc[buy_mask, 'signal'] = TradeAction.BUY.value
                sell_mask = ((results['prediction'] == 0) & (results['confidence'] >= self.confidence_threshold))
                results.loc[sell_mask, 'signal'] = TradeAction.SELL.value
                self.logger.debug("Signals generated for classifier.")
            except (AttributeError, ValueError) as e:
                self.logger.warning("Could not compute prediction probabilities for signals: %s", str(e))
                results['signal'] = TradeAction.HOLD.value
                results.loc[results['prediction'] == 1, 'signal'] = TradeAction.BUY.value
                results.loc[results['prediction'] == 0, 'signal'] = TradeAction.SELL.value
        else:
            results['signal'] = TradeAction.HOLD.value
            if self.target_type == PredictionTarget.PRICE.value:
                results.loc[results['prediction'] > results['close'], 'signal'] = TradeAction.BUY.value
                results.loc[results['prediction'] < results['close'], 'signal'] = TradeAction.SELL.value
            elif self.target_type == PredictionTarget.RETURN.value:
                buy_threshold = 0.005
                sell_threshold = -0.005
                results.loc[results['prediction'] > buy_threshold, 'signal'] = TradeAction.BUY.value
                results.loc[results['prediction'] < sell_threshold, 'signal'] = TradeAction.SELL.value
            self.logger.debug("Signals generated for regression model.")

        return results

    def plot_predictions(self, results: pd.DataFrame, title: str = "Model Predictions") -> None:
        """Plot price data with predictions and signals."""
        self.logger.info("Plotting predictions and signals.")
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['close'], label='Close Price')
        buy_signals = results[results['signal'] == TradeAction.BUY.value]
        plt.scatter(buy_signals.index, buy_signals['close'],
                    marker='^', color='green', label='Buy Signal', s=100)
        sell_signals = results[results['signal'] == TradeAction.SELL.value]
        plt.scatter(sell_signals.index, sell_signals['close'],
                    marker='v', color='red', label='Sell Signal', s=100)
        plt.title(title)
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        if self.is_classifier:
            plt.step(results.index, results['prediction'], label='Direction Prediction', color='blue')
            if 'confidence' in results.columns:
                plt.plot(results.index, results['confidence'], label='Confidence', color='purple', alpha=0.7)
                plt.axhline(y=self.confidence_threshold, color='grey', linestyle='--',
                            label=f'Confidence Threshold ({self.confidence_threshold})')
        else:
            plt.plot(results.index, results['prediction'], label='Prediction', color='blue')
            plt.plot(results.index, results['close'], label='Actual', color='green', alpha=0.5)
        plt.ylabel('Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calculate_trading_metrics(self, results: pd.DataFrame, initial_balance: float = 10000.0,
                                  position_size: float = 0.1, include_spread: bool = True,
                                  include_slippage: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
        """Calculate trading performance metrics from signals."""
        self.logger.info("Calculating trading metrics.")
        df = results.copy()
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['trade_return'] = 0.0
        df['balance'] = initial_balance
        df['equity'] = initial_balance

        spread_pips = self.config.get('risk', {}).get('spread_pips', 30)
        slippage_pips = self.config.get('risk', {}).get('slippage_pips', 5)
        spread_cost = spread_pips * 0.01 if include_spread else 0
        slippage_cost = slippage_pips * 0.01 if include_slippage else 0

        trades = []
        active_trade = None

        for i in range(1, len(df)):
            prev_idx = df.index[i - 1]
            curr_idx = df.index[i]
            signal = df.loc[curr_idx, 'signal']
            prev_position = df.loc[prev_idx, 'position']

            if signal == TradeAction.BUY.value and prev_position <= 0:
                if prev_position < 0:
                    exit_price = df.loc[curr_idx, 'open'] + slippage_cost
                    df.loc[curr_idx, 'exit_price'] = exit_price
                    entry_price = active_trade['entry_price']
                    trade_return = (entry_price - exit_price) / entry_price
                    trade_return -= spread_cost / entry_price
                    active_trade.update({
                        'exit_date': curr_idx,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'pnl': trade_return * active_trade['size']
                    })
                    trades.append(active_trade)
                    active_trade = None

                entry_price = df.loc[curr_idx, 'open'] + slippage_cost
                df.loc[curr_idx, 'position'] = 1
                df.loc[curr_idx, 'entry_price'] = entry_price
                trade_size = df.loc[prev_idx, 'balance'] * position_size
                active_trade = {
                    'type': 'long',
                    'entry_date': curr_idx,
                    'entry_price': entry_price,
                    'size': trade_size
                }

            elif signal == TradeAction.SELL.value and prev_position >= 0:
                if prev_position > 0:
                    exit_price = df.loc[curr_idx, 'open'] - slippage_cost
                    df.loc[curr_idx, 'exit_price'] = exit_price
                    entry_price = active_trade['entry_price']
                    trade_return = (exit_price - entry_price) / entry_price
                    trade_return -= spread_cost / entry_price
                    active_trade.update({
                        'exit_date': curr_idx,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'pnl': trade_return * active_trade['size']
                    })
                    trades.append(active_trade)
                    active_trade = None

                entry_price = df.loc[curr_idx, 'open'] - slippage_cost
                df.loc[curr_idx, 'position'] = -1
                df.loc[curr_idx, 'entry_price'] = entry_price
                trade_size = df.loc[prev_idx, 'balance'] * position_size
                active_trade = {
                    'type': 'short',
                    'entry_date': curr_idx,
                    'entry_price': entry_price,
                    'size': trade_size
                }

            if active_trade is None:
                df.loc[curr_idx, 'balance'] = df.loc[prev_idx, 'balance']
                df.loc[curr_idx, 'equity'] = df.loc[prev_idx, 'balance']
            else:
                df.loc[curr_idx, 'balance'] = df.loc[prev_idx, 'balance']
                curr_price = df.loc[curr_idx, 'close']
                if active_trade['type'] == 'long':
                    unrealized_return = (curr_price - active_trade['entry_price']) / active_trade['entry_price']
                else:
                    unrealized_return = (active_trade['entry_price'] - curr_price) / active_trade['entry_price']
                unrealized_pnl = unrealized_return * active_trade['size']
                df.loc[curr_idx, 'equity'] = df.loc[prev_idx, 'balance'] + unrealized_pnl

            if i > 0 and not np.isnan(df.loc[curr_idx, 'exit_price']):
                last_trade = trades[-1]
                df.loc[curr_idx:, 'balance'] = df.loc[prev_idx, 'balance'] + last_trade['pnl']

        metrics = {}
        if trades:
            trades_df = pd.DataFrame(trades)
            total_pnl = trades_df['pnl'].sum()
            metrics['total_pnl'] = total_pnl
            metrics['return_pct'] = (total_pnl / initial_balance) * 100
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            metrics['win_rate'] = win_rate
            metrics['avg_trade_return'] = trades_df['return'].mean()
            metrics['avg_trade_pnl'] = trades_df['pnl'].mean()
            if len(winning_trades) > 0 and len(trades_df) - len(winning_trades) > 0:
                avg_win = winning_trades['pnl'].mean()
                losing_trades = trades_df[trades_df['pnl'] <= 0]
                avg_loss = losing_trades['pnl'].mean()
                metrics['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                metrics['profit_factor'] = float('inf') if win_rate == 1 else 0

            equity_curve = df['equity']
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            metrics['max_drawdown'] = max_drawdown
            metrics['max_drawdown_pct'] = max_drawdown * 100
            metrics['n_trades'] = len(trades_df)
            metrics['n_winning_trades'] = len(winning_trades)
            metrics['n_losing_trades'] = len(trades_df) - len(winning_trades)
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(df['balance'].pct_change().dropna())
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(df['balance'].pct_change().dropna())
            metrics['final_balance'] = df['balance'].iloc[-1]
            self.logger.debug("Trading metrics calculated: %s", metrics)
        else:
            metrics = {
                'total_pnl': 0,
                'return_pct': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'avg_trade_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'n_trades': 0,
                'n_winning_trades': 0,
                'n_losing_trades': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'final_balance': initial_balance
            }
            self.logger.debug("No trades executed; metrics set to default values.")
        return metrics, df, trades

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio from returns."""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0
        if downside_deviation == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
        return excess_returns.mean() / downside_deviation * np.sqrt(252)

    def plot_equity_curve(self, results: pd.DataFrame, title: str = "Equity Curve") -> None:
        """Plot equity curve and drawdown."""
        self.logger.info("Plotting equity curve.")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(results.index, results['balance'], label='Balance', color='blue')
        ax1.plot(results.index, results['equity'], label='Equity', color='green', alpha=0.7)
        ax1.set_title(title)
        ax1.set_ylabel('Account Value')
        ax1.legend()
        ax1.grid(True)
        equity = results['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        ax2.fill_between(results.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        plt.tight_layout()
        plt.show()


def get_model_by_type(model_type: str, config: Dict) -> Any:
    """Factory to return model instance based on type."""
    if model_type == 'random_forest':
        from models.random_forest import RandomForestModel
        return RandomForestModel(config)
    elif model_type == 'xgboost':
        from models.xgboost_model import XGBoostModel
        return XGBoostModel(config)
    elif model_type == 'lstm':
        from models.lstm_model import LSTMModel
        return LSTMModel(config)
    else:
        from models.ensemble_model import EnsembleModel
        return EnsembleModel(config)


def filter_features(X, exact_features: list) -> Tuple[Any, list]:
    """Filter test features to match the training set."""
    missing = [f for f in exact_features if f not in X.columns]
    available = [f for f in exact_features if f in X.columns]
    if missing:
        logger.warning("Missing %d features from test data: %s", len(missing), missing[:10])
    return X[available], available


def backtest_model(config: Dict, model_path: str, timeframe: str, initial_balance: float = 10000,
                   position_size: float = 0.1, include_spread: bool = True, include_slippage: bool = True) -> Tuple[
    Dict[str, Any], Any, Any]:
    logger.info("Starting backtest for model: %s", model_path)

    # Initialize required objects
    storage = DataStorage(config_path="config/config.yaml")
    processor = DataProcessor(config_path="config/config.yaml")

    # Load the model
    model_type = config.get('model', {}).get('type', 'ensemble')
    model = get_model_by_type(model_type, config)
    model.load(model_path)
    logger.info("Loaded model of type '%s'", model_type)

    # Load processed data
    processed_files = storage.find_latest_processed_data()
    if timeframe not in processed_files:
        logger.error("Timeframe %s not found in processed files", timeframe)
        raise ValueError(f"No processed data found for timeframe {timeframe}")

    data = processor.load_data({timeframe: processed_files[timeframe]})[timeframe]
    logger.info("Loaded data with %d rows for timeframe: %s", len(data), timeframe)

    # Ensure target column exists
    target = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
    horizon = config.get('model', {}).get('prediction_horizon', 12)
    target_col = f"target_{horizon}"
    if target_col not in data.columns:
        logger.info("Target column '%s' not found. Creating it now...", target_col)
        data = processor.create_target_variable(data, target, horizon)

    # Split data for testing
    split_idx = int(len(data) * 0.8)
    test_data = data.iloc[split_idx:]
    X_test, y_test = processor.prepare_ml_features(test_data, horizon)
    logger.info("Prepared features with shape: %s", X_test.shape)

    # Instead of trying to load metrics file, extract features directly from the model
    # or use all features from the training
    logger.info("Using all available test features, matching those used in training")

    # Create train/test split for our backtest
    split = int(len(X_test) * 0.8)
    X_train_part, y_train_part = X_test.iloc[:split], y_test.iloc[:split]
    X_test_part, y_test_part = X_test.iloc[split:], y_test.iloc[split:]
    test_segment = test_data.iloc[split:]
    logger.debug("Training split: %d train / %d test", len(X_train_part), len(X_test_part))

    # Train a compatible model
    compatible_model = get_model_by_type(model_type, config)
    if hasattr(model, 'get_hyperparameters'):
        compatible_model.hyperparams = model.get_hyperparameters()
        logger.debug("Transferred hyperparameters from original model")

    compatible_model.fit(X_train_part, y_train_part)
    logger.info("Trained compatible model")

    # Generate signals and evaluate
    evaluator = ModelEvaluator(config)
    signals = evaluator.generate_signals(compatible_model, X_test_part, test_segment)
    logger.debug("Generated %d signals", len(signals))

    metrics, results_df, trades = evaluator.calculate_trading_metrics(
        signals, initial_balance, position_size, include_spread, include_slippage
    )

    model_metrics = evaluator.evaluate(compatible_model, X_test_part, y_test_part)
    metrics['model_metrics'] = model_metrics

    # Add metadata to metrics
    test_start = test_segment.index[0]
    test_end = test_segment.index[-1]
    metrics.update({
        'test_start_date': test_start,
        'test_end_date': test_end,
        'test_duration_days': (test_end - test_start).days,
        'config': {
            'target_type': target,
            'prediction_horizon': horizon,
            'timeframe': timeframe,
            'position_size': position_size,
            'include_spread': include_spread,
            'include_slippage': include_slippage
        }
    })

    # Fix: Create an absolute path for results directory and correct filename creation.
    import os
    import pickle
    from datetime import datetime

    # Use an absolute path based on this file's directory as the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Remove any existing extension from the model filename
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{model_name}_{timeframe}_{timestamp}.pkl"
    filepath = os.path.join(results_dir, filename)

    # Save the results
    with open(filepath, 'wb') as f:
        pickle.dump({'metrics': metrics, 'results': results_df.to_dict(), 'trades': trades}, f)

    logger.info("Backtest results saved to: %s", filepath)

    # Log summary
    logger.info("Backtest Summary: %s → %s | Return: %.2f%% | Trades: %d | Sharpe: %.2f",
                test_start, test_end,
                metrics['return_pct'], metrics['n_trades'],
                metrics['sharpe_ratio'])

    if model_metrics:
        if 'accuracy' in model_metrics:
            logger.info("Accuracy: %.4f | F1: %.4f",
                        model_metrics['accuracy'], model_metrics['f1_score'])
        elif 'rmse' in model_metrics:
            logger.info("RMSE: %.4f", model_metrics['rmse'])

    return metrics, results_df, trades
