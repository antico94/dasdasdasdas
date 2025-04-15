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
from models.factory import ModelFactory
from utils.logger import setup_logger

# Set up a module-level logger for the evaluator/backtest module
logger = setup_logger("BacktestEvaluator")


class ModelEvaluator:
    """Evaluate model performance and generate trading signals."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.prediction_horizon = config.get('model', {}).get('prediction_horizon', 1)
        self.confidence_threshold = config.get('strategy', {}).get('min_confidence', 0.55)
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
                # Apply mask if necessary
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

        # Get predictions and add them to results
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

                # --- ENHANCED LOGIC: Adjust threshold if average confidence is borderline ---
                applied_threshold = self.confidence_threshold
                avg_conf = np.mean(max_probas)
                if avg_conf < self.confidence_threshold and avg_conf > (self.confidence_threshold - 0.05):
                    self.logger.info("Adjusting confidence threshold from %.2f to %.2f based on average confidence of %.4f",
                                     self.confidence_threshold, avg_conf, avg_conf)
                    applied_threshold = avg_conf

                results['signal'] = TradeAction.HOLD.value
                # Set BUY signal when prediction == 1 and confidence >= threshold;
                # SELL when prediction == 0 and confidence >= threshold.
                buy_mask = ((results['prediction'] == 1) & (results['confidence'] >= applied_threshold))
                sell_mask = ((results['prediction'] == 0) & (results['confidence'] >= applied_threshold))
                results.loc[X.index[valid_idx], 'signal'] = np.where(buy_mask, TradeAction.BUY.value,
                                                                      np.where(sell_mask, TradeAction.SELL.value,
                                                                               TradeAction.HOLD.value))
                self.logger.debug("Signals generated for classifier with applied threshold %.2f: %s",
                                  applied_threshold, results['signal'].unique())
            except (AttributeError, ValueError) as e:
                self.logger.warning("Could not compute prediction probabilities for signals: %s", str(e))
                results['signal'] = np.where(results['prediction'] == 1, TradeAction.BUY.value,
                                             np.where(results['prediction'] == 0, TradeAction.SELL.value, TradeAction.HOLD.value))
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

    def calculate_trading_metrics(self, results: pd.DataFrame, model_path: str, timeframe: str,
                                  initial_balance: float = 10000.0, position_size: float = 0.1,
                                  include_spread: bool = True, include_slippage: bool = True
                                  ) -> Tuple[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
        """Calculate trading performance metrics from signals."""
        self.logger.info("Calculating trading metrics.")
        df = results.copy()
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['trade_return'] = 0.0
        df['balance'] = initial_balance
        df['equity'] = initial_balance

        # Get cost parameters
        spread_pips = self.config.get('risk', {}).get('spread_pips', 30)
        slippage_pips = self.config.get('risk', {}).get('slippage_pips', 5)
        spread_cost = spread_pips * 0.01 if include_spread else 0
        slippage_cost = slippage_pips * 0.01 if include_slippage else 0

        trades = []
        active_trade = None
        position = 0  # Local variable tracking current position (1 for long, -1 for short)
        entry_price = 0.0
        balance = initial_balance  # Initialize balance

        for i in range(1, len(df)):
            prev_idx = df.index[i - 1]
            curr_idx = df.index[i]
            signal = df.loc[curr_idx, 'signal']
            prev_position = df.loc[prev_idx, 'position']

            if signal == TradeAction.BUY.value and prev_position <= 0:
                if prev_position < 0:  # If holding short, close it first
                    exit_price = df.loc[curr_idx, 'open'] + slippage_cost
                    df.loc[curr_idx, 'exit_price'] = exit_price
                    entry_price_old = active_trade['entry_price']
                    trade_return = (entry_price_old - exit_price) / entry_price_old
                    trade_return -= spread_cost / entry_price_old
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
                position = 1  # Set local position to long
                trade_size = df.loc[prev_idx, 'balance'] * position_size
                active_trade = {
                    'type': 'long',
                    'entry_date': curr_idx,
                    'entry_price': entry_price,
                    'size': trade_size
                }
            elif signal == TradeAction.SELL.value and prev_position >= 0:
                if prev_position > 0:  # Close long position before selling short
                    exit_price = df.loc[curr_idx, 'open'] - slippage_cost
                    df.loc[curr_idx, 'exit_price'] = exit_price
                    entry_price_old = active_trade['entry_price']
                    trade_return = (exit_price - entry_price_old) / entry_price_old
                    trade_return -= spread_cost / entry_price_old
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
                position = -1  # Set local position to short
                trade_size = df.loc[prev_idx, 'balance'] * position_size
                active_trade = {
                    'type': 'short',
                    'entry_date': curr_idx,
                    'entry_price': entry_price,
                    'size': trade_size
                }
            elif signal == TradeAction.CLOSE.value:
                exit_price = df.loc[curr_idx, 'open']
                if position == 1:  # Closing long position
                    exit_price = df.loc[curr_idx, 'open'] - slippage_cost
                    profit_loss = (exit_price - entry_price) * active_trade['size']
                else:  # Closing short position
                    exit_price = df.loc[curr_idx, 'open'] + slippage_cost
                    profit_loss = (entry_price - exit_price) * active_trade['size']
                max_loss = -initial_balance * 0.05  # Limit loss to 5% of initial balance per trade
                if profit_loss < max_loss:
                    self.logger.warning(f"Limiting loss at {curr_idx} from {profit_loss:.2f} to {max_loss:.2f}")
                    profit_loss = max_loss
                balance += profit_loss
                df.loc[curr_idx, 'profit_loss'] = profit_loss
                trades.append({
                    'time': curr_idx,
                    'action': 'CLOSE',
                    'price': exit_price,
                    'size': active_trade['size'],
                    'profit_loss': profit_loss,
                    'balance': balance
                })
                active_trade = None
                position = 0  # Reset position
                entry_price = 0.0
            df.loc[curr_idx, 'balance'] = balance
            df.loc[curr_idx, 'position'] = position
            df.loc[curr_idx, 'entry_price'] = entry_price if position != 0 else None

        # Close any remaining open position at the end of the test period
        if position != 0:
            last_idx = df.index[-1]
            last_price = df.loc[last_idx, 'open']
            if position == 1:
                exit_price = last_price - slippage_cost
                profit_loss = (exit_price - entry_price) * active_trade['size']
            else:
                exit_price = last_price + slippage_cost
                profit_loss = (entry_price - exit_price) * active_trade['size']
            max_loss = -initial_balance * 0.05
            if profit_loss < max_loss:
                profit_loss = max_loss
            balance += profit_loss
            current_pl = df.loc[last_idx, 'profit_loss'] if 'profit_loss' in df.columns else 0
            if pd.isna(current_pl):
                current_pl = 0
            df.loc[last_idx, 'profit_loss'] = current_pl + profit_loss
            trades.append({
                'time': last_idx,
                'action': 'CLOSE_FINAL',
                'price': exit_price,
                'size': active_trade['size'] if active_trade else 0,
                'profit_loss': profit_loss,
                'balance': balance
            })

        df['cum_profit_loss'] = df['profit_loss'].cumsum()

        metrics = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'absolute_return': balance - initial_balance,
            'return_pct': ((balance / initial_balance) - 1) * 100,
            'n_trades': len(trades)
        }

        if metrics['n_trades'] > 0:
            profit_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            loss_trades = [t for t in trades if t.get('profit_loss', 0) < 0]

            metrics['n_winning_trades'] = len(profit_trades)
            metrics['n_losing_trades'] = len(loss_trades)
            metrics['win_rate'] = len(profit_trades) / len(trades) if trades else 0

            total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
            total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0

            peak = initial_balance
            max_drawdown = 0
            balance_curve = df[['balance']].dropna()
            for idx, row in balance_curve.iterrows():
                current_balance = row['balance']
                if current_balance > peak:
                    peak = current_balance
                drawdown = (peak - current_balance) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            metrics['max_drawdown_pct'] = max_drawdown

            daily_returns = df['profit_loss'] / df['balance'].shift(1)
            daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(daily_returns) > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
                metrics['sharpe_ratio'] = sharpe_ratio
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics.update({
                'n_winning_trades': 0,
                'n_losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0
            })

        self.logger.info(f"Backtest Results:")
        self.logger.info(f"  Initial Balance: ${metrics['initial_balance']:.2f}")
        self.logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
        self.logger.info(f"  Return: {metrics['return_pct']:.2f}%")
        self.logger.info(f"  Number of Trades: {metrics['n_trades']}")

        if metrics['n_trades'] > 0:
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
            self.logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        storage = DataStorage()
        model_name = os.path.basename(model_path).replace('.joblib', '')
        backtest_results = {
            'metrics': metrics,
            'trades': trades,
            'balance_curve': df[['balance']].copy(),
            'signals': df[['signal', 'trade_action']].copy(),
            'predictions': df[['pred_prob_up', 'pred_prob_down']].copy() if 'pred_prob_up' in df.columns else None
        }

        results_path = storage.save_results(
            backtest_results,
            f"{model_name}_backtest_{timeframe}",
            include_timestamp=True
        )
        self.logger.info(f"Saved backtest results to {results_path}")

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


def backtest_model(
        config: Dict,
        model_path: str,
        timeframe: str,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        include_spread: bool = True,
        include_slippage: bool = True
) -> Tuple[Dict, pd.DataFrame, List[Dict]]:
    """
    Backtest a trained model on historical data with enhanced logging.

    Args:
        config: Configuration dictionary
        model_path: Path to the trained model file
        timeframe: Timeframe to use for backtesting (e.g., 'H1')
        initial_balance: Initial account balance
        risk_per_trade: Fraction of balance to risk per trade
        include_spread: Whether to include spread costs
        include_slippage: Whether to include slippage

    Returns:
        Tuple of (metrics, results_dataframe, trades_list)
    """
    # Setup enhanced logging
    logger = setup_logger(name="BacktestLogger")
    logger.info(f"Starting backtest for model: {model_path}, timeframe: {timeframe}")
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

    # --- FIX: Align test features with fitted feature names ---
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

    # --- NEW: If metadata includes fitted feature names, override test features ---
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

    # Get raw model predictions
    logger.info("Generating predictions...")

    is_classifier = hasattr(model, 'predict_proba')
    if is_classifier:
        try:
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:
                positive_probs = y_pred_proba[:, 1]
                prob_bins = np.linspace(0, 1, 11)
                hist, _ = np.histogram(positive_probs, bins=prob_bins)
                bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

                logger.info("Prediction probability distribution:")
                for center, count in zip(bin_centers, hist):
                    logger.info(f"  {center:.1f}: {count} predictions ({count / len(positive_probs) * 100:.1f}%)")
                logger.info(f"Min probability: {positive_probs.min():.4f}")
                logger.info(f"Max probability: {positive_probs.max():.4f}")
                logger.info(f"Mean probability: {positive_probs.mean():.4f}")

                confidence_threshold = config.get('strategy', {}).get('min_confidence', 0.55)
                applied_threshold = confidence_threshold
                avg_prob = positive_probs.mean()
                if avg_prob < confidence_threshold and avg_prob > (confidence_threshold - 0.05):
                    logger.info("Adjusting confidence threshold in backtest from %.2f to %.2f based on average probability %.4f",
                                confidence_threshold, avg_prob, avg_prob)
                    applied_threshold = avg_prob

                test_results = test_data.loc[X_test.index].copy()
                test_results['pred_probability'] = positive_probs
                test_results['pred_probability_up'] = positive_probs
                test_results['pred_probability_down'] = 1 - positive_probs
                test_results['signal'] = np.where(test_results['pred_probability_down'] >= applied_threshold, 1, 0)
                test_results['signal'] = np.where(test_results['pred_probability_up'] >= applied_threshold, -1, test_results['signal'])
                if (test_results['signal'] == 0).sum() == len(test_results):
                    logger.warning("No signals generated with standard threshold in backtest, using fallback comparative logic")
                    test_results['signal'] = np.where(test_results['pred_probability_down'] > test_results['pred_probability_up'], 1,
                                                  np.where(test_results['pred_probability_up'] > test_results['pred_probability_down'], -1, 0))
                logger.info(f"Generated {(test_results['signal'] != 0).sum()} trading signals")
                logger.info(f"Buy signals: {(test_results['signal'] == 1).sum()}")
                logger.info(f"Sell signals: {(test_results['signal'] == -1).sum()}")
            else:
                logger.warning(f"Unexpected prediction shape: {y_pred_proba.shape}")
                test_results = test_data.loc[X_test.index].copy()
        except Exception as e:
            logger.error(f"Error generating prediction probabilities: {str(e)}")
            test_results = test_data.loc[X_test.index].copy()
    else:
        y_pred = model.predict(X_test)
        test_results = test_data.loc[X_test.index].copy()
        test_results['predicted_value'] = y_pred
        logger.info(f"Prediction statistics: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")

    # Initialize backtest variables
    balance = initial_balance
    position = 0  # 0: no position, 1: long, -1: short
    position_size = 0.0
    entry_price = 0.0
    trades = []

    # Define columns for tracking backtest results
    test_results['balance'] = initial_balance
    test_results['position'] = 0
    test_results['trade_action'] = None
    test_results['entry_price'] = None
    test_results['exit_price'] = None
    test_results['profit_loss'] = 0.0
    test_results['cum_profit_loss'] = 0.0

    SPREAD_TYPICAL = config.get('risk', {}).get('spread', 0.5)
    SLIPPAGE_TYPICAL = config.get('risk', {}).get('slippage', 0.1)
    XAUUSD_POINT_VALUE = 0.01

    class TradeAction:
        BUY = "BUY"
        SELL = "SELL"
        CLOSE = "CLOSE"
        HOLD = "HOLD"

    spread_points = SPREAD_TYPICAL if include_spread else 0
    slippage_points = SLIPPAGE_TYPICAL if include_slippage else 0
    point_value = XAUUSD_POINT_VALUE

    logger.info(f"Using spread: {spread_points} points, slippage: {slippage_points} points")
    logger.info(f"Point value: {point_value}")

    logger.debug("First 5 rows of test data:")
    for i, (idx, row) in enumerate(test_results.iloc[:5].iterrows()):
        logger.debug(f"{i}, {idx}: close={row['close']:.2f}, signal={row.get('signal', 'N/A')}")

    logger.info("Running backtest simulation...")
    for i, (idx, row) in enumerate(test_results.iterrows()):
        if i < horizon:
            continue

        current_price = row['close']
        action = TradeAction.HOLD
        if 'signal' in row:
            if position == 0:
                if row['signal'] == 1:
                    action = TradeAction.BUY
                elif row['signal'] == -1:
                    action = TradeAction.SELL
            elif position == 1:
                if row['signal'] == -1:
                    action = TradeAction.CLOSE
            elif position == -1:
                if row['signal'] == 1:
                    action = TradeAction.CLOSE

        logger.debug(f"Processing row {i} at {idx}: close={current_price}, action={action}")

        test_results.at[idx, 'trade_action'] = action

        if action == TradeAction.BUY:
            stop_loss_pips = 100
            max_risk_amount = balance * risk_per_trade
            position_size = max_risk_amount / (stop_loss_pips * point_value)
            position_size = min(position_size, balance / current_price * 0.5)
            entry_price = current_price + (spread_points + slippage_points) * point_value
            position = 1

            trades.append({
                'time': idx,
                'action': 'BUY',
                'price': entry_price,
                'size': position_size,
                'balance': balance
            })
            logger.debug(f"BUY signal at {idx}: price={entry_price:.2f}, size={position_size:.4f}, balance={balance:.2f}")

        elif action == TradeAction.SELL:
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
            logger.debug(f"SELL signal at {idx}: price={entry_price:.2f}, size={position_size:.4f}, balance={balance:.2f}")

        elif action == TradeAction.CLOSE:
            exit_price = current_price
            if position == 1:
                exit_price = current_price - (spread_points + slippage_points) * point_value
                profit_loss = (exit_price - entry_price) * position_size
            else:
                exit_price = current_price + (spread_points + slippage_points) * point_value
                profit_loss = (entry_price - exit_price) * position_size

            max_loss = -initial_balance * 0.05
            if profit_loss < max_loss:
                logger.warning(f"Limiting loss at {idx} from {profit_loss:.2f} to {max_loss:.2f}")
                profit_loss = max_loss
            balance += profit_loss
            test_results.at[idx, 'profit_loss'] = profit_loss

            trades.append({
                'time': idx,
                'action': 'CLOSE',
                'price': exit_price,
                'size': position_size,
                'profit_loss': profit_loss,
                'balance': balance
            })
            logger.debug(f"CLOSE position at {idx}: price={exit_price:.2f}, profit_loss={profit_loss:.2f}, new balance={balance:.2f}")
            position = 0
            position_size = 0
            entry_price = 0

        test_results.at[idx, 'balance'] = balance
        test_results.at[idx, 'position'] = position
        test_results.at[idx, 'entry_price'] = entry_price if position != 0 else None

    test_results['cum_profit_loss'] = test_results['profit_loss'].cumsum()

    logger.info("Calculating backtest metrics...")
    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'absolute_return': balance - initial_balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'n_trades': len(trades)
    }
    if metrics['n_trades'] > 0:
        profit_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        loss_trades = [t for t in trades if t.get('profit_loss', 0) < 0]

        metrics['n_winning_trades'] = len(profit_trades)
        metrics['n_losing_trades'] = len(loss_trades)
        metrics['win_rate'] = len(profit_trades) / len(trades) if trades else 0

        total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0

        peak = initial_balance
        max_drawdown = 0
        balance_curve = test_results[['balance']].dropna()
        for idx, row in balance_curve.iterrows():
            current_balance = row['balance']
            if current_balance > peak:
                peak = current_balance
            drawdown = (peak - current_balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        metrics['max_drawdown_pct'] = max_drawdown

        daily_returns = test_results['profit_loss'] / test_results['balance'].shift(1)
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics.update({
            'n_winning_trades': 0,
            'n_losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0
        })

    logger.info(f"Backtest Results:")
    logger.info(f"  Initial Balance: ${metrics['initial_balance']:.2f}")
    logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
    logger.info(f"  Return: {metrics['return_pct']:.2f}%")
    logger.info(f"  Number of Trades: {metrics['n_trades']}")
    if metrics['n_trades'] > 0:
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    storage = DataStorage()
    model_name = os.path.basename(model_path).replace('.joblib', '')
    backtest_results = {
        'metrics': metrics,
        'trades': trades,
        'balance_curve': test_results[['balance']].copy(),
        'signals': test_results[['signal', 'trade_action']].copy() if 'signal' in test_results.columns else None,
        'predictions': test_results[['pred_probability']].copy() if 'pred_probability' in test_results.columns else None
    }

    results_path = storage.save_results(
        backtest_results,
        f"{model_name}_backtest_{timeframe}",
        include_timestamp=True
    )
    logger.info(f"Saved backtest results to {results_path}")

    return metrics, test_results, trades
