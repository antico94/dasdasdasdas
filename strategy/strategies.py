import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import logging


class SignalType(Enum):
    """Enum for different signal types."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class ExitReason(Enum):
    """Enum for position exit reasons."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    OPPOSITE_SIGNAL = "opposite_signal"
    SESSION_CLOSE = "session_close"
    MANUAL = "manual"
    TARGET_REACHED = "target_reached"
    VOLATILITY_EXIT = "volatility_exit"


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("Strategy")
        self.name = self.__class__.__name__
        self.logger.info(f"Initializing strategy: {self.name}")

        # Default strategy parameters
        self.take_profit_pct = config.get('strategy', {}).get('take_profit_pct', 1.0)
        self.stop_loss_pct = config.get('strategy', {}).get('stop_loss_pct', 0.5)
        self.trailing_stop_pct = config.get('strategy', {}).get('trailing_stop_pct', 0.3)
        self.max_hold_hours = config.get('strategy', {}).get('max_hold_hours', 48)

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, model_predictions: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from model predictions and market data."""
        pass

    @abstractmethod
    def calculate_position_size(self, account_balance: float, current_price: float,
                                risk_per_trade: float, stop_loss_price: Optional[float] = None) -> float:
        """Calculate position size based on account balance and risk parameters."""
        pass

    @abstractmethod
    def should_exit_position(self, position_type: int, entry_price: float, current_price: float,
                             entry_time: pd.Timestamp, current_time: pd.Timestamp,
                             position_data: Dict) -> Tuple[bool, Optional[ExitReason]]:
        """Determine if current position should be exited."""
        pass

    def update_stops(self, position_type: int, entry_price: float, current_price: float,
                     highest_price: float, lowest_price: float, stop_loss: float,
                     take_profit: float) -> Tuple[float, float]:
        """Update stop loss and take profit levels based on price movement."""
        # Default implementation with no trailing - subclasses should override if needed
        return stop_loss, take_profit

    def get_stats(self) -> Dict:
        """Return strategy statistics and parameters."""
        return {
            'name': self.name,
            'parameters': {
                'take_profit_pct': self.take_profit_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'trailing_stop_pct': self.trailing_stop_pct,
                'max_hold_hours': self.max_hold_hours,
            },
            'tracking': {}
        }


class GoldTrendStrategy(Strategy):
    """Trading strategy focused on gold trend following with sophisticated exits."""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Entry parameters
        self.entry_confidence_threshold = config.get('strategy', {}).get('min_confidence', 0.65)
        self.use_contrarian_entries = config.get('strategy', {}).get('contrarian_entries', False)

        # Exit parameters - from parent class
        # self.take_profit_pct already set in parent
        # self.stop_loss_pct already set in parent
        # self.trailing_stop_pct already set in parent
        # self.max_hold_hours already set in parent

        self.use_trailing_stop = config.get('strategy', {}).get('use_trailing_stop', True)

        # Partial profit taking
        self.partial_tp_enabled = config.get('strategy', {}).get('partial_tp_enabled', True)
        self.partial_tp_pct = config.get('strategy', {}).get('partial_tp_pct', 0.5)
        self.partial_tp_size = config.get('strategy', {}).get('partial_tp_size', 0.5)

        # Risk parameters
        self.risk_per_trade = config.get('risk', {}).get('risk_per_trade', 0.02)
        self.max_risk_per_trade = config.get('risk', {}).get('max_risk_per_trade', 0.05)

        # Gold-specific parameters
        self.gold_price_levels = config.get('gold', {}).get('psychological_levels',
                                                            [1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200])

        # Additional tracking for diagnostics
        self.tracking = {
            'total_signals': 0,
            'entry_signals_ignored': 0,
            'exit_reasons': {reason.value: 0 for reason in ExitReason},
            'max_consecutive_losses': 0
        }

        self.logger.info(f"GoldTrendStrategy initialized with TP: {self.take_profit_pct}%, "
                         f"SL: {self.stop_loss_pct}%, max hold: {self.max_hold_hours}h")

    def generate_signals(self, data: pd.DataFrame, model_predictions: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on model predictions and gold-specific logic."""
        results = data.copy()

        # Merge model predictions if needed
        if model_predictions is not None and not model_predictions.empty:
            # Get the common columns for logging purposes
            pred_columns = model_predictions.columns.tolist()
            self.logger.debug(f"Model prediction columns: {pred_columns}")

            # Ensure model_predictions has the same index as data
            common_indices = model_predictions.index.intersection(results.index)
            if len(common_indices) < len(results):
                self.logger.warning(
                    f"Only {len(common_indices)}/{len(results)} indices match between data and predictions")

            # Merge predictions with results
            for col in pred_columns:
                if col in model_predictions.columns:
                    results.loc[common_indices, col] = model_predictions.loc[common_indices, col]

        # Determine if model is biased (all predictions favor one class)
        is_biased = False
        bias_direction = None
        if 'pred_probability_up' in results.columns:
            up_probs = results['pred_probability_up'].dropna()
            if len(up_probs) > 0:
                avg_up_prob = up_probs.mean()
                self.logger.info(f"Average UP probability: {avg_up_prob:.4f}")

                if avg_up_prob > 0.8 or avg_up_prob < 0.2:
                    is_biased = True
                    bias_direction = "UP" if avg_up_prob > 0.8 else "DOWN"
                    self.logger.warning(
                        f"Model appears strongly biased towards {bias_direction}: avg prob={avg_up_prob:.4f}")

        # Initialize signal column
        results['signal'] = SignalType.HOLD.value

        # Modified signal generation logic to handle potential model bias
        if 'pred_probability_up' in results.columns and 'pred_probability_down' in results.columns:
            if is_biased and self.use_contrarian_entries:
                self.logger.info("Using contrarian entry logic due to model bias")
                # Higher UP probability = SELL signal (contrarian)
                sell_mask = results['pred_probability_up'] >= 0.7
                buy_mask = results['pred_probability_up'] <= 0.3
            else:
                # Use a more balanced threshold for entries
                # For uptrend bias, increase UP threshold and lower DOWN threshold
                up_threshold = 0.7  # Higher threshold for UP signals
                down_threshold = 0.6  # Lower threshold for DOWN signals

                buy_mask = results['pred_probability_up'] >= up_threshold
                sell_mask = results['pred_probability_down'] >= down_threshold

            results.loc[buy_mask, 'signal'] = SignalType.BUY.value
            results.loc[sell_mask, 'signal'] = SignalType.SELL.value

            self.logger.info(f"Generated {buy_mask.sum()} BUY and {sell_mask.sum()} SELL signals")

        # Additional Gold-specific filters
        # 1. Avoid trading during major economic releases (if data available)
        if 'high_impact_news' in results.columns:
            news_mask = results['high_impact_news'] == True
            results.loc[news_mask, 'signal'] = SignalType.HOLD.value
            self.logger.info(f"Filtered out {news_mask.sum()} signals due to high impact news")

        # 2. Add caution near psychological price levels
        if 'close' in results.columns:
            for level in self.gold_price_levels:
                # Within 0.5% of psychological level
                near_level_mask = (abs(results['close'] - level) / level < 0.005)
                # Don't completely filter out, but could modify strategy behavior
                self.logger.info(f"Found {near_level_mask.sum()} periods near psychological level {level}")
                # Optional: reduce position size near psychological levels

        # Track statistics
        self.tracking['total_signals'] = (results['signal'] != SignalType.HOLD.value).sum()

        return results

    def calculate_position_size(self, account_balance: float, current_price: float,
                                risk_per_trade: float = None, stop_loss_price: Optional[float] = None) -> float:
        """Calculate position size based on risk parameters."""
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade

        # Cap at maximum risk percentage
        risk_per_trade = min(risk_per_trade, self.max_risk_per_trade)

        # Calculate risk amount in dollars
        risk_amount = account_balance * risk_per_trade

        # If we have a stop loss price, use it to calculate position size
        if stop_loss_price is not None and stop_loss_price > 0:
            # Calculate risk per unit
            price_risk = abs(current_price - stop_loss_price)
            if price_risk > 0:
                # Position size = risk amount / risk per unit
                position_size = risk_amount / price_risk
                return position_size

        # Fallback calculation based on fixed percentage
        # For gold, typically use smaller position sizes due to volatility
        # Using percentage of price as approximate risk
        risk_price_percent = self.stop_loss_pct / 100
        position_size = risk_amount / (current_price * risk_price_percent)

        # Limit position size to 50% of account at maximum
        max_position = account_balance * 0.5 / current_price
        position_size = min(position_size, max_position)

        return position_size

    def should_exit_position(self, position_type: int, entry_price: float, current_price: float,
                             entry_time: pd.Timestamp, current_time: pd.Timestamp,
                             position_data: Dict) -> Tuple[bool, Optional[ExitReason]]:

        # 1. Time-based exit
        time_diff = (current_time - entry_time).total_seconds() / 3600  # in hours
        if time_diff >= self.max_hold_hours:
            self.tracking['exit_reasons'][ExitReason.TIME_EXIT.value] += 1
            return True, ExitReason.TIME_EXIT

        # 2. Take profit and stop loss
        if position_type == 1:  # Long position
            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price * 100

            # Take profit check
            if profit_pct >= self.take_profit_pct:
                self.tracking['exit_reasons'][ExitReason.TAKE_PROFIT.value] += 1
                return True, ExitReason.TAKE_PROFIT

            # Stop loss check
            if profit_pct <= -self.stop_loss_pct:
                self.tracking['exit_reasons'][ExitReason.STOP_LOSS.value] += 1
                return True, ExitReason.STOP_LOSS

            # Trailing stop if enabled and in profit
            if self.use_trailing_stop and 'highest_price' in position_data:
                highest_price = position_data['highest_price']
                # If we've moved significantly into profit and then retraced
                if highest_price > entry_price:
                    # Calculate how much we've retraced from the highest point
                    retrace_pct = (highest_price - current_price) / highest_price * 100
                    if retrace_pct >= self.trailing_stop_pct:
                        self.tracking['exit_reasons'][ExitReason.TRAILING_STOP.value] += 1
                        return True, ExitReason.TRAILING_STOP

        elif position_type == -1:  # Short position
            # Calculate profit percentage (inverted for shorts)
            profit_pct = (entry_price - current_price) / entry_price * 100

            # Take profit check
            if profit_pct >= self.take_profit_pct:
                self.tracking['exit_reasons'][ExitReason.TAKE_PROFIT.value] += 1
                return True, ExitReason.TAKE_PROFIT

            # Stop loss check
            if profit_pct <= -self.stop_loss_pct:
                self.tracking['exit_reasons'][ExitReason.STOP_LOSS.value] += 1
                return True, ExitReason.STOP_LOSS

            # Trailing stop if enabled and in profit
            if self.use_trailing_stop and 'lowest_price' in position_data:
                lowest_price = position_data['lowest_price']
                # If we've moved significantly into profit and then retraced
                if lowest_price < entry_price:
                    # Calculate how much we've retraced from the lowest point
                    retrace_pct = (current_price - lowest_price) / lowest_price * 100
                    if retrace_pct >= self.trailing_stop_pct:
                        self.tracking['exit_reasons'][ExitReason.TRAILING_STOP.value] += 1
                        return True, ExitReason.TRAILING_STOP

        # 3. Check for opposing signal if provided
        if 'current_signal' in position_data:
            if position_type == 1 and position_data['current_signal'] == SignalType.SELL.value:
                self.tracking['exit_reasons'][ExitReason.OPPOSITE_SIGNAL.value] += 1
                return True, ExitReason.OPPOSITE_SIGNAL
            elif position_type == -1 and position_data['current_signal'] == SignalType.BUY.value:
                self.tracking['exit_reasons'][ExitReason.OPPOSITE_SIGNAL.value] += 1
                return True, ExitReason.OPPOSITE_SIGNAL

        # 4. Check for exit at psychological levels
        if hasattr(self, 'gold_price_levels') and self.gold_price_levels:
            # Find nearest psychological level
            closest_level = min(self.gold_price_levels, key=lambda x: abs(x - current_price))
            # If we're very close to a psychological level and in profit, consider exiting
            if abs(current_price - closest_level) / closest_level < 0.002:  # Within 0.2%
                if (position_type == 1 and current_price > entry_price) or \
                        (position_type == -1 and current_price < entry_price):
                    self.tracking['exit_reasons'][ExitReason.TARGET_REACHED.value] += 1
                    return True, ExitReason.TARGET_REACHED

        # No exit signal
        return False, None

    def update_stops(self, position_type: int, entry_price: float, current_price: float,
                     highest_price: float, lowest_price: float, stop_loss: float,
                     take_profit: float) -> Tuple[float, float]:
        """Update stop loss and take profit levels based on price movement."""
        new_stop_loss = stop_loss
        new_take_profit = take_profit

        # Implement trailing stop logic
        if not self.use_trailing_stop:
            return new_stop_loss, new_take_profit

        if position_type == 1:  # Long position
            # Only trail stops if we're in profit
            if highest_price > entry_price:
                # Calculate trailing stop level based on highest price
                trailing_stop_level = highest_price * (1 - self.trailing_stop_pct / 100)
                # Only raise stop loss, never lower it
                if trailing_stop_level > new_stop_loss:
                    new_stop_loss = trailing_stop_level
                    self.logger.debug(f"Updated long trailing stop to {new_stop_loss:.2f} "
                                      f"(highest: {highest_price:.2f}, trail: {self.trailing_stop_pct}%)")

        elif position_type == -1:  # Short position
            # Only trail stops if we're in profit
            if lowest_price < entry_price:
                # Calculate trailing stop level based on lowest price
                trailing_stop_level = lowest_price * (1 + self.trailing_stop_pct / 100)
                # Only lower stop loss, never raise it
                if trailing_stop_level < new_stop_loss or new_stop_loss == 0:
                    new_stop_loss = trailing_stop_level
                    self.logger.debug(f"Updated short trailing stop to {new_stop_loss:.2f} "
                                      f"(lowest: {lowest_price:.2f}, trail: {self.trailing_stop_pct}%)")

        return new_stop_loss, new_take_profit

    def get_stats(self) -> Dict:
        """Return strategy statistics."""
        return {
            'name': self.name,
            'parameters': {
                'take_profit_pct': self.take_profit_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'trailing_stop_pct': self.trailing_stop_pct,
                'max_hold_hours': self.max_hold_hours,
                'entry_confidence': self.entry_confidence_threshold,
                'use_trailing_stop': self.use_trailing_stop,
                'contrarian_entries': self.use_contrarian_entries
            },
            'tracking': self.tracking
        }


class StrategyFactory:
    """Factory class to create strategy instances."""

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict) -> Strategy:
        """Create and return a strategy instance based on the name."""
        if strategy_name.lower() == 'goldtrend':
            return GoldTrendStrategy(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
