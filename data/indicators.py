from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Class for calculating optimized technical indicators for XAUUSD H1 trading.
    """

    def __init__(self, config_path: Optional[str] = None):
        # If no config_path is provided, compute it relative to this file's location.
        if config_path is None:
            # __file__ is something like:
            # C:\Users\ameiu\PycharmProjects\GoldML\data\indicators.py
            current_file = Path(__file__).resolve()
            # The project root should be one level up from the 'data' folder
            # current_file.parents[0] is the 'data' folder, and current_file.parents[1] is GoldML.
            project_root = current_file.parents[1]
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators (EMAs and MACD) to the dataframe.
        Uses parameters from config if available.
        """
        # Get trend indicators configuration from config
        indicators_config = self.config.get("features", {}).get("technical_indicators", [])
        trend_config = next((item["indicators"] for item in indicators_config if item.get("type") == "trend"), [])

        for indicator in trend_config:
            if indicator.get("name") == "ema":
                # Default relevant periods for gold trading
                default_periods = [9, 21, 55, 200]
                params = indicator.get("params", [])
                for period in params:
                    if period in default_periods:
                        try:
                            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
                            # Verify EMA has multiple unique values
                            unique_vals = df[f"ema_{period}"].nunique()
                            if unique_vals < 5:
                                logger.warning(f"EMA_{period} has only {unique_vals} unique values")
                        except Exception as e:
                            logger.warning(f"Failed to compute EMA for period {period}: {e}")

            elif indicator.get("name") == "macd":
                params = indicator.get("params", {})
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                try:
                    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
                    # Get standard column names from pandas_ta
                    macd_key = f"MACD_{fast}_{slow}_{signal}"
                    macds_key = f"MACDs_{fast}_{slow}_{signal}"

                    # Directly use the standardized names
                    df["MACD_12_26_9"] = macd.get(macd_key, np.nan)
                    df["MACDs_12_26_9"] = macd.get(macds_key, np.nan)

                    # Verify MACD has multiple unique values
                    for col in ["MACD_12_26_9", "MACDs_12_26_9"]:
                        unique_vals = df[col].nunique()
                        if unique_vals < 5:
                            logger.warning(f"{col} has only {unique_vals} unique values")
                except Exception as e:
                    logger.warning(f"Failed to compute MACD: {e}")

        # Add EMA crossover signal between ema_9 and ema_21 if available
        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["ema_9_21_cross"] = np.where(df["ema_9"] > df["ema_21"], 1, -1)

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators, such as RSI, to the dataframe.
        """
        indicators_config = self.config.get("features", {}).get("technical_indicators", [])
        momentum_config = next((item["indicators"] for item in indicators_config if item.get("type") == "momentum"), [])

        for indicator in momentum_config:
            if indicator.get("name") == "rsi":
                # Use provided period or default to 14
                period = indicator.get("params", [14])[0]
                try:
                    df["rsi_14"] = ta.rsi(df["close"], length=period)
                    # Verify RSI has multiple unique values
                    unique_vals = df["rsi_14"].nunique()
                    if unique_vals < 5:
                        logger.warning(f"RSI_14 has only {unique_vals} unique values")
                except Exception as e:
                    logger.warning(f"Failed to compute RSI: {e}")

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators such as Bollinger Bands and ATR.
        Parameters are taken from config where available.
        """
        indicators_config = self.config.get("features", {}).get("technical_indicators", [])
        volatility_config = next(
            (item["indicators"] for item in indicators_config if item.get("type") == "volatility"), []
        )

        for indicator in volatility_config:
            if indicator.get("name") == "bbands":
                params = indicator.get("params", {})
                length = params.get("length", 20)
                std_val = params.get("std", 2)
                try:
                    bbands = ta.bbands(df["close"], length=length, std=std_val)

                    # Directly use standardized naming convention
                    df["BBL_20_2.0"] = bbands.get(f"BBL_{length}_{std_val}", np.nan)
                    df["BBM_20_2.0"] = bbands.get(f"BBM_{length}_{std_val}", np.nan)
                    df["BBU_20_2.0"] = bbands.get(f"BBU_{length}_{std_val}", np.nan)
                    df["BBP_20_2.0"] = bbands.get(f"BBP_{length}_{std_val}", np.nan)

                    # Instead of filling with constants, use forward/backward fill
                    for col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBP_20_2.0"]:
                        if col in df.columns and df[col].isna().any():
                            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                            # If still NaN (at the edges), use median
                            if df[col].isna().any():
                                df[col] = df[col].fillna(df[col].median())

                            # Verify Bollinger Bands have multiple unique values
                            unique_vals = df[col].nunique()
                            if unique_vals < 5:
                                logger.warning(f"{col} has only {unique_vals} unique values")
                except Exception as e:
                    logger.warning(f"Failed to compute Bollinger Bands: {e}")

            elif indicator.get("name") == "atr":
                period = indicator.get("params", [14])[0] if isinstance(indicator.get("params"), list) else 14
                try:
                    # Directly calculate with standardized names
                    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=period)
                    df["atr_pct_14"] = df["atr_14"] / df["close"] * 100

                    # Verify ATR has multiple unique values
                    for col in ["atr_14", "atr_pct_14"]:
                        unique_vals = df[col].nunique()
                        if unique_vals < 5:
                            logger.warning(f"{col} has only {unique_vals} unique values")
                except Exception as e:
                    logger.warning(f"Failed to compute ATR: {e}")

        return df

    def add_custom_gold_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom indicators specifically useful for gold trading on H1.
        For example: daily price change and high-low range.
        """
        try:
            # Calculate daily change (assuming 24 H1 bars per day)
            df["daily_change"] = df["close"].pct_change(24) * 100
            # Verify daily_change has multiple unique values
            unique_vals = df["daily_change"].nunique()
            if unique_vals < 5:
                logger.warning(f"daily_change has only {unique_vals} unique values")
        except Exception as e:
            logger.warning(f"Failed to compute daily change: {e}")

        try:
            # High-low range as a percentage of close
            df["high_low_range"] = (df["high"] - df["low"]) / df["close"] * 100
            # Verify high_low_range has multiple unique values
            unique_vals = df["high_low_range"].nunique()
            if unique_vals < 5:
                logger.warning(f"high_low_range has only {unique_vals} unique values")
        except Exception as e:
            logger.warning(f"Failed to compute high_low_range: {e}")

        return df

    def add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate local highs/lows, distance to support/resistance,
        and generate breakout signals.
        """
        try:
            df["local_high"] = df["high"].rolling(window).max()
            df["local_low"] = df["low"].rolling(window).min()
            df["dist_to_resistance"] = (df["local_high"] - df["close"]) / df["close"] * 100
            df["dist_to_support"] = (df["close"] - df["local_low"]) / df["close"] * 100

            # Generate breakout signals (shift by 1 for previous period check)
            df["break_resistance"] = np.where(df["close"] > df["local_high"].shift(1), 1, 0)
            df["break_support"] = np.where(df["close"] < df["local_low"].shift(1), 1, 0)

            # Verify support/resistance indicators have multiple unique values
            for col in ["local_high", "local_low", "dist_to_resistance", "dist_to_support"]:
                unique_vals = df[col].nunique()
                if unique_vals < 5:
                    logger.warning(f"{col} has only {unique_vals} unique values")
        except Exception as e:
            logger.warning(f"Failed to compute support/resistance: {e}")

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicator conditions.
        - Create MACD crossover signals.
        - Generate buy signals if price breaks resistance with a positive MACD.
        - Generate sell signals if price breaks support with a negative MACD.
        """
        try:
            if all(col in df.columns for col in ["MACD_12_26_9", "MACDs_12_26_9"]):
                df["macd_cross_up"] = np.where(
                    (df["MACD_12_26_9"] > df["MACDs_12_26_9"]) &
                    (df["MACD_12_26_9"].shift(1) <= df["MACDs_12_26_9"].shift(1)),
                    1, 0
                )
                df["macd_cross_down"] = np.where(
                    (df["MACD_12_26_9"] < df["MACDs_12_26_9"]) &
                    (df["MACD_12_26_9"].shift(1) >= df["MACDs_12_26_9"].shift(1)),
                    1, 0
                )
        except Exception as e:
            logger.warning(f"Failed to generate MACD crossover signals: {e}")

        # Initialize signal columns
        df["buy_signal"] = 0
        df["sell_signal"] = 0

        try:
            # Buy signal: when price breaks resistance and MACD is positive
            if all(col in df.columns for col in ["break_resistance", "MACD_12_26_9"]):
                df.loc[(df["break_resistance"] == 1) & (df["MACD_12_26_9"] > 0), "buy_signal"] = 1
        except Exception as e:
            logger.warning(f"Failed to generate buy signals: {e}")

        try:
            # Sell signal: when price breaks support and MACD is negative
            if all(col in df.columns for col in ["break_support", "MACD_12_26_9"]):
                df.loc[(df["break_support"] == 1) & (df["MACD_12_26_9"] < 0), "sell_signal"] = 1
        except Exception as e:
            logger.warning(f"Failed to generate sell signals: {e}")

        return df

    def add_all_indicators(
            self, df: pd.DataFrame, external_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe for XAUUSD H1 trading.
        This includes trend, momentum, volatility, custom gold, support/resistance, and signal generation.
        """
        original_shape = df.shape
        logger.info(f"Adding technical indicators to dataframe with shape {original_shape}")

        # Before processing, check for sufficient data
        if len(df) < 200:  # Enough data for most indicators including EMA-200
            logger.warning(f"Dataset may be too small ({len(df)} rows) for reliable indicator calculation")

        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_custom_gold_indicators(df)
        df = self.add_support_resistance(df)
        df = self.generate_signals(df)

        # Final validation - check if any indicator columns are constant
        for col in df.columns:
            if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                unique_vals = df[col].nunique()
                if unique_vals == 1:
                    logger.warning(f"Column {col} has only 1 unique value: {df[col].iloc[0]}")

        new_shape = df.shape
        logger.info(f"Added {new_shape[1] - original_shape[1]} new indicator columns")

        return df