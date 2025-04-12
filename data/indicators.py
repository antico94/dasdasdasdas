from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta


class TechnicalIndicators:
    """Class for calculating technical indicators relevant for XAUUSD trading."""

    @staticmethod
    def add_trend_indicators(
            df: pd.DataFrame,
            config: Dict
    ) -> pd.DataFrame:
        """Add trend indicators to dataframe."""
        # Moving Averages
        for indicator in config:
            if indicator["name"] == "sma":
                for period in indicator["params"]:
                    df[f"sma_{period}"] = ta.sma(df["close"], length=period)

            elif indicator["name"] == "ema":
                for period in indicator["params"]:
                    df[f"ema_{period}"] = ta.ema(df["close"], length=period)

            elif indicator["name"] == "macd":
                macd = ta.macd(
                    df["close"],
                    fast=indicator["params"]["fast"],
                    slow=indicator["params"]["slow"],
                    signal=indicator["params"]["signal"]
                )
                df = pd.concat([df, macd], axis=1)

            elif indicator["name"] == "adx":
                adx = ta.adx(
                    df["high"],
                    df["low"],
                    df["close"],
                    length=indicator["params"]
                )
                df = pd.concat([df, adx], axis=1)

        # Add MA crossovers
        if "sma_50" in df.columns and "sma_200" in df.columns:
            df["sma_50_200_cross"] = np.where(
                df["sma_50"] > df["sma_200"], 1, -1
            )

        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["ema_9_21_cross"] = np.where(
                df["ema_9"] > df["ema_21"], 1, -1
            )

        return df

    @staticmethod
    def add_momentum_indicators(
            df: pd.DataFrame,
            config: Dict
    ) -> pd.DataFrame:
        """Add momentum indicators to dataframe."""
        for indicator in config:
            if indicator["name"] == "rsi":
                for period in indicator["params"]:
                    df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

            elif indicator["name"] == "stoch":
                stoch = ta.stoch(
                    df["high"],
                    df["low"],
                    df["close"],
                    k=indicator["params"]["k"],
                    d=indicator["params"]["d"],
                    smooth_k=indicator["params"]["smooth_k"]
                )
                df = pd.concat([df, stoch], axis=1)

            elif indicator["name"] == "cci":
                df[f"cci_{indicator['params']}"] = ta.cci(
                    df["high"],
                    df["low"],
                    df["close"],
                    length=indicator["params"]
                )

        # Add RSI thresholds (important for gold trading)
        if "rsi_14" in df.columns:
            df["rsi_14_ob"] = np.where(df["rsi_14"] > 70, 1, 0)  # Overbought
            df["rsi_14_os"] = np.where(df["rsi_14"] < 30, 1, 0)  # Oversold

        return df

    @staticmethod
    def add_volatility_indicators(
            df: pd.DataFrame,
            config: Dict
    ) -> pd.DataFrame:
        """Add volatility indicators to dataframe."""
        for indicator in config:
            if indicator["name"] == "bbands":
                bbands = ta.bbands(
                    df["close"],
                    length=indicator["params"]["length"],
                    std=indicator["params"]["std"]
                )
                df = pd.concat([df, bbands], axis=1)

                # Add BB position (where price is within bands) - useful for gold volatility
                if "BBL_20_2.0" in df.columns and "BBU_20_2.0" in df.columns:
                    df["bb_pos"] = (df["close"] - df["BBL_20_2.0"]) / (df["BBU_20_2.0"] - df["BBL_20_2.0"])

            elif indicator["name"] == "atr":
                for period in indicator["params"]:
                    df[f"atr_{period}"] = ta.atr(
                        df["high"],
                        df["low"],
                        df["close"],
                        length=period
                    )

                    # Normalize ATR (ATR%)
                    df[f"atr_pct_{period}"] = df[f"atr_{period}"] / df["close"] * 100

        return df

    @staticmethod
    def add_volume_indicators(
            df: pd.DataFrame,
            config: Dict
    ) -> pd.DataFrame:
        """Add volume indicators to dataframe."""
        # Note: MT5 provides tick volume for XAUUSD
        for indicator in config:
            if indicator["name"] == "obv":
                df["obv"] = ta.obv(df["close"], df["tick_volume"])

            elif indicator["name"] == "vwap":
                df["vwap"] = ta.vwap(
                    df["high"],
                    df["low"],
                    df["close"],
                    df["tick_volume"]
                )

        return df

    @staticmethod
    def add_custom_gold_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators specifically useful for gold trading."""
        # Daily price change (useful for gold's daily volatility)
        df["daily_change"] = df["close"].pct_change(1) * 100

        # Weekly price change (for longer-term trends in gold)
        df["weekly_change"] = df["close"].pct_change(5) * 100  # Assuming 5 trading days

        # Gold-specific volatility indicator (higher timeframe ATR ratio)
        if "atr_14" in df.columns and "atr_21" in df.columns:
            df["gold_vol_ratio"] = df["atr_14"] / df["atr_21"]

        # Price distance from key MA in ATR units (useful for gold mean reversion)
        if "ema_50" in df.columns and "atr_14" in df.columns:
            df["price_to_ema50_atr"] = (df["close"] - df["ema_50"]) / df["atr_14"]

        # Extreme moves indicator (for capturing gold's sharp moves)
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"] * 100
        df["extreme_move"] = np.where(
            df["high_low_range"] > 1.5 * df["high_low_range"].rolling(20).mean(),
            1, 0
        )

        return df

    @staticmethod
    def add_usd_correlation_features(
            df: pd.DataFrame,
            usd_index_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Add USD correlation features (critical for XAUUSD)."""
        if usd_index_df is None:
            return df

        # Resample USD index to match df frequency if needed
        if usd_index_df.index.freq != df.index.freq:
            usd_index_df = usd_index_df.reindex(df.index, method="ffill")

        # Add USD index features
        df["usd_index"] = usd_index_df["Close"]
        df["usd_index_change"] = usd_index_df["Close"].pct_change(1) * 100

        # Inverse correlation feature (gold usually moves opposite to USD)
        df["inverse_usd"] = -df["usd_index_change"]

        # Rolling correlation between gold and USD (negative expected)
        df["gold_usd_corr"] = (
            df["close"].rolling(20).corr(df["usd_index"])
        )

        return df

    @staticmethod
    def add_interest_rate_features(
            df: pd.DataFrame,
            rates_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Add interest rate features (important for gold)."""
        if rates_df is None:
            return df

        # Resample rates data to match df frequency if needed
        if rates_df.index.freq != df.index.freq:
            rates_df = rates_df.reindex(df.index, method="ffill")

        # Add interest rate features
        df["us_10y_yield"] = rates_df["Close"]
        df["yield_change"] = rates_df["Close"].pct_change(1) * 100

        # Real rates proxy (important for gold)
        # Ideally we would use inflation data but for simplicity using yield only
        df["real_rate_proxy"] = df["us_10y_yield"]

        # Gold-yield relationship (historically inverse)
        df["gold_yield_ratio"] = df["close"] / df["us_10y_yield"]

        return df

    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels based on recent highs/lows."""
        # Local highs and lows
        df["local_high"] = df["high"].rolling(window).max()
        df["local_low"] = df["low"].rolling(window).min()

        # Price distance to support/resistance as percentage
        df["dist_to_resistance"] = (df["local_high"] - df["close"]) / df["close"] * 100
        df["dist_to_support"] = (df["close"] - df["local_low"]) / df["close"] * 100

        # Support/resistance level breaks
        df["break_resistance"] = np.where(
            df["close"] > df["local_high"].shift(1), 1, 0
        )
        df["break_support"] = np.where(
            df["close"] < df["local_low"].shift(1), 1, 0
        )

        return df

    @staticmethod
    def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add combined signal features useful for machine learning."""
        # Trend strength
        if "ADX_14" in df.columns:
            df["trend_strength"] = np.where(df["ADX_14"] > 25, 1, 0)

        # RSI divergence signal
        if "rsi_14" in df.columns:
            price_higher_high = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
            rsi_lower_high = (df["rsi_14"] < df["rsi_14"].shift(1)) & (df["rsi_14"].shift(1) > df["rsi_14"].shift(2))
            df["bearish_divergence"] = np.where(price_higher_high & rsi_lower_high, 1, 0)

            price_lower_low = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
            rsi_higher_low = (df["rsi_14"] > df["rsi_14"].shift(1)) & (df["rsi_14"].shift(1) < df["rsi_14"].shift(2))
            df["bullish_divergence"] = np.where(price_lower_low & rsi_higher_low, 1, 0)

        # MACD signal
        if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
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

        # Combined signals
        df["buy_signal"] = 0
        df["sell_signal"] = 0

        # Buy conditions
        if all(col in df.columns for col in ["rsi_14_os", "macd_cross_up"]):
            df["buy_signal"] = np.where(
                (df["rsi_14_os"] == 1) & (df["macd_cross_up"] == 1),
                1, df["buy_signal"]
            )

        # Sell conditions
        if all(col in df.columns for col in ["rsi_14_ob", "macd_cross_down"]):
            df["sell_signal"] = np.where(
                (df["rsi_14_ob"] == 1) & (df["macd_cross_down"] == 1),
                1, df["sell_signal"]
            )

        return df
