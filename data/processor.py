import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config.constants import IndicatorType
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger


class DataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        import yaml
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.scalers = {}
        self.indicators = TechnicalIndicators()
        # Initialize logger for the DataProcessor
        self.logger = setup_logger(name="DataProcessorLogger")

    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        data_dict = {}
        for timeframe, path in data_paths.items():
            if os.path.exists(path):
                data_dict[timeframe] = pd.read_csv(path, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded {timeframe} data: {len(data_dict[timeframe])} rows")
            else:
                self.logger.warning(f"File not found - {path}")

        return data_dict

    def load_external_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load external data from CSV files."""
        data_dict = {}
        for name, path in data_paths.items():
            if os.path.exists(path):
                try:
                    data_dict[name] = pd.read_csv(path, index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded external data {name}: {len(data_dict[name])} rows")
                except Exception as e:
                    self.logger.warning(f"Error loading external data - {path}: {str(e)}")
            else:
                self.logger.warning(f"External data file not found - {path}")

        return data_dict

    def add_all_indicators(
            self,
            df: pd.DataFrame,
            external_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Add all configured indicators to the dataframe."""
        # Add configured technical indicators
        for indicator_group in self.config["features"]["technical_indicators"]:
            indicator_type = indicator_group["type"]
            indicators = indicator_group["indicators"]

            if indicator_type == IndicatorType.TREND.value:
                df = self.indicators.add_trend_indicators(df, indicators)
            elif indicator_type == IndicatorType.MOMENTUM.value:
                df = self.indicators.add_momentum_indicators(df, indicators)
            elif indicator_type == IndicatorType.VOLATILITY.value:
                df = self.indicators.add_volatility_indicators(df, indicators)
            elif indicator_type == IndicatorType.VOLUME.value:
                df = self.indicators.add_volume_indicators(df, indicators)

        # Add custom gold indicators
        df = self.indicators.add_custom_gold_indicators(df)

        # Add support/resistance levels
        df = self.indicators.add_support_resistance(df)

        # Add external data features if available
        if external_data and isinstance(external_data, dict):
            if "usd_index" in external_data and external_data["usd_index"] is not None:
                try:
                    df = self.indicators.add_usd_correlation_features(df, external_data["usd_index"])
                except Exception as e:
                    self.logger.warning(f"Could not add USD correlation features: {str(e)}")

            if "us_10y_yield" in external_data and external_data["us_10y_yield"] is not None:
                try:
                    df = self.indicators.add_interest_rate_features(df, external_data["us_10y_yield"])
                except Exception as e:
                    self.logger.warning(f"Could not add interest rate features: {str(e)}")

        # Add signal features
        df = self.indicators.add_signal_features(df)

        return df

    def create_target_variable(
            self,
            df: pd.DataFrame,
            target_type: str = None,
            horizon: int = None
    ) -> pd.DataFrame:
        """Create target variable for supervised learning."""
        if target_type is None:
            target_type = self.config["model"]["prediction_target"]

        if horizon is None:
            horizon = self.config["model"]["prediction_horizon"]

        if target_type == "direction":
            # Binary classification: 1 for price up, 0 for price down
            df[f"target_{horizon}"] = np.where(
                df["close"].shift(-horizon) > df["close"], 1, 0
            )
        elif target_type == "return":
            # Regression: future return
            df[f"target_{horizon}"] = (df["close"].shift(-horizon) / df["close"] - 1) * 100
        elif target_type == "price":
            # Regression: future price
            df[f"target_{horizon}"] = df["close"].shift(-horizon)
        elif target_type == "volatility":
            # Binary classification: 1 for high volatility, 0 for low volatility
            future_volatility = df["high"].shift(-horizon) / df["low"].shift(-horizon) - 1
            volatility_threshold = future_volatility.rolling(100).quantile(0.7)
            df[f"target_{horizon}"] = np.where(
                future_volatility > volatility_threshold, 1, 0
            )

        return df

    def process_data(
            self,
            data_dict: Dict[str, pd.DataFrame],
            external_data: Optional[Dict[str, pd.DataFrame]] = None,
            add_target: bool = True,
            feature_list: Optional[List[str]] = None  # Add this parameter
    ) -> Dict[str, pd.DataFrame]:
        """Process all dataframes with indicators and target variables."""
        processed_data = {}

        for timeframe, df in data_dict.items():
            self.logger.info(f"Processing {timeframe} data...")

            # Add all indicators
            processed_df = self.add_all_indicators(df.copy(), external_data)

            # Create target variable if needed
            if add_target:
                processed_df = self.create_target_variable(processed_df)

            # If a specific feature list is provided, filter to only those features
            if feature_list is not None:
                # Keep only the features that exist in the processed data
                available_features = [f for f in feature_list if f in processed_df.columns]
                missing_features = [f for f in feature_list if f not in processed_df.columns]

                if missing_features:
                    self.logger.warning(f"{len(missing_features)} requested features are not available")

                # Add back the target columns even if they weren't in the feature list
                target_cols = [col for col in processed_df.columns if col.startswith("target_")]
                available_features.extend([col for col in target_cols if col not in available_features])

                # Filter the dataframe
                processed_df = processed_df[available_features]
                self.logger.info(f"  → Filtered to {len(available_features)} specified features")

            processed_data[timeframe] = processed_df
            added_features = processed_df.shape[1] - df.shape[1]
            self.logger.info(f"  → Added {added_features} new features for {timeframe}")

        return processed_data

    def normalize_data(
            self,
            data_dict: Dict[str, pd.DataFrame],
            method: str = "standard",
            fit_scalers: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Normalize data using specified method."""
        normalized_data = {}

        for timeframe, df in data_dict.items():
            self.logger.info(f"Normalizing {timeframe} data...")

            # Separate features from target
            features = df.columns[~df.columns.str.startswith("target_")]
            target_cols = df.columns[df.columns.str.startswith("target_")]

            # Select numerical columns only
            numerical_cols = df[features].select_dtypes(include=np.number).columns

            # Get or create scaler
            if fit_scalers or timeframe not in self.scalers:
                if method == "minmax":
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                else:  # default: standard
                    scaler = StandardScaler()

                if fit_scalers:
                    self.scalers[timeframe] = scaler.fit(df[numerical_cols].dropna())

            # Transform data
            normalized_df = df.copy()
            normalized_df[numerical_cols] = self.scalers[timeframe].transform(
                df[numerical_cols].fillna(df[numerical_cols].mean())
            )

            normalized_data[timeframe] = normalized_df

        return normalized_data

    def split_train_test(
            self,
            data_dict: Dict[str, pd.DataFrame],
            split_ratio: Optional[float] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data into training and testing sets."""
        if split_ratio is None:
            split_ratio = self.config["data"]["split_ratio"]

        train_dict = {}
        test_dict = {}

        for timeframe, df in data_dict.items():
            split_idx = int(len(df) * split_ratio)
            train_dict[timeframe] = df.iloc[:split_idx].copy()
            test_dict[timeframe] = df.iloc[split_idx:].copy()

            self.logger.info(f"{timeframe} - Train: {len(train_dict[timeframe])}, Test: {len(test_dict[timeframe])}")

        return train_dict, test_dict

    def prepare_ml_features(
            self,
            df: pd.DataFrame,
            horizon: Optional[int] = None,
            drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML model training."""
        if horizon is None:
            horizon = self.config["model"]["prediction_horizon"]

        target_col = f"target_{horizon}"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Run create_target_variable first.")

        # Set up a logger for debugging
        logger = setup_logger("DataProcessorLogger")
        logger.debug("Original DataFrame columns: %s", df.columns.tolist())

        # Select all columns except target columns
        feature_cols = df.columns[~df.columns.str.startswith("target_")]
        logger.debug("Feature columns before dropping NaN: %s", feature_cols.tolist())

        # Remove columns that are entirely NaN
        valid_cols = [col for col in feature_cols if not df[col].isna().all()]
        logger.debug("Valid feature columns (non-all NaN): %s", valid_cols)

        X = df[valid_cols]
        y = df[target_col]

        if drop_na:
            mask = ~X.isna().any(axis=1) & ~y.isna()
            pre_drop_shape = X.shape
            X = X[mask]
            y = y[mask]
            dropped = pre_drop_shape[0] - X.shape[0]
            logger.debug("Dropped %d rows with NaNs. New shapes: X: %s, y: %s", dropped, X.shape, y.shape)

        logger.debug("Final ML features columns: %s", X.columns.tolist())
        logger.debug("Final features shape: %s, Target shape: %s", X.shape, y.shape)

        return X, y

    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], suffix: str = "processed") -> None:
        """Save processed data to disk."""
        save_path = self.config["data"]["save_path"]
        symbol = self.config["data"]["symbol"]

        for timeframe, df in data_dict.items():
            filename = f"{save_path}/{symbol}_{timeframe}_{suffix}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
            self.logger.info(f"Saved processed data to {filename}")


def main():
    """Test function to process data."""
    from data.fetcher import MT5DataFetcher

    # Load historical data
    fetcher = MT5DataFetcher()
    latest_mt5_data_paths = fetcher.get_latest_data_paths()
    latest_external_data_paths = fetcher.get_latest_external_data_paths()

    if not latest_mt5_data_paths:
        setup_logger(name="DataProcessorLogger").warning("No historical data found. Fetching data...")
        if fetcher.initialize():
            try:
                data_dict = fetcher.fetch_all_timeframes()
                fetcher.save_data(data_dict)

                external_data = fetcher.fetch_external_data()
                fetcher.save_external_data(external_data)

                latest_mt5_data_paths = fetcher.get_latest_data_paths()
                latest_external_data_paths = fetcher.get_latest_external_data_paths()
            finally:
                fetcher.shutdown()

    processor = DataProcessor()
    data_dict = processor.load_data(latest_mt5_data_paths)
    external_data = processor.load_external_data(latest_external_data_paths)

    # Add indicators and target variable
    processed_data = processor.process_data(data_dict, external_data)

    # Normalize data
    normalized_data = processor.normalize_data(processed_data)

    # Split into train/test sets
    train_data, test_data = processor.split_train_test(normalized_data)

    # Save processed data
    processor.save_processed_data(processed_data)

    # Example: Prepare ML features for one timeframe
    timeframe = "H1"  # Example timeframe
    if timeframe in train_data:
        X_train, y_train = processor.prepare_ml_features(train_data[timeframe])
        processor.logger.info(f"Training features shape: {X_train.shape}, Target shape: {y_train.shape}")
        processor.logger.info(f"Feature columns: {X_train.columns.tolist()[:5]}...")


if __name__ == "__main__":
    main()
