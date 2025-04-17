import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger


class DataProcessor:
    def __init__(self, config_path: str = None):
        # If no config_path is provided, compute it relative to this file's location.
        if config_path is None:
            # Get the directory of this file (data/processor.py)
            processor_dir = os.path.dirname(os.path.abspath(__file__))
            # Assume the project root is one level above the 'data' folder
            project_root = os.path.dirname(processor_dir)
            # Construct the absolute path to the configuration file
            config_path = os.path.join(project_root, "config", "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.scalers = {}
        # Initialize TechnicalIndicators instance (using its internal config loading)
        self.indicators = TechnicalIndicators()
        # Initialize logger for the DataProcessor
        self.logger = setup_logger(name="DataProcessorLogger")

    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        data_dict = {}
        for timeframe, path in data_paths.items():
            if os.path.exists(path):
                try:
                    data_dict[timeframe] = pd.read_csv(path, index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded {timeframe} data: {len(data_dict[timeframe])} rows")
                except Exception as e:
                    self.logger.warning(f"Error loading {timeframe} data from {path}: {str(e)}")
            else:
                self.logger.warning(f"File not found - {path}")
        return data_dict

    def load_split_data(self, split_paths: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load train, validation and test data from paths."""
        split_data = {"train": {}, "validation": {}, "test": {}}

        for split_type, timeframe_paths in split_paths.items():
            for timeframe, path in timeframe_paths.items():
                if os.path.exists(path):
                    try:
                        split_data[split_type][timeframe] = pd.read_csv(path, index_col=0, parse_dates=True)
                        self.logger.info(
                            f"Loaded {split_type} {timeframe} data: {len(split_data[split_type][timeframe])} rows")
                    except Exception as e:
                        self.logger.warning(f"Error loading {split_type} {timeframe} data from {path}: {str(e)}")
                else:
                    self.logger.warning(f"File not found - {path}")

        return split_data

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

        self.logger.info(f"Creating target variable for horizon: {horizon} periods")

        if target_type == "direction":
            # Binary classification: 1 for price up, 0 for price down
            price_change_pct = (df["close"].shift(-horizon) / df["close"] - 1) * 100

            # Increased threshold for gold's volatility (from 0.1% to 0.2%)
            min_change_threshold = 0.2  # More appropriate for gold price movements

            # Create target labels based on price movement
            df[f"target_{horizon}"] = np.where(
                price_change_pct > min_change_threshold, 1,
                np.where(price_change_pct < -min_change_threshold, 0, np.nan)
            )

            # Log the percentage of insignificant movements
            na_count = df[f"target_{horizon}"].isna().sum()
            if na_count > 0:
                self.logger.info(
                    f"Found {na_count} ({na_count / len(df) * 100:.2f}%) rows with insignificant movements"
                )

                # Instead of random filling, use trend continuation (previous value)
                # This maintains the temporal consistency of the data
                df[f"target_{horizon}"] = df[f"target_{horizon}"].fillna(method='ffill')

                # For any remaining NaNs at the beginning, use the next available value
                df[f"target_{horizon}"] = df[f"target_{horizon}"].fillna(method='bfill')

                # If there are still NaNs, use balanced class distribution
                if df[f"target_{horizon}"].isna().any():
                    # Get current class distribution for balancing
                    class_counts = df[f"target_{horizon}"].value_counts(normalize=True)
                    if len(class_counts) == 2:
                        # Fill remaining NaNs with the minority class to improve balance
                        minority_class = 0 if class_counts.get(0, 0) < class_counts.get(1, 0) else 1
                        df[f"target_{horizon}"] = df[f"target_{horizon}"].fillna(minority_class)
                    else:
                        # Default to 0 if we can't determine distribution
                        df[f"target_{horizon}"] = df[f"target_{horizon}"].fillna(0)

                self.logger.info(f"Filled insignificant movements using trend continuation and balancing")

            df[f"target_{horizon}"] = df[f"target_{horizon}"].astype(int)

            # Log the final class distribution
            up_pct = df[f"target_{horizon}"].mean() * 100
            self.logger.info(f"Target variable distribution: {up_pct:.2f}% up, {100 - up_pct:.2f}% down")
            self.logger.info(f"Total samples for training: {len(df)}")

            # Add a warning if the class distribution is highly imbalanced
            if up_pct < 30 or up_pct > 70:
                self.logger.warning(
                    f"Highly imbalanced class distribution detected: {up_pct:.2f}% up, {100 - up_pct:.2f}% down. "
                    f"Consider adjusting the threshold or implementing class weights during training."
                )

        elif target_type == "return":
            df[f"target_{horizon}"] = (df["close"].shift(-horizon) / df["close"] - 1) * 100
        elif target_type == "price":
            df[f"target_{horizon}"] = df["close"].shift(-horizon)
        elif target_type == "volatility":
            future_volatility = df["high"].shift(-horizon) / df["low"].shift(-horizon) - 1
            volatility_threshold = future_volatility.rolling(100).quantile(0.7)
            df[f"target_{horizon}"] = np.where(future_volatility > volatility_threshold, 1, 0)
        else:
            self.logger.warning(f"Unknown target type: {target_type}. No target variable created.")
        return df

    def process_data(
            self,
            data_dict: Dict[str, pd.DataFrame],
            external_data: Optional[Dict[str, pd.DataFrame]] = None,
            add_target: bool = True,
            feature_list: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Process all dataframes by adding technical indicators and target variables."""
        processed_data = {}
        for timeframe, df in data_dict.items():
            self.logger.info(f"Processing {timeframe} data...")

            # Focus on H1 timeframe for gold trading; skip others if H1 data exists.
            if timeframe != "H1" and "H1" in data_dict:
                self.logger.info(f"Skipping {timeframe} as we're focusing on H1 for gold trading")
                continue

            processed_df = df.copy()

            # Check if we have enough data for indicator calculation
            if len(processed_df) < 200:  # Minimum data needed for EMA-200
                self.logger.warning(
                    f"Dataset may be too small ({len(processed_df)} rows) for reliable indicator calculation")

            # Add all technical indicators using the TechnicalIndicators class
            self.logger.info(f"Adding technical indicators to {timeframe} data...")
            before_indicators = processed_df.shape
            processed_df = self.indicators.add_all_indicators(processed_df, external_data)
            after_indicators = processed_df.shape
            self.logger.info(f"Added {after_indicators[1] - before_indicators[1]} indicator columns")

            # Check for constant columns after adding indicators
            for col in processed_df.columns:
                if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                    unique_vals = processed_df[col].nunique()
                    if unique_vals <= 1:
                        self.logger.warning(
                            f"Column {col} has only {unique_vals} unique value(s) after indicator calculation")

            # Add target variables if requested
            if add_target:
                processed_df = self.create_target_variable(processed_df)

            # Handle NaN values: use forward and backward filling for non-target columns.
            nan_count = processed_df.isna().sum().sum()
            if nan_count > 0:
                self.logger.info(f"Found {nan_count} total NaN values across the dataframe")
                feature_cols = processed_df.columns[~processed_df.columns.str.startswith("target_")]

                # Log columns with many NaN values
                col_nan_counts = processed_df[feature_cols].isna().sum()
                cols_with_nans = col_nan_counts[col_nan_counts > 0]
                if not cols_with_nans.empty:
                    self.logger.info(f"Columns with NaN values: {cols_with_nans.to_dict()}")

                # Fill NaN values with forward/backward fill
                processed_df[feature_cols] = processed_df[feature_cols].fillna(method='ffill').fillna(method='bfill')

                # Check if any NaNs remain
                remaining_nans = processed_df.isna().sum().sum()
                if remaining_nans > 0:
                    self.logger.warning(
                        f"There are still {remaining_nans} NaN values remaining after fill; these will be handled later during feature preparation.")

            # If a specific feature list is provided, subset the dataframe accordingly.
            if feature_list is not None:
                available_features = [f for f in feature_list if f in processed_df.columns]
                missing_features = [f for f in feature_list if f not in processed_df.columns]
                if missing_features:
                    self.logger.warning(
                        f"{len(missing_features)} requested features are not available: {missing_features[:10]}")
                target_cols = [col for col in processed_df.columns if col.startswith("target_")]
                available_features.extend([col for col in target_cols if col not in available_features])
                processed_df = processed_df[available_features]
                self.logger.info(f"Filtered processed data to {len(available_features)} specified features")

            processed_data[timeframe] = processed_df
            added_features = processed_df.shape[1] - df.shape[1]
            self.logger.info(f"Added {added_features} new features for {timeframe}")

        return processed_data

    def process_split_data(
            self,
            split_data: Dict[str, Dict[str, pd.DataFrame]],
            external_data: Optional[Dict[str, pd.DataFrame]] = None,
            add_target: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process train, validation, and test data separately."""
        processed_split_data = {"train": {}, "validation": {}, "test": {}}

        # Process each split separately to prevent data leakage
        for split_type, timeframe_data in split_data.items():
            self.logger.info(f"Processing {split_type} data...")
            processed_split_data[split_type] = self.process_data(timeframe_data, external_data, add_target)

        return processed_split_data

    def normalize_data(
            self,
            data_dict: Dict[str, pd.DataFrame],
            method: str = "standard",
            fit_scalers: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Normalize data using StandardScaler (or other methods if extended)."""
        normalized_data = {}
        for timeframe, df in data_dict.items():
            self.logger.info(f"Normalizing {timeframe} data...")

            # Identify columns to normalize
            features = df.columns[~df.columns.str.startswith("target_")]
            numerical_cols = df[features].select_dtypes(include=np.number).columns

            # Exclude columns that should not be normalized (e.g., signal columns)
            non_normalize_cols = ['buy_signal', 'sell_signal', 'break_resistance', 'break_support',
                                  'macd_cross_up', 'macd_cross_down']
            normalize_cols = [col for col in numerical_cols if col not in non_normalize_cols]

            # Log stats before normalization
            self.logger.info(
                f"Pre-normalization stats: Mean={df[normalize_cols].mean().mean():.4f}, Std={df[normalize_cols].std().mean():.4f}")

            # Fit scalers on training data if needed
            if fit_scalers or timeframe not in self.scalers:
                scaler = StandardScaler()
                train_size = int(len(df) * self.config["data"]["train_ratio"])

                # Fill missing values in training data before fitting the scaler
                train_data = df.iloc[:train_size].copy()
                train_data[normalize_cols] = train_data[normalize_cols].fillna(train_data[normalize_cols].median())

                # Fit the scaler and store it
                self.scalers[timeframe] = scaler.fit(train_data[normalize_cols])
                self.logger.info(f"Fitted scaler on {train_size} training samples for {timeframe}")

            # Apply the scaler to the entire dataset
            normalized_df = df.copy()
            normalized_df[normalize_cols] = self.scalers[timeframe].transform(
                df[normalize_cols].fillna(df[normalize_cols].median())
            )

            # Log stats after normalization
            self.logger.info(
                f"Post-normalization stats: Mean={normalized_df[normalize_cols].mean().mean():.4f}, Std={normalized_df[normalize_cols].std().mean():.4f}")

            # Check for columns that have abnormal values after normalization
            for col in normalize_cols:
                col_stats = {
                    'min': normalized_df[col].min(),
                    'max': normalized_df[col].max(),
                    'mean': normalized_df[col].mean(),
                    'std': normalized_df[col].std(),
                    'nunique': normalized_df[col].nunique()
                }

                # Flag potential issues
                if abs(col_stats['mean']) > 3 or col_stats['std'] < 0.1 or col_stats['nunique'] < 5:
                    self.logger.warning(f"Column {col} may have normalization issues: {col_stats}")

            normalized_data[timeframe] = normalized_df

        return normalized_data

    def normalize_split_data(
            self,
            split_data: Dict[str, Dict[str, pd.DataFrame]],
            method: str = "standard"
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Normalize train, validation, and test data separately, using training data for fitting the scalers."""
        normalized_split_data = {"train": {}, "validation": {}, "test": {}}

        # First, normalize training data and fit scalers
        normalized_split_data["train"] = self.normalize_data(split_data["train"], method, fit_scalers=True)

        # Then normalize validation and test data using the same scalers
        for split_type in ["validation", "test"]:
            if split_type in split_data and split_data[split_type]:
                normalized_split_data[split_type] = self.normalize_data(
                    split_data[split_type], method, fit_scalers=False
                )

        return normalized_split_data

    def prepare_ml_features(
            self,
            df: pd.DataFrame,
            horizon: Optional[int] = None,
            drop_na: bool = True,
            remove_constant_cols: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare ML features and target from the raw dataframe."""
        if horizon is None:
            horizon = self.config["model"]["prediction_horizon"]

        # Ensure target column exists
        target_col = f"target_{horizon}"
        if target_col not in df.columns:
            self.logger.info(f"Target column '{target_col}' not found. Creating it now.")
            df = self.create_target_variable(df, horizon=horizon)

        if target_col not in df.columns:
            self.logger.error(f"Failed to create target column '{target_col}'")
            # Return empty DataFrame and Series rather than raising an error
            return pd.DataFrame(), pd.Series()

        self.logger.debug("Original DataFrame columns: %s", df.columns.tolist())

        # Get feature columns
        feature_cols = df.columns[~df.columns.str.startswith("target_")]
        self.logger.debug("Feature columns before dropping NaN: %s", feature_cols.tolist())

        # Define relevant features for gold trading
        relevant_features = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread',
            'ema_9', 'ema_21', 'ema_55',
            'MACD_12_26_9', 'MACDs_12_26_9',
            'ema_9_21_cross',
            'rsi_14',
            'atr_14', 'atr_pct_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
            'daily_change',
            'high_low_range',
            'local_high', 'local_low',
            'dist_to_resistance', 'dist_to_support',
            'break_resistance', 'break_support',
            'macd_cross_up', 'macd_cross_down',
            'buy_signal', 'sell_signal'
        ]

        # Print all available columns for debugging
        self.logger.info(f"Available columns in data: {df.columns.tolist()}")

        # Filter features based on availability
        available_features = [f for f in relevant_features if f in feature_cols]
        self.logger.info(f"Using {len(available_features)} relevant features for gold trading")
        self.logger.info(f"Selected features: {available_features}")

        # Check for constant columns - only remove if explicitly requested
        constant_cols = []
        if remove_constant_cols:
            for col in available_features:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)

            if constant_cols:
                self.logger.warning(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
                available_features = [f for f in available_features if f not in constant_cols]
        elif not remove_constant_cols and any(df[col].nunique() <= 1 for col in available_features):
            # Just log but don't remove the constant columns
            constant_cols = [col for col in available_features if df[col].nunique() <= 1]
            self.logger.info(f"Found {len(constant_cols)} constant columns but keeping them: {constant_cols}")

        # Remove columns that are entirely NaN
        valid_cols = [col for col in available_features if not df[col].isna().all()]
        if len(valid_cols) < len(available_features):
            self.logger.warning(f"Removed {len(available_features) - len(valid_cols)} columns that were entirely NaN")
            self.logger.warning(f"Columns removed: {set(available_features) - set(valid_cols)}")

        self.logger.debug("Valid feature columns (non-all NaN): %s", valid_cols)

        # Check if we have any valid features left
        if not valid_cols:
            self.logger.error("No valid features available after filtering")
            return pd.DataFrame(), pd.Series()

        # Create feature matrix and target vector
        try:
            X = df[valid_cols].copy()
            y = df[target_col].copy()

            # Fill remaining NaN values with median
            X = X.fillna(X.median())

            # Drop rows with remaining NaN values if requested
            if drop_na:
                na_mask = X.isna().any(axis=1) | y.isna()
                if na_mask.any():
                    pre_drop_shape = X.shape
                    X = X[~na_mask].copy()
                    y = y[~na_mask].copy()
                    dropped = pre_drop_shape[0] - X.shape[0]
                    self.logger.debug("Dropped %d rows with NaNs. New shapes: X: %s, y: %s", dropped, X.shape, y.shape)

            self.logger.debug("Final ML features columns: %s", X.columns.tolist())
            self.logger.debug("Final features shape: %s, Target shape: %s", X.shape, y.shape)

            # Final validation checks
            if X.empty or y.empty:
                self.logger.error("After filtering, no data remains for training. Check your data preprocessing steps.")
                return pd.DataFrame(), pd.Series()

            if not y.empty:
                class_counts = y.value_counts()
                self.logger.info(f"Target class distribution: {class_counts}")
                if len(class_counts) > 1:
                    min_class = class_counts.min()
                    max_class = class_counts.max()
                    if min_class / max_class < 0.1:
                        self.logger.warning("Severe class imbalance detected in target variable!")

            return X, y

        except Exception as e:
            self.logger.error(f"Error in prepare_ml_features: {str(e)}")
            import traceback
            self.logger.error(f"Full error traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.Series()


    def save_processed_split_data(self, split_data: Dict[str, Dict[str, pd.DataFrame]], suffix: str = "processed") -> \
            Dict[str, Dict[str, str]]:
        """Save processed train, validation and test data to separate directories."""
        symbol = self.config["data"]["symbol"]
        saved_paths = {"train": {}, "validation": {}, "test": {}}

        # Get paths from config
        train_path = self.config["data"].get("train_path", "data_output/processed_data/train")
        val_path = self.config["data"].get("validation_path", "data_output/processed_data/validation")
        test_path = self.config["data"].get("test_path", "data_output/processed_data/test")

        for split_name, timeframe_data in split_data.items():
            # Determine the appropriate path
            if split_name == "train":
                path = train_path
            elif split_name == "validation":
                path = val_path
            else:  # test
                path = test_path

            # Save each timeframe
            for timeframe, df in timeframe_data.items():
                filename = os.path.join(path, f"{symbol}_{timeframe}_{suffix}.csv")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                df.to_csv(filename)
                saved_paths[split_name][timeframe] = filename
                self.logger.info(f"Saved {split_name} {timeframe} processed data to {filename}")

        return saved_paths
