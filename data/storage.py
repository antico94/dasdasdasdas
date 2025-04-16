import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
import yaml
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PathManager:
    """Manages file paths for data storage operations."""

    def __init__(self, project_root: Path, config: Dict):
        self.project_root = project_root
        self.config = config

        # Set up base paths
        self.base_path = self._resolve_path(config["data"]["save_path"])
        self.models_dir = self.project_root / "data_output" / "trained_models"
        self.results_dir = self._resolve_path(config["data"].get("results_path", "data_output/results/models"))
        self.train_path = self._resolve_path(config["data"].get("train_path", "data_output/processed_data/train"))
        self.validation_path = self._resolve_path(
            config["data"].get("validation_path", "data_output/processed_data/validation"))
        self.test_path = self._resolve_path(config["data"].get("test_path", "data_output/processed_data/test"))
        self.scalers_dir = self.project_root / "data_output" / "models" / "scalers"

        # Create directories
        self._create_dirs()

    def _resolve_path(self, path_str: str) -> Path:
        """Convert string path to absolute Path object."""
        path = Path(path_str)
        if not path.is_absolute():
            return self.project_root / path
        return path

    def _create_dirs(self) -> None:
        """Create all necessary directories."""
        dirs = [
            self.base_path,
            self.models_dir,
            self.results_dir,
            self.train_path,
            self.validation_path,
            self.test_path,
            self.scalers_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_split_path(self, split_type: str) -> Path:
        """Get the path for a specific data split."""
        if split_type == "train":
            return self.train_path
        elif split_type == "validation":
            return self.validation_path
        elif split_type == "test":
            return self.test_path
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def build_filename(self, name: str, suffix: str = "", include_timestamp: bool = True) -> str:
        """Build a filename with optional timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        if suffix:
            return f"{name}_{suffix}_{timestamp}" if timestamp else f"{name}_{suffix}"
        return f"{name}_{timestamp}" if timestamp else name


class FileManager:
    """Handles file operations like finding and matching files."""

    @staticmethod
    def find_latest_file(directory: Path, pattern: str) -> Optional[str]:
        """Find the latest file matching a pattern in a directory."""
        matching_files = list(directory.glob(pattern))
        if not matching_files:
            return None
        latest_file = max(matching_files, key=lambda f: f.stat().st_ctime)
        return str(latest_file)

    @staticmethod
    def find_files_by_pattern(directory: Path, pattern: str) -> List[str]:
        """Find all files matching a pattern in a directory."""
        matching_files = list(directory.glob(pattern))
        return [str(f) for f in matching_files]


class DataStorage:
    """Handles storage and retrieval of data, models, and results."""

    def __init__(self, config_path: Optional[str] = None):
        # Find project root
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parents[1]  # e.g., C:/Users/ameiu/PycharmProjects/GoldML

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize path and file managers
        self.path_manager = PathManager(self.project_root, self.config)
        self.file_manager = FileManager()

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file."""
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        logger.info(f"Attempting to load config from: {config_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    # DataFrame Operations

    def save_dataframe(self, df: pd.DataFrame, name: str, include_timestamp: bool = True) -> str:
        """Save DataFrame to CSV file."""
        filename = self.path_manager.build_filename(name, include_timestamp=include_timestamp) + ".csv"
        filepath = self.path_manager.base_path / filename

        df.to_csv(filepath)
        logger.info(f"Saved DataFrame to {filepath}")
        return str(filepath)

    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from CSV file."""
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        logger.info(f"Loaded DataFrame from {fp}: {len(df)} rows, {df.shape[1]} columns")
        return df

    # Model Operations

    def save_model(self, model: Any, name: str, include_timestamp: bool = True, metadata: Optional[Dict] = None) -> str:
        """Save model to file with optional metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.joblib" if timestamp else f"{name}.joblib"
        filepath = self.path_manager.models_dir / filename

        # Save model
        joblib.dump(model, filepath)

        # Save metadata if provided
        if metadata:
            metadata_filename = f"{name}_{timestamp}_metadata.pkl" if timestamp else f"{name}_metadata.pkl"
            metadata_path = self.path_manager.models_dir / metadata_filename
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved model metadata to {metadata_path}")

        logger.info(f"Saved model to {filepath}")
        return str(filepath)

    def load_model(self, filepath: str) -> Union[Any, Tuple[Any, Dict]]:
        """Load model from file and optional metadata."""
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Model file not found: {fp}")

        # Load model
        model = joblib.load(fp)
        logger.info(f"Loaded model from {fp}")

        # Try to load metadata
        metadata_path = fp.with_name(fp.stem + "_metadata.pkl")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded model metadata: {list(metadata.keys())}")
            return model, metadata

        return model

    # Results Operations

    def save_results(self, results: Dict, name: str, include_timestamp: bool = True) -> str:
        """Save results dictionary to file."""
        # Remove .pkl extension if present
        if name.endswith(".pkl"):
            name = name[:-4]

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.pkl" if timestamp else f"{name}.pkl"
        filepath = self.path_manager.results_dir / filename

        # Save results
        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Saved results to {filepath}")
        return str(filepath)

    def load_results(self, filepath: str) -> Dict:
        """Load results dictionary from file."""
        fp = Path(filepath)
        if not fp.is_absolute():
            # Assume relative to the default results directory
            fp = self.path_manager.results_dir / fp

        if not fp.exists():
            raise FileNotFoundError(f"Results file not found: {fp}")

        with open(fp, "rb") as f:
            results = pickle.load(f)

        logger.info(f"Loaded results from {fp}")
        return results

    # Scaler Operations

    def save_scaler(self, scaler: Any, name: str) -> str:
        """Save scaler to file."""
        filepath = self.path_manager.scalers_dir / f"{name}_scaler.joblib"
        joblib.dump(scaler, filepath)
        logger.info(f"Saved scaler to {filepath}")
        return str(filepath)

    def load_scaler(self, filepath: str) -> Any:
        """Load scaler from file."""
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Scaler file not found: {fp}")

        scaler = joblib.load(fp)
        logger.info(f"Loaded scaler from {fp}")
        return scaler

    # Data Finding Operations

    def find_latest_file(self, pattern: str) -> Optional[str]:
        """Find the latest file matching a pattern in base path."""
        return self.file_manager.find_latest_file(self.path_manager.base_path, pattern)

    def find_latest_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, str]:
        """Find the latest data files for a symbol and optional timeframe."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        if timeframe:
            pattern = f"{symbol}_{timeframe}_*.csv"
            latest = self.find_latest_file(pattern)
            return {timeframe: latest} if latest else {}
        else:
            latest_files = {}
            for tf in self.config["data"]["timeframes"]:
                pattern = f"{symbol}_{tf}_*.csv"
                latest = self.find_latest_file(pattern)
                if latest:
                    latest_files[tf] = latest
            return latest_files

    def find_latest_processed_data(self, symbol: Optional[str] = None) -> Dict[str, str]:
        """Find the latest processed data files for a symbol."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        processed_files = {}
        for tf in self.config["data"]["timeframes"]:
            filename = f"{symbol}_{tf}_processed.csv"
            file_path = self.path_manager.base_path / filename
            if file_path.exists():
                processed_files[tf] = str(file_path)

        return processed_files

    def find_latest_model(self, model_type: str) -> Optional[str]:
        """Find the latest model file of a specific type."""
        pattern = f"{model_type}_*.joblib"
        return self.file_manager.find_latest_file(self.path_manager.models_dir, pattern)

    # Split Data Operations

    def find_latest_split_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[
        str, Dict[str, str]]:
        """Find the latest data files in each of the train, validation, and test directories."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        split_paths = {"train": {}, "validation": {}, "test": {}}

        for split_type in split_paths.keys():
            dir_path = self.path_manager.get_split_path(split_type)

            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                continue

            if timeframe:
                # Find latest file for specific timeframe
                pattern = f"{symbol}_{timeframe}_*.csv"
                latest_file = self.file_manager.find_latest_file(dir_path, pattern)
                if latest_file:
                    split_paths[split_type][timeframe] = latest_file
            else:
                # Find latest files for all timeframes
                for tf in self.config["data"]["timeframes"]:
                    pattern = f"{symbol}_{tf}_*.csv"
                    latest_file = self.file_manager.find_latest_file(dir_path, pattern)
                    if latest_file:
                        split_paths[split_type][tf] = latest_file

        return split_paths

    def save_split_dataframes(self, split_data: Dict[str, Dict[str, pd.DataFrame]], name: str,
                              include_timestamp: bool = True) -> Dict[str, Dict[str, str]]:
        """Save train, validation, and test dataframes to their respective directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        saved_paths = {"train": {}, "validation": {}, "test": {}}

        for split_type, timeframe_data in split_data.items():
            split_dir = self.path_manager.get_split_path(split_type)

            for tf, df in timeframe_data.items():
                filename = f"{name}_{tf}_{timestamp}.csv" if timestamp else f"{name}_{tf}.csv"
                filepath = split_dir / filename

                df.to_csv(filepath)
                saved_paths[split_type][tf] = str(filepath)
                logger.info(f"Saved {split_type} {tf} data to {filepath}")

        return saved_paths

    def load_split_data(self, split_paths: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load train, validation, and test data from paths."""
        split_data = {"train": {}, "validation": {}, "test": {}}

        for split_type, timeframe_paths in split_paths.items():
            for timeframe, path in timeframe_paths.items():
                fp = Path(path)
                if fp.exists():
                    try:
                        split_data[split_type][timeframe] = pd.read_csv(fp, index_col=0, parse_dates=True)
                        logger.info(
                            f"Loaded {split_type} {timeframe} data from {fp}: {len(split_data[split_type][timeframe])} rows")
                    except Exception as e:
                        logger.warning(f"Error loading {split_type} {timeframe} data from {path}: {str(e)}")
                else:
                    logger.warning(f"File not found: {path}")

        return split_data

    def find_latest_processed_split_data(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """Find the latest processed data files in train, validation, and test directories."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        processed_files = {"train": {}, "validation": {}, "test": {}}
        suffix = "processed"

        for split_type in processed_files.keys():
            dir_path = self.path_manager.get_split_path(split_type)

            if not dir_path.exists():
                continue

            for tf in self.config["data"]["timeframes"]:
                # First try to find processed files
                filename = f"{symbol}_{tf}_{suffix}.csv"
                file_path = dir_path / filename

                if file_path.exists():
                    processed_files[split_type][tf] = str(file_path)
                else:
                    # Fall back to latest file for this timeframe
                    pattern = f"{symbol}_{tf}_*.csv"
                    latest_file = self.file_manager.find_latest_file(dir_path, pattern)
                    if latest_file:
                        processed_files[split_type][tf] = latest_file

        return processed_files

    def load_train_val_test_data(self, timeframe: str, processor: Any = None) -> Dict[
        str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Load pre-split train, validation, and test data for a specific timeframe.

        Returns a dictionary with 'train', 'validation', and 'test' keys, each containing a tuple of (X, y).
        """
        split_paths = self.find_latest_split_data()

        if processor is None:
            # Import here to avoid circular imports
            from data.processor import DataProcessor
            processor = DataProcessor()

        result = {}

        # Load training data
        if "train" in split_paths and timeframe in split_paths["train"]:
            train_data_dict = processor.load_data({timeframe: split_paths["train"][timeframe]})
            train_df = train_data_dict[timeframe]
            X_train, y_train = processor.prepare_ml_features(train_df, horizon=1)
            result['train'] = (X_train, y_train)
            logger.info(f"Loaded training data: {len(train_df)} rows, {X_train.shape[1]} features")
        else:
            logger.warning(f"No training data found for {timeframe}")

        # Load validation data
        if "validation" in split_paths and timeframe in split_paths["validation"]:
            val_data_dict = processor.load_data({timeframe: split_paths["validation"][timeframe]})
            val_df = val_data_dict[timeframe]
            X_val, y_val = processor.prepare_ml_features(val_df, horizon=1)
            result['validation'] = (X_val, y_val)
            logger.info(f"Loaded validation data: {len(val_df)} rows, {X_val.shape[1]} features")
        else:
            logger.warning(f"No validation data found for {timeframe}")

        # Load test data
        if "test" in split_paths and timeframe in split_paths["test"]:
            test_data_dict = processor.load_data({timeframe: split_paths["test"][timeframe]})
            test_df = test_data_dict[timeframe]
            X_test, y_test = processor.prepare_ml_features(test_df, horizon=1)
            result['test'] = (X_test, y_test)
            logger.info(f"Loaded test data: {len(test_df)} rows, {X_test.shape[1]} features")
        else:
            logger.warning(f"No test data found for {timeframe}")

        return result