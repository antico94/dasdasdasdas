import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
import yaml


class DataStorage:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.base_path = self.config["data"]["save_path"]
        os.makedirs(self.base_path, exist_ok=True)

    def save_dataframe(
            self,
            df: pd.DataFrame,
            name: str,
            include_timestamp: bool = True
    ) -> str:
        """Save a DataFrame to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.csv" if timestamp else f"{name}.csv"
        filepath = os.path.join(self.base_path, filename)

        df.to_csv(filepath)
        print(f"Saved DataFrame to {filepath}")
        return filepath

    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        """Load a DataFrame from CSV."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded DataFrame from {filepath}: {len(df)} rows, {df.shape[1]} columns")
        return df

    def save_model(
            self,
            model: Any,
            name: str,
            include_timestamp: bool = True,
            metadata: Optional[Dict] = None
    ) -> str:
        """Save a trained model to disk."""
        models_dir = os.path.join(self.base_path, "../models")
        os.makedirs(models_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.joblib" if timestamp else f"{name}.joblib"
        filepath = os.path.join(models_dir, filename)

        # Save model
        joblib.dump(model, filepath)

        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(models_dir,
                                         f"{name}_{timestamp}_metadata.pkl" if timestamp else f"{name}_metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

        print(f"Saved model to {filepath}")
        return filepath

    def load_model(self, filepath: str) -> Any:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        print(f"Loaded model from {filepath}")

        # Check for metadata
        metadata_path = filepath.replace(".joblib", "_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            print(f"Loaded model metadata: {list(metadata.keys())}")
            return model, metadata

        return model

    def save_scaler(self, scaler: Any, name: str) -> str:
        """Save a fitted scaler to disk."""
        scalers_dir = os.path.join(self.base_path, "../models/scalers")
        os.makedirs(scalers_dir, exist_ok=True)

        filepath = os.path.join(scalers_dir, f"{name}_scaler.joblib")
        joblib.dump(scaler, filepath)
        print(f"Saved scaler to {filepath}")
        return filepath

    def load_scaler(self, filepath: str) -> Any:
        """Load a fitted scaler from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")

        scaler = joblib.load(filepath)
        print(f"Loaded scaler from {filepath}")
        return scaler

    def save_results(
            self,
            results: Dict,
            name: str,
            include_timestamp: bool = True
    ) -> str:
        """Save results (like backtest results) to disk."""
        results_dir = os.path.join(self.base_path, "../results")
        os.makedirs(results_dir, exist_ok=True)

        # If the provided name ends with '.pkl', remove it to avoid duplicating the extension.
        if name.endswith(".pkl"):
            name = name[:-4]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.pkl" if timestamp else f"{name}.pkl"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved results to {filepath}")
        return filepath

    def load_results(self, filepath: str) -> Dict:
        """Load results from disk."""
        # Check if filepath already has .pkl extension to avoid duplication
        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'

        # Make sure we have an absolute path if a relative path was provided
        if not os.path.isabs(filepath):
            if os.path.exists(filepath):
                pass  # Use as is if it exists
            elif filepath.startswith('models/'):
                # Fix path for model-specific files that might be in a different directory
                filepath = os.path.join(self.base_path, "../", filepath)
            else:
                # Default to looking in the results directory
                filepath = os.path.join(self.base_path, "../results", filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "rb") as f:
            results = pickle.load(f)

        print(f"Loaded results from {filepath}")
        return results

    def find_latest_file(self, pattern: str) -> Optional[str]:
        """Find the latest file matching the pattern."""
        matching_files = list(Path(self.base_path).glob(pattern))
        if not matching_files:
            return None

        latest_file = max(matching_files, key=os.path.getctime)
        return str(latest_file)

    def find_latest_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, str]:
        """Find the latest data files."""
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
        """Find the latest processed data files."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        processed_files = {}
        for tf in self.config["data"]["timeframes"]:
            pattern = f"{symbol}_{tf}_processed.csv"
            file_path = os.path.join(self.base_path, pattern)
            if os.path.exists(file_path):
                processed_files[tf] = file_path

        return processed_files

    def find_latest_model(self, model_type: str) -> Optional[str]:
        """Find the latest model file of a given type."""
        models_dir = os.path.join(self.base_path, "../models")
        if not os.path.exists(models_dir):
            return None

        pattern = f"{model_type}_*.joblib"
        matching_files = list(Path(models_dir).glob(pattern))
        if not matching_files:
            return None

        latest_model = max(matching_files, key=os.path.getctime)
        return str(latest_model)


def main():
    """Test function for data storage."""
    storage = DataStorage()

    # Test finding latest data
    latest_files = storage.find_latest_data()
    print("Latest data files:")
    for tf, path in latest_files.items():
        print(f"  {tf}: {path}")

    # Test finding latest processed data
    processed_files = storage.find_latest_processed_data()
    print("\nLatest processed data files:")
    for tf, path in processed_files.items():
        print(f"  {tf}: {path}")


if __name__ == "__main__":
    main()
