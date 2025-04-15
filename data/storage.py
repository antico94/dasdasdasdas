import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import yaml
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataStorage:
    def __init__(self, config_path: Optional[str] = None):
        # Compute the project root based on the location of this file.
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parents[1]  # e.g., C:/Users/ameiu/PycharmProjects/GoldML

        # Determine the configuration file path.
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        logger.info(f"Attempting to load config from: {config_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        # Load configuration.
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Convert the save_path (from config) to an absolute path relative to the project root.
        save_path = self.config["data"]["save_path"]
        save_path = Path(save_path)
        if not save_path.is_absolute():
            self.base_path = self.project_root / save_path
        else:
            self.base_path = save_path

        # Ensure the base path exists.
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base path set to: {self.base_path}")

    def save_dataframe(self, df: pd.DataFrame, name: str, include_timestamp: bool = True) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.csv" if timestamp else f"{name}.csv"
        filepath = self.base_path / filename

        df.to_csv(filepath)
        logger.info(f"Saved DataFrame to {filepath}")
        return str(filepath)

    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        logger.info(f"Loaded DataFrame from {fp}: {len(df)} rows, {df.shape[1]} columns")
        return df

    def save_model(self, model: Any, name: str, include_timestamp: bool = True, metadata: Optional[Dict] = None) -> str:
        # Build the models directory relative to the project root.
        models_dir = self.project_root / "data_output" / "trained_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create the file name; include timestamp only if desired.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.joblib" if timestamp else f"{name}.joblib"
        filepath = models_dir / filename

        # Save the model.
        joblib.dump(model, filepath)

        if metadata:
            metadata_filename = f"{name}_{timestamp}_metadata.pkl" if timestamp else f"{name}_metadata.pkl"
            metadata_path = models_dir / metadata_filename
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved model metadata to {metadata_path}")

        logger.info(f"Saved model to {filepath}")
        return str(filepath)

    def load_model(self, filepath: str) -> Any:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Model file not found: {fp}")
        model = joblib.load(fp)
        logger.info(f"Loaded model from {fp}")

        # Load model metadata if available.
        metadata_path = fp.with_name(fp.stem + "_metadata.pkl")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded model metadata: {list(metadata.keys())}")
            return model, metadata
        return model

    def save_scaler(self, scaler: Any, name: str) -> str:
        scalers_dir = self.project_root / "data_output" / "models" / "scalers"
        scalers_dir.mkdir(parents=True, exist_ok=True)
        filepath = scalers_dir / f"{name}_scaler.joblib"
        joblib.dump(scaler, filepath)
        logger.info(f"Saved scaler to {filepath}")
        return str(filepath)

    def load_scaler(self, filepath: str) -> Any:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Scaler file not found: {fp}")
        scaler = joblib.load(fp)
        logger.info(f"Loaded scaler from {fp}")
        return scaler

    def save_results(self, results: Dict, name: str, include_timestamp: bool = True) -> str:
        # Fetch results save path from config. If not set, use a default.
        config_results_path = self.config["data"].get("results_path", "data_output/results/models")
        results_dir = self.project_root / config_results_path
        results_dir.mkdir(parents=True, exist_ok=True)

        # Remove the .pkl extension from name if already present.
        if name.endswith(".pkl"):
            name = name[:-4]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{name}_{timestamp}.pkl" if timestamp else f"{name}.pkl"
        filepath = results_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Saved results to {filepath}")
        return str(filepath)

    def load_results(self, filepath: str) -> Dict:
        fp = Path(filepath)
        if not fp.is_absolute():
            # Assume relative to the default results directory.
            fp = self.project_root / "data_output" / "results" / fp
        if not fp.exists():
            raise FileNotFoundError(f"Results file not found: {fp}")
        with open(fp, "rb") as f:
            results = pickle.load(f)
        logger.info(f"Loaded results from {fp}")
        return results

    def find_latest_file(self, pattern: str) -> Optional[str]:
        matching_files = list(self.base_path.glob(pattern))
        if not matching_files:
            return None
        latest_file = max(matching_files, key=lambda f: f.stat().st_ctime)
        return str(latest_file)

    def find_latest_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, str]:
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
        if symbol is None:
            symbol = self.config["data"]["symbol"]
        processed_files = {}
        for tf in self.config["data"]["timeframes"]:
            filename = f"{symbol}_{tf}_processed.csv"
            file_path = self.base_path / filename
            if file_path.exists():
                processed_files[tf] = str(file_path)
        return processed_files

    def find_latest_model(self, model_type: str) -> Optional[str]:
        models_dir = self.project_root / "data_output" / "trained_models"
        if not models_dir.exists():
            return None
        pattern = f"{model_type}_*.joblib"
        matching_files = list(models_dir.glob(pattern))
        if not matching_files:
            return None
        latest_model = max(matching_files, key=lambda f: f.stat().st_ctime)
        return str(latest_model)


def main():
    storage = DataStorage()
    latest_files = storage.find_latest_data()
    print("Latest data files:")
    for tf, path in latest_files.items():
        print(f"  {tf}: {path}")
    processed_files = storage.find_latest_processed_data()
    print("\nLatest processed data files:")
    for tf, path in processed_files.items():
        print(f"  {tf}: {path}")


if __name__ == "__main__":
    main()
