import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.constants import ModelType, PredictionTarget
from data.processor import DataProcessor
from data.storage import DataStorage
from models.feature_selection import FeatureSelector
from models.base import BaseModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel
from utils.logger import setup_logger

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.use_feature_selection = config.get('model', {}).get('feature_selection', True)
        self.use_cross_validation = config.get('model', {}).get('cross_validation', True)
        self.feature_selector = FeatureSelector(config)
        # Initialize logger for this class
        self.logger = setup_logger(name="ModelTrainerLogger")
        self.logger.debug("Initialized ModelTrainer with model_type: %s, target_type: %s, feature_selection: %s",
                            self.model_type, self.target_type, self.use_feature_selection)

    def create_model(self) -> BaseModel:
        """Create a model based on configuration."""
        if self.model_type == ModelType.RANDOM_FOREST.value:
            return RandomForestModel(self.config)
        elif self.model_type == ModelType.XGBOOST.value:
            return XGBoostModel(self.config)
        elif self.model_type == ModelType.LSTM.value:
            return LSTMModel(self.config)
        elif self.model_type == ModelType.ENSEMBLE.value:
            return EnsembleModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            selected_features: Optional[List[str]] = None
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Train model and return model and metrics."""
        start_time = time.time()

        # Instead of performing feature selection, always use all available features.
        if selected_features is None:
            self.logger.info("Skipping feature selection; using all available features.")
            selected_features = X_train.columns.tolist()
        else:
            self.logger.info("Using provided feature list.")

        # Filter training (and validation) data to use all these features.
        original_columns = X_train.columns.tolist()
        X_train = X_train[selected_features]
        if X_val is not None:
            X_val = X_val[selected_features]
        self.logger.debug("Filtered training features. Original columns: %s", original_columns)
        self.logger.debug("Using columns for training: %s", selected_features)
        self.logger.debug("Training data shape: %s", X_train.shape)

        # Create and train model
        model = self.create_model()
        self.logger.info("Training model with %d samples and %d features", X_train.shape[0], X_train.shape[1])
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate model on training data and (if available) on validation data.
        train_metrics = self.evaluate_model(model, X_train, y_train, "train")
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")

        metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "training_time": time.time() - start_time,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            "feature_importance": model.get_feature_importance(),
            "selected_features": selected_features
        }
        self.logger.info("Completed training in %.2f seconds using %d features", metrics["training_time"],
                         metrics["n_features"])
        return model, metrics

    def train_with_cross_validation(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 5
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Train model with time series cross-validation."""
        # If cross-validation is disabled, simply split the data into training and validation.
        if not self.use_cross_validation:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            return self.train_model(X_train, y_train, X_val, y_val)

        # For cross-validation, we now skip feature selection on the full dataset.
        self.logger.info("Skipping feature selection on full dataset; using all features.")
        selected_features = X.columns.tolist()
        # Ensure X is filtered (even though it should already contain all columns)
        X = X[selected_features]
        self.logger.debug("Features used for cross-validation: %s", selected_features)

        fold_metrics = []
        final_model = None
        final_train_idx = None
        final_val_idx = None

        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=n_splits)

        for i, (train_idx, val_idx) in enumerate(cv.split(X)):
            self.logger.info("Training fold %d/%d", i + 1, n_splits)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # In each fold, use all features
            model, metrics = self.train_model(X_train, y_train, X_val, y_val, selected_features)
            fold_metrics.append(metrics)

            # Save the last fold's model to be used as final model
            if i == n_splits - 1:
                final_model = model
                final_train_idx = train_idx
                final_val_idx = val_idx

        avg_metrics = self._average_metrics(fold_metrics)
        avg_metrics["final_model"] = {
            "train_size": len(final_train_idx),
            "val_size": len(final_val_idx),
            "feature_importance": final_model.get_feature_importance(),
            "selected_features": selected_features
        }
        self.logger.info("Completed cross-validation with %d folds", len(fold_metrics))
        return final_model, avg_metrics

    def evaluate_model(
            self,
            model: BaseModel,
            X: pd.DataFrame,
            y: pd.Series,
            dataset: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        y_pred = model.predict(X)
        is_classifier = self.target_type in [
            PredictionTarget.DIRECTION.value,
            PredictionTarget.VOLATILITY.value
        ]
        metrics = {}

        if is_classifier:
            mask = ~np.isnan(y_pred)
            if not all(mask):
                y_pred = y_pred[mask]
                y_true = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
            else:
                y_true = y

            metrics = {
                f"{dataset}_accuracy": accuracy_score(y_true, y_pred),
                f"{dataset}_precision": precision_score(y_true, y_pred, zero_division=0),
                f"{dataset}_recall": recall_score(y_true, y_pred, zero_division=0),
                f"{dataset}_f1": f1_score(y_true, y_pred, zero_division=0)
            }
            y_dist = np.bincount(y_true) / len(y_true)
            for i, val in enumerate(y_dist):
                metrics[f"{dataset}_class_{i}_pct"] = val
        else:
            mask = ~np.isnan(y_pred)
            if not all(mask):
                y_pred = y_pred[mask]
                y_true = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
            else:
                y_true = y

            metrics = {
                f"{dataset}_mae": mean_absolute_error(y_true, y_pred),
                f"{dataset}_mse": mean_squared_error(y_true, y_pred),
                f"{dataset}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                f"{dataset}_r2": r2_score(y_true, y_pred)
            }
        self.logger.debug("Evaluation metrics on %s set: %s", dataset, metrics)
        return metrics

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Calculate average metrics across multiple runs."""
        avg_metrics = {}
        skip_keys = ['feature_importance', 'selected_features']
        all_keys = set()
        for metrics in metrics_list:
            for key in metrics:
                if key not in skip_keys and isinstance(metrics[key], (int, float)):
                    all_keys.add(key)
        for key in all_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        avg_metrics["n_folds"] = len(metrics_list)
        self.logger.debug("Averaged metrics: %s", avg_metrics)
        return avg_metrics


def train_model_pipeline(config: Dict, timeframe: str) -> Tuple[BaseModel, Dict[str, Any]]:
    """Complete pipeline for training a model using all available features."""
    logger = setup_logger(name="TrainModelPipelineLogger")
    logger.info("Starting training pipeline for timeframe: %s", timeframe)

    from data.storage import DataStorage
    from data.processor import DataProcessor
    storage = DataStorage(config_path="config/config.yaml")
    processor = DataProcessor(config_path="config/config.yaml")

    processed_files = storage.find_latest_processed_data()
    if timeframe not in processed_files:
        raise ValueError(f"No processed data found for timeframe {timeframe}")

    data = processor.load_data({timeframe: processed_files[timeframe]})[timeframe]
    logger.info("Loaded data with %d rows for timeframe: %s", len(data), timeframe)

    from config.constants import PredictionTarget
    target = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
    horizon = config.get('model', {}).get('prediction_horizon', 12)
    split_ratio = config.get('data', {}).get('split_ratio', 0.8)

    target_col = f"target_{horizon}"
    if target_col not in data.columns:
        data = processor.create_target_variable(data, target, horizon)
        logger.info("Created target column: %s", target_col)

    train_size = int(len(data) * split_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    logger.info("Data split: %d training samples and %d test samples", train_data.shape[0], test_data.shape[0])

    # Prepare ML features from training & test data
    X_train, y_train = processor.prepare_ml_features(train_data, horizon)
    X_test, y_test = processor.prepare_ml_features(test_data, horizon)
    logger.debug("Initial training features: %s", X_train.columns.tolist())
    logger.debug("Initial test features: %s", X_test.columns.tolist())

    # Log common and extra features between training and test data
    common_features = set(X_train.columns) & set(X_test.columns)
    extra_in_test = set(X_test.columns) - set(X_train.columns)
    logger.debug("Common features: %s", list(common_features))
    if extra_in_test:
        logger.warning("Extra features in test data: %s", list(extra_in_test))

    from models.trainer import ModelTrainer
    trainer = ModelTrainer(config)
    model, metrics = trainer.train_with_cross_validation(X_train, y_train)

    # Use all training features for the test set
    selected_features = X_train.columns.tolist()
    logger.info("Using all training features for test set: %s", selected_features)
    logger.debug("Test features before filtering: %s", X_test.columns.tolist())
    X_test = X_test[selected_features]
    logger.info("Filtered test features shape: %s", X_test.shape)
    logger.debug("Test features after filtering: %s", X_test.columns.tolist())

    # Evaluate model on test set
    test_metrics = trainer.evaluate_model(model, X_test, y_test, "test")
    metrics["test"] = test_metrics
    metrics["selected_features"] = selected_features

    from config.constants import ModelType

    # --- Build absolute paths based on the project root ---
    # Compute the project root relative to this file.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Save the model in an absolute "models" folder.
    model_folder = os.path.join(project_root, "models")
    os.makedirs(model_folder, exist_ok=True)
    model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
    model_filename = f"{model_type}_{timeframe}_{target}_{horizon}.joblib"
    model_filepath = os.path.join(model_folder, model_filename)
    model.save(model_filepath)

    # Build an absolute path for the metrics file.
    # Remove any existing extension from the model filename to avoid duplicates.
    base_name = os.path.splitext(model_filename)[0]
    metrics_filename = f"{base_name}_metrics.pkl"
    results_folder = os.path.join(project_root, "data", "results", "models")
    os.makedirs(results_folder, exist_ok=True)
    metrics_filepath = os.path.join(results_folder, metrics_filename)

    storage.save_results(metrics, metrics_filepath, include_timestamp=False)
    logger.info("Model saved to %s", model_filepath)
    logger.info("Metrics saved to %s", metrics_filepath)

    logger.info("Model Performance:")
    logger.info("-----------------")
    for dataset in ["train", "val", "test"]:
        if dataset in metrics:
            if f"{dataset}_accuracy" in metrics[dataset]:
                logger.info("%s Accuracy: %.4f", dataset.capitalize(), metrics[dataset][f"{dataset}_accuracy"])
                logger.info("%s F1 Score: %.4f", dataset.capitalize(), metrics[dataset][f"{dataset}_f1"])
            elif f"{dataset}_rmse" in metrics[dataset]:
                logger.info("%s RMSE: %.4f", dataset.capitalize(), metrics[dataset][f"{dataset}_rmse"])
                logger.info("%s R²: %.4f", dataset.capitalize(), metrics[dataset][f"{dataset}_r2"])

    return model, metrics


