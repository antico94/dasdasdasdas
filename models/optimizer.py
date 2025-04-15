import os
import datetime as dt
from typing import Dict, Tuple, Any

import numpy as np

# Backwards-compatibility patch for deprecated np.int
if not hasattr(np, 'int'):
    np.int = int

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from config.constants import ModelType, PredictionTarget
from data.processor import DataProcessor
from data.storage import DataStorage
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from utils.logger import setup_logger

logger = setup_logger("HyperparameterOptimizer")


class HyperparameterOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
        self.n_splits = config.get('model', {}).get('cv_splits', 3)
        self.n_jobs = config.get('model', {}).get('optimizer_jobs', -1)
        self.n_iter = config.get('model', {}).get('optimizer_iterations', 20)

        if self.target_type in [PredictionTarget.DIRECTION.value, PredictionTarget.VOLATILITY.value]:
            self.scoring = 'f1'
        else:
            self.scoring = 'neg_mean_squared_error'
        logger.info(f"Initialized optimizer for model type '{self.model_type}' with scoring '{self.scoring}'")

    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        logger.info("Optimizing Random Forest hyperparameters...")
        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
        model = RandomForestModel(self.config).model
        optimizer = BayesSearchCV(
            model,
            search_space,
            n_iter=self.n_iter,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        optimizer.fit(X_train, y_train)
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': optimizer.cv_results_
        }
        logger.info(f"RF Best parameters: {best_params}")
        logger.info(f"RF Best score: {best_score}")
        return best_params, results

    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.info("Optimizing XGBoost hyperparameters...")
        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 12),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Real(0, 5),
            'min_child_weight': Integer(1, 10)
        }
        model = XGBoostModel(self.config).model
        optimizer = BayesSearchCV(
            model,
            search_space,
            n_iter=self.n_iter,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        optimizer.fit(X_train, y_train)
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': optimizer.cv_results_
        }
        logger.info(f"XGB Best parameters: {best_params}")
        logger.info(f"XGB Best score: {best_score}")
        return best_params, results

    def optimize_lstm(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.info("LSTM hyperparameter optimization is not implemented with BayesSearchCV.")
        logger.info("Returning default LSTM hyperparameters.")
        best_params = {
            'lstm_units': [64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
        results = {
            'best_params': best_params,
            'best_score': None,
            'all_results': []
        }
        logger.info(f"Default LSTM parameters: {best_params}")
        return best_params, results

    def optimize_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.model_type == ModelType.RANDOM_FOREST.value:
            return self.optimize_random_forest(X_train, y_train)
        elif self.model_type == ModelType.XGBOOST.value:
            return self.optimize_xgboost(X_train, y_train)
        elif self.model_type == ModelType.LSTM.value:
            return self.optimize_lstm(X_train, y_train)
        elif self.model_type == ModelType.ENSEMBLE.value:
            logger.info("Optimizing ensemble models...")
            rf_params, rf_results = self.optimize_random_forest(X_train, y_train)
            xgb_params, xgb_results = self.optimize_xgboost(X_train, y_train)
            best_params = {
                'random_forest': rf_params,
                'xgboost': xgb_params
            }
            results = {
                'random_forest': rf_results,
                'xgboost': xgb_results
            }
            return best_params, results
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def update_config_with_best_params(self, best_params: Dict[str, Any]) -> Dict:
        config_copy = self.config.copy()
        if 'model' not in config_copy:
            config_copy['model'] = {}
        if 'hyperparameters' not in config_copy['model']:
            config_copy['model']['hyperparameters'] = {}
        if self.model_type == ModelType.ENSEMBLE.value:
            config_copy['model']['hyperparameters']['random_forest'] = best_params['random_forest']
            config_copy['model']['hyperparameters']['xgboost'] = best_params['xgboost']
        else:
            config_copy['model']['hyperparameters'][self.model_type] = best_params
        logger.info("Configuration updated with best hyperparameters.")
        return config_copy


def optimize_hyperparameters(config: Dict, timeframe: str) -> Dict:
    storage = DataStorage(config_path="config/config.yaml")
    processor = DataProcessor(config_path="config/config.yaml")
    processed_files = storage.find_latest_processed_data()
    if timeframe not in processed_files:
        raise ValueError(f"No processed data found for timeframe {timeframe}")

    data = processor.load_data({timeframe: processed_files[timeframe]})[timeframe]
    target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
    horizon = config.get('model', {}).get('prediction_horizon', 1)
    target_col = f"target_{horizon}"
    if target_col not in data.columns:
        data = processor.create_target_variable(data, target_type, horizon)
    split_ratio = config.get('data', {}).get('split_ratio', 0.8)
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx]
    X_train, y_train = processor.prepare_ml_features(train_data, horizon)
    optimizer = HyperparameterOptimizer(config)
    best_params, results = optimizer.optimize_model(X_train, y_train)
    updated_config = optimizer.update_config_with_best_params(best_params)

    # Build a base filename (do not include any directory path)
    model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
    base_filename = f"{model_type}_{timeframe}_{target_type}_{horizon}_optimization"
    # Call save_results; DataStorage will build the full path based on its config.
    saved_results_path = storage.save_results(results, base_filename, include_timestamp=False)
    logger.info(f"Optimization results saved to {saved_results_path}")
    logger.info(f"Best parameters: {best_params}")
    print(f"Optimization results saved to {saved_results_path}")
    print(f"Best parameters: {best_params}")
    return updated_config
