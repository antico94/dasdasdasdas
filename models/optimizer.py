import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from config.constants import ModelType, PredictionTarget
from data.processor import DataProcessor
from data.storage import DataStorage
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.trainer import ModelTrainer


class HyperparameterOptimizer:
    """Hyperparameter optimization for trading models."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
        self.n_splits = config.get('model', {}).get('cv_splits', 3)
        self.n_jobs = config.get('model', {}).get('optimizer_jobs', -1)
        self.n_iter = config.get('model', {}).get('optimizer_iterations', 20)

        # Metric for optimization
        if self.target_type in [PredictionTarget.DIRECTION.value, PredictionTarget.VOLATILITY.value]:
            self.scoring = 'f1'  # For classification
        else:
            self.scoring = 'neg_mean_squared_error'  # For regression

    def optimize_random_forest(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize hyperparameters for Random Forest model."""
        print("Optimizing Random Forest hyperparameters...")

        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }

        # Create base model
        model = RandomForestModel(self.config).model

        # Create optimizer
        optimizer = BayesSearchCV(
            model,
            search_space,
            n_iter=self.n_iter,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        # Run optimization
        optimizer.fit(X_train, y_train)

        # Extract results
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Convert results to proper format
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': optimizer.cv_results_
        }

        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

        return best_params, results

    def optimize_xgboost(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize hyperparameters for XGBoost model."""
        print("Optimizing XGBoost hyperparameters...")

        # Define search space
        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 12),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Real(0, 5),
            'min_child_weight': Integer(1, 10)
        }

        # Create base model
        model = XGBoostModel(self.config).model

        # Create optimizer
        optimizer = BayesSearchCV(
            model,
            search_space,
            n_iter=self.n_iter,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        # Run optimization
        optimizer.fit(X_train, y_train)

        # Extract results
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Convert results to proper format
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': optimizer.cv_results_
        }

        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

        return best_params, results

    def optimize_lstm(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize hyperparameters for LSTM model."""
        print("LSTM hyperparameter optimization is not fully implemented through BayesSearchCV")
        print("Using manual grid search instead...")

        # Define parameters to try
        lstm_units_options = [[64], [128], [64, 32], [128, 64]]
        dropout_options = [0.2, 0.3, 0.4]
        learning_rate_options = [0.001, 0.005, 0.01]

        best_score = float('-inf')
        best_params = {}
        all_results = []

        # Create cross-validation splits
        cv = TimeSeriesSplit(n_splits=self.n_splits)

        # Manual grid search
        for lstm_units in lstm_units_options:
            for dropout_rate in dropout_options:
                for learning_rate in learning_rate_options:
                    params = {
                        'lstm_units': lstm_units,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate
                    }

                    # Evaluate with cross-validation
                    scores = []
                    for train_idx, val_idx in cv.split(X_train):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                        # Update config with current params
                        config_copy = self.config.copy()
                        if 'model' not in config_copy:
                            config_copy['model'] = {}
                        if 'hyperparameters' not in config_copy['model']:
                            config_copy['model']['hyperparameters'] = {}

                        config_copy['model']['hyperparameters']['lstm'] = params

                        # Create and train model
                        trainer = ModelTrainer(config_copy)
                        model, _ = trainer.train_model(X_tr, y_tr, X_val, y_val)

                        # Evaluate model
                        if self.target_type in [PredictionTarget.DIRECTION.value, PredictionTarget.VOLATILITY.value]:
                            from sklearn.metrics import f1_score
                            y_pred = model.predict(X_val)
                            score = f1_score(y_val, y_pred)
                        else:
                            from sklearn.metrics import mean_squared_error
                            y_pred = model.predict(X_val)
                            score = -mean_squared_error(y_val, y_pred)

                        scores.append(score)

                    # Average score across folds
                    avg_score = np.mean(scores)

                    # Record result
                    result = {
                        'params': params,
                        'score': avg_score,
                        'std': np.std(scores)
                    }
                    all_results.append(result)

                    # Update best if improved
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params
                        print(f"New best: {params} with score {avg_score}")

        # Convert results to proper format
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }

        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

        return best_params, results

    def optimize_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize hyperparameters for the selected model type."""
        if self.model_type == ModelType.RANDOM_FOREST.value:
            return self.optimize_random_forest(X_train, y_train)
        elif self.model_type == ModelType.XGBOOST.value:
            return self.optimize_xgboost(X_train, y_train)
        elif self.model_type == ModelType.LSTM.value:
            return self.optimize_lstm(X_train, y_train)
        elif self.model_type == ModelType.ENSEMBLE.value:
            # Optimize each model type separately
            print("Optimizing ensemble models...")

            rf_params, rf_results = self.optimize_random_forest(X_train, y_train)
            xgb_params, xgb_results = self.optimize_xgboost(X_train, y_train)

            # Combine results
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

    def update_config_with_best_params(
            self,
            best_params: Dict[str, Any]
    ) -> Dict:
        """Update configuration with best hyperparameters."""
        config_copy = self.config.copy()

        if 'model' not in config_copy:
            config_copy['model'] = {}

        if 'hyperparameters' not in config_copy['model']:
            config_copy['model']['hyperparameters'] = {}

        if self.model_type == ModelType.ENSEMBLE.value:
            # Update parameters for each model in ensemble
            config_copy['model']['hyperparameters']['random_forest'] = best_params['random_forest']
            config_copy['model']['hyperparameters']['xgboost'] = best_params['xgboost']
        else:
            # Update parameters for single model
            config_copy['model']['hyperparameters'][self.model_type] = best_params

        return config_copy


def optimize_hyperparameters(
        config: Dict,
        timeframe: str
) -> Dict:
    """Run hyperparameter optimization and return updated config."""
    # Load data
    storage = DataStorage(config_path="config/config.yaml")
    processor = DataProcessor(config_path="config/config.yaml")

    # Find processed data
    processed_files = storage.find_latest_processed_data()
    if timeframe not in processed_files:
        raise ValueError(f"No processed data found for timeframe {timeframe}")

    # Load data
    data = processor.load_data({timeframe: processed_files[timeframe]})[timeframe]

    # Prepare features and target
    target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
    horizon = config.get('model', {}).get('prediction_horizon', 12)

    # Ensure target column exists
    target_col = f"target_{horizon}"
    if target_col not in data.columns:
        data = processor.create_target_variable(data, target_type, horizon)

    # Use only training data for optimization (first 80%)
    split_ratio = config.get('data', {}).get('split_ratio', 0.8)
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx]

    # Get features and target
    X_train, y_train = processor.prepare_ml_features(train_data, horizon)

    # Create optimizer
    optimizer = HyperparameterOptimizer(config)

    # Run optimization
    best_params, results = optimizer.optimize_model(X_train, y_train)

    # Update config with best parameters
    updated_config = optimizer.update_config_with_best_params(best_params)

    # Save optimization results
    model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
    results_filepath = os.path.join(
        "models",
        f"{model_type}_{timeframe}_{target_type}_{horizon}_optimization.pkl"
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)

    # Save results
    storage.save_results(results, results_filepath)

    print(f"Optimization results saved to {results_filepath}")
    print(f"Best parameters: {best_params}")

    return updated_config