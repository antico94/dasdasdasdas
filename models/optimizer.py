import os
import datetime as dt
from typing import Dict, Tuple, Any, Optional
from collections import OrderedDict

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


def convert_ordered_dicts(params):
    """Convert OrderedDict objects to regular dictionaries (recursively)."""
    if isinstance(params, OrderedDict):
        return dict(params)
    if isinstance(params, dict):
        return {k: convert_ordered_dicts(v) for k, v in params.items()}
    return params


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

    def validate_hyperparameters(self, best_params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Validate and clean optimized hyperparameters to ensure they're valid."""
        if not best_params:
            logger.warning(f"Empty best parameters for {model_type}. Using defaults.")
            if model_type == ModelType.RANDOM_FOREST.value:
                return {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'max_features': 'sqrt'
                }
            elif model_type == ModelType.XGBOOST.value:
                return {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'min_child_weight': 1
                }

        # For ensemble, validate each submodel's parameters
        if model_type == ModelType.ENSEMBLE.value:
            if 'random_forest' in best_params and not best_params['random_forest']:
                best_params['random_forest'] = self.validate_hyperparameters({}, ModelType.RANDOM_FOREST.value)
            if 'xgboost' in best_params and not best_params['xgboost']:
                best_params['xgboost'] = self.validate_hyperparameters({}, ModelType.XGBOOST.value)

        return best_params

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

        # Log search space
        logger.info(f"RF search space: {search_space}")

        try:
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

            # Validate results
            if not best_params:
                raise ValueError("BayesSearchCV returned empty best_params")

            # Convert OrderedDict to dict
            best_params = convert_ordered_dicts(best_params)

            results = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': optimizer.cv_results_
            }
            logger.info(f"RF Best parameters: {best_params}")
            logger.info(f"RF Best score: {best_score}")

            return best_params, results

        except Exception as e:
            logger.error(f"Error in Random Forest optimization: {str(e)}")
            # Return default parameters if optimization fails
            best_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt'
            }
            logger.warning(f"Using default RF parameters due to optimization failure: {best_params}")

            results = {
                'best_params': best_params,
                'best_score': None,
                'all_results': None,
                'error': str(e)
            }

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

        # Log search space
        logger.info(f"XGB search space: {search_space}")

        try:
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

            # Validate results
            if not best_params:
                raise ValueError("BayesSearchCV returned empty best_params")

            # Convert OrderedDict to dict
            best_params = convert_ordered_dicts(best_params)

            results = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': optimizer.cv_results_
            }
            logger.info(f"XGB Best parameters: {best_params}")
            logger.info(f"XGB Best score: {best_score}")

            return best_params, results

        except Exception as e:
            logger.error(f"Error in XGBoost optimization: {str(e)}")
            # Return default parameters if optimization fails
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'min_child_weight': 1
            }
            logger.warning(f"Using default XGB parameters due to optimization failure: {best_params}")

            results = {
                'best_params': best_params,
                'best_score': None,
                'all_results': None,
                'error': str(e)
            }

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
            best_params, results = self.optimize_random_forest(X_train, y_train)
            best_params = self.validate_hyperparameters(best_params, ModelType.RANDOM_FOREST.value)
            return best_params, results

        elif self.model_type == ModelType.XGBOOST.value:
            best_params, results = self.optimize_xgboost(X_train, y_train)
            best_params = self.validate_hyperparameters(best_params, ModelType.XGBOOST.value)
            return best_params, results

        elif self.model_type == ModelType.LSTM.value:
            return self.optimize_lstm(X_train, y_train)

        elif self.model_type == ModelType.ENSEMBLE.value:
            logger.info("Optimizing ensemble models...")
            rf_params, rf_results = self.optimize_random_forest(X_train, y_train)
            xgb_params, xgb_results = self.optimize_xgboost(X_train, y_train)

            # Validate parameters to ensure they're not empty
            rf_params = self.validate_hyperparameters(rf_params, ModelType.RANDOM_FOREST.value)
            xgb_params = self.validate_hyperparameters(xgb_params, ModelType.XGBOOST.value)

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
    """Run hyperparameter optimization for the specified model type and timeframe."""
    try:
        print("Starting hyperparameter optimization")
        storage = DataStorage(config_path="config/config.yaml")
        processor = DataProcessor(config_path="config/config.yaml")

        # Find and load processed data
        processed_files = storage.find_latest_processed_data()
        print(f"Found processed files: {processed_files}")

        if timeframe not in processed_files:
            raise ValueError(f"No processed data found for timeframe {timeframe}")

        data = processor.load_data({timeframe: processed_files[timeframe]})[timeframe]
        logger.info(f"Loaded processed data for {timeframe}: {data.shape} rows")
        print(f"Loaded processed data for {timeframe}: {data.shape} rows")

        # Set up target variables
        target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        horizon = config.get('model', {}).get('prediction_horizon', 1)
        target_col = f"target_{horizon}"

        print(f"Using target type: {target_type}, horizon: {horizon}")

        if target_col not in data.columns:
            print(f"Creating target variable {target_col}")
            data = processor.create_target_variable(data, target_type, horizon)

        # Split data for training
        split_ratio = config.get('data', {}).get('split_ratio', 0.8)
        split_idx = int(len(data) * split_ratio)
        train_data = data.iloc[:split_idx]
        logger.info(f"Using {len(train_data)} rows for optimization ({split_ratio * 100:.0f}% of data)")
        print(f"Using {len(train_data)} rows for optimization ({split_ratio * 100:.0f}% of data)")

        # Prepare features and target
        X_train, y_train = processor.prepare_ml_features(train_data, horizon)
        print(f"Prepared features shape: {X_train.shape}, target shape: {y_train.shape}")

        # Check for class balance in classification tasks
        if target_type in [PredictionTarget.DIRECTION.value, PredictionTarget.VOLATILITY.value]:
            class_counts = y_train.value_counts()
            print(f"Class distribution: {class_counts}")

            # Raise a warning if classes are severely imbalanced
            if len(class_counts) > 1:
                minority_class_pct = class_counts.min() / len(y_train) * 100
                if minority_class_pct < 10:
                    logger.warning(
                        f"Severe class imbalance detected. Minority class is only {minority_class_pct:.1f}% of data.")

        # Run optimization
        model_type = config.get('model', {}).get('type', ModelType.ENSEMBLE.value)
        print(f"Running optimization for model type: {model_type}")

        optimizer = HyperparameterOptimizer(config)

        # Debug mode can be enabled for testing
        debug_mode = False  # Set to True to bypass actual optimization for testing

        if debug_mode:
            print("DEBUG MODE: Setting test parameters directly")
            if model_type == ModelType.ENSEMBLE.value:
                best_params = {
                    'random_forest': {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'min_samples_split': 10,
                        'min_samples_leaf': 4,
                        'max_features': 'sqrt'
                    },
                    'xgboost': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'gamma': 0,
                        'min_child_weight': 1
                    }
                }
                detailed_results = {'debug': True}
            else:
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'max_features': 'sqrt'
                }
                detailed_results = {'debug': True}
        else:
            # Run actual optimization
            print("Starting actual optimization (this may take a while)...")
            best_params, detailed_results = optimizer.optimize_model(X_train, y_train)
            print(f"Optimization completed. Best params: {best_params}")

        # Convert OrderedDict to regular dict (again, to be sure)
        best_params = convert_ordered_dicts(best_params)

        updated_config = optimizer.update_config_with_best_params(best_params)

        # Validate and report results
        if model_type == ModelType.ENSEMBLE.value:
            rf_params_valid = bool(best_params.get('random_forest', {}))
            xgb_params_valid = bool(best_params.get('xgboost', {}))
            print(f"Ensemble parameters validity: RF={rf_params_valid}, XGB={xgb_params_valid}")

            # If empty params, set some defaults for testing
            if not rf_params_valid:
                print("WARNING: Empty RF params, setting defaults")
                best_params['random_forest'] = {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'max_features': 'sqrt'
                }
            if not xgb_params_valid:
                print("WARNING: Empty XGB params, setting defaults")
                best_params['xgboost'] = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'min_child_weight': 1
                }

        # Format results in the expected structure for the training code
        results_to_save = {
            "best_params": best_params,  # Top-level key for training code to access
            "optimization_details": detailed_results,
            "config": {
                "timeframe": timeframe,
                "target_type": target_type,
                "horizon": horizon,
                "model_type": model_type,
                "optimization_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Debug: Print the results structure
        print("\nResults structure to save:")
        print(f"Keys: {list(results_to_save.keys())}")
        print(f"best_params type: {type(results_to_save['best_params'])}")
        print(f"best_params content: {results_to_save['best_params']}")

        # Build a base filename (do not include any directory path)
        base_filename = f"{model_type}_{timeframe}_{target_type}_{horizon}_optimization"

        # Add direct joblib save as a backup
        import joblib
        direct_save_path = os.path.join("data_output", "trained_models", base_filename + ".pkl")
        print(f"\nDirectly saving optimization results to {direct_save_path}")
        joblib.dump(results_to_save, direct_save_path)

        # Save results with storage class too
        saved_results_path = storage.save_results(results_to_save, base_filename, include_timestamp=False)
        print(f"Optimization results saved to {saved_results_path}")
        logger.info(f"Optimization results saved to {saved_results_path}")
        logger.info(f"Best parameters: {best_params}")

        # Try loading the file back to verify
        print("\nVerifying saved file...")
        try:
            loaded_results = joblib.load(direct_save_path)
            print(f"Successfully loaded results from {direct_save_path}")
            print(f"Loaded best_params: {loaded_results.get('best_params', 'NOT FOUND')}")
        except Exception as e:
            print(f"Error loading results file: {str(e)}")

        return updated_config

    except Exception as e:
        import traceback
        print(f"Error during optimization: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"Error during hyperparameter optimization: {str(e)}", exc_info=True)
        # Return original config if optimization fails
        return config