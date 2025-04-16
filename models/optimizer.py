import os
import datetime as dt
from typing import Dict, Tuple, Any, Optional
from collections import OrderedDict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Tuple[
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
            # If validation data is provided, use it directly instead of CV
            if X_val is not None and y_val is not None:
                logger.info("Using provided validation data instead of cross-validation")

                # Initialize best parameters
                best_params = None
                best_score = -float('inf')
                cv_results = {"param_" + k: [] for k in search_space.keys()}
                cv_results["mean_test_score"] = []

                # Generate parameter combinations
                from skopt.utils import create_result
                from skopt.space import Dimension

                # Convert skopt space to list of Dimension objects
                dimensions = []
                for key, value in search_space.items():
                    if isinstance(value, Dimension):
                        dimensions.append(value)
                    else:
                        raise ValueError(f"Invalid search space item: {key}: {value}")

                # Use random search for simplicity
                from skopt.utils import dimensions_aslist
                from numpy.random import RandomState

                random_state = RandomState(self.config.get('model', {}).get('random_seed', 42))
                n_iterations = self.n_iter

                logger.info(f"Performing random search with {n_iterations} iterations")

                for i in range(n_iterations):
                    # Sample random parameters
                    params = {key: dim.rvs(random_state=random_state)[0]
                              for key, dim in zip(search_space.keys(), dimensions)}

                    # Create and train model with these parameters
                    rf_model = RandomForestClassifier(
                        **params,
                        class_weight='balanced',
                        random_state=self.config.get('model', {}).get('random_seed', 42),
                        n_jobs=self.n_jobs
                    )

                    rf_model.fit(X_train, y_train)

                    # Predict on validation set
                    y_pred = rf_model.predict(X_val)
                    score = f1_score(y_val, y_pred)

                    # Track parameters and score
                    for key in search_space.keys():
                        cv_results["param_" + key].append(params[key])
                    cv_results["mean_test_score"].append(score)

                    # Update best if needed
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(
                            f"New best RF params (iter {i + 1}/{n_iterations}): {best_params}, F1: {best_score:.4f}")

                results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'all_results': cv_results
                }
                logger.info(f"RF Best parameters: {best_params}")
                logger.info(f"RF Best score: {best_score}")

                return best_params, results
            else:
                # Use BayesSearchCV with cross-validation if no validation data
                logger.info("No validation data provided, using BayesSearchCV with TimeSeriesSplit")
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

    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

            # If validation data is provided, use it directly instead of CV
            if X_val is not None and y_val is not None:
                logger.info("Using provided validation data instead of cross-validation")

                # Initialize best parameters
                best_params = None
                best_score = -float('inf')
                cv_results = {"param_" + k: [] for k in search_space.keys()}
                cv_results["mean_test_score"] = []

                # Generate parameter combinations
                from skopt.utils import create_result
                from skopt.space import Dimension

                # Convert skopt space to list of Dimension objects
                dimensions = []
                for key, value in search_space.items():
                    if isinstance(value, Dimension):
                        dimensions.append(value)
                    else:
                        raise ValueError(f"Invalid search space item: {key}: {value}")

                # Use random search for simplicity
                from skopt.utils import dimensions_aslist
                from numpy.random import RandomState

                random_state = RandomState(self.config.get('model', {}).get('random_seed', 42))
                n_iterations = self.n_iter

                logger.info(f"Performing random search with {n_iterations} iterations")

                # Calculate class weights if needed for XGBoost
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0

                for i in range(n_iterations):
                    # Sample random parameters
                    params = {key: dim.rvs(random_state=random_state)[0]
                              for key, dim in zip(search_space.keys(), dimensions)}

                    # Create and train model with these parameters
                    xgb_model = xgb.XGBClassifier(
                        **params,
                        scale_pos_weight=scale_pos_weight,
                        random_state=self.config.get('model', {}).get('random_seed', 42),
                        use_label_encoder=False,
                        eval_metric='logloss',
                        n_jobs=self.n_jobs
                    )

                    # Create evaluation set if validation data is provided
                    eval_set = [(X_val, y_val)]

                    xgb_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)

                    # Predict on validation set
                    y_pred = xgb_model.predict(X_val)
                    score = f1_score(y_val, y_pred)

                    # Track parameters and score
                    for key in search_space.keys():
                        cv_results["param_" + key].append(params[key])
                    cv_results["mean_test_score"].append(score)

                    # Update best if needed
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(
                            f"New best XGB params (iter {i + 1}/{n_iterations}): {best_params}, F1: {best_score:.4f}")

                results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'all_results': cv_results
                }
                logger.info(f"XGB Best parameters: {best_params}")
                logger.info(f"XGB Best score: {best_score}")

                return best_params, results
            else:
                # Use BayesSearchCV with cross-validation if no validation data
                logger.info("No validation data provided, using BayesSearchCV with TimeSeriesSplit")
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

    def optimize_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize model hyperparameters using training data and validation data.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Tuple of (best_params, detailed_results)
        """
        if self.model_type == ModelType.RANDOM_FOREST.value:
            best_params, results = self.optimize_random_forest(X_train, y_train, X_val, y_val)
            best_params = self.validate_hyperparameters(best_params, ModelType.RANDOM_FOREST.value)
            return best_params, results

        elif self.model_type == ModelType.XGBOOST.value:
            best_params, results = self.optimize_xgboost(X_train, y_train, X_val, y_val)
            best_params = self.validate_hyperparameters(best_params, ModelType.XGBOOST.value)
            return best_params, results

        elif self.model_type == ModelType.LSTM.value:
            return self.optimize_lstm(X_train, y_train, X_val, y_val)

        elif self.model_type == ModelType.ENSEMBLE.value:
            logger.info("Optimizing ensemble models...")
            rf_params, rf_results = self.optimize_random_forest(X_train, y_train, X_val, y_val)
            xgb_params, xgb_results = self.optimize_xgboost(X_train, y_train, X_val, y_val)

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
        storage = DataStorage()
        processor = DataProcessor()

        # Use pre-split training and validation data
        split_paths = storage.find_latest_split_data()

        if "train" not in split_paths or timeframe not in split_paths["train"]:
            raise ValueError(f"No training data found for timeframe {timeframe}")

        if "validation" not in split_paths or timeframe not in split_paths["validation"]:
            logger.warning(f"No validation data found for {timeframe}. Will use part of training data for validation.")

            # Load training data
            train_data_dict = processor.load_data({timeframe: split_paths["train"][timeframe]})
            train_data = train_data_dict[timeframe]
            logger.info(f"Loaded training data: {len(train_data)} rows")

            # Split training data for validation
            train_ratio = config.get('data', {}).get('train_ratio', 0.8)
            split_idx = int(len(train_data) * train_ratio)
            train_subset = train_data.iloc[:split_idx]
            val_subset = train_data.iloc[split_idx:]

            # Prepare features
            X_train, y_train = processor.prepare_ml_features(train_subset, horizon=1)
            X_val, y_val = processor.prepare_ml_features(val_subset, horizon=1)
        else:
            # Load both training and validation data
            train_data_dict = processor.load_data({timeframe: split_paths["train"][timeframe]})
            val_data_dict = processor.load_data({timeframe: split_paths["validation"][timeframe]})

            train_data = train_data_dict[timeframe]
            val_data = val_data_dict[timeframe]

            logger.info(f"Loaded training data: {len(train_data)} rows")
            logger.info(f"Loaded validation data: {len(val_data)} rows")

            # Prepare features
            X_train, y_train = processor.prepare_ml_features(train_data, horizon=1)
            X_val, y_val = processor.prepare_ml_features(val_data, horizon=1)

        logger.info(f"Prepared training features shape: {X_train.shape}, target shape: {y_train.shape}")
        logger.info(f"Prepared validation features shape: {X_val.shape}, target shape: {y_val.shape}")

        # Set up target variables
        target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        horizon = config.get('model', {}).get('prediction_horizon', 1)
        target_col = f"target_{horizon}"

        print(f"Using target type: {target_type}, horizon: {horizon}")

        if target_col not in train_data.columns:
            print(f"Creating target variable {target_col}")
            train_data = processor.create_target_variable(train_data, target_type, horizon)
            if "validation" in split_paths and timeframe in split_paths["validation"]:
                val_data = processor.create_target_variable(val_data, target_type, horizon)

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
            # Run actual optimization with training and validation data
            print("Starting actual optimization (this may take a while)...")
            best_params, detailed_results = optimizer.optimize_model(X_train, y_train, X_val, y_val)
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