import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

from config.constants import ModelType, PredictionTarget
from data.processor import DataProcessor
from data.storage import DataStorage
from models.feature_selection import FeatureSelector
from utils.logger import setup_logger

logger = setup_logger("ModelTrainer")


def check_data_split_integrity(train_indices, val_indices=None, test_indices=None):
    """
    Check if there's any overlap between train, validation, and test indices.

    Args:
        train_indices: Index of training data
        val_indices: Index of validation data
        test_indices: Index of test data

    Returns:
        True if no overlap, False if overlap detected
    """
    if val_indices is not None:
        train_val_overlap = set(train_indices).intersection(set(val_indices))
        if train_val_overlap:
            logger.error(
                f"DATA LEAKAGE DETECTED! {len(train_val_overlap)} overlapping indices between train and validation")
            return False

    if test_indices is not None:
        train_test_overlap = set(train_indices).intersection(set(test_indices))
        if train_test_overlap:
            logger.error(f"DATA LEAKAGE DETECTED! {len(train_test_overlap)} overlapping indices between train and test")
            return False

        if val_indices is not None:
            val_test_overlap = set(val_indices).intersection(set(test_indices))
            if val_test_overlap:
                logger.error(
                    f"DATA LEAKAGE DETECTED! {len(val_test_overlap)} overlapping indices between validation and test")
                return False

    logger.info("Data split integrity check passed - no overlap between train, validation, and test sets")
    return True


def load_split_datasets(timeframe: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load train, validation, and test datasets for a specific timeframe.

    Args:
        timeframe: The timeframe to load (e.g., "H1")

    Returns:
        Dictionary with 'train', 'validation', and 'test' keys, each containing (X, y) tuples
    """
    storage = DataStorage()
    processor = DataProcessor()
    split_paths = storage.find_latest_split_data()

    result = {}

    # Load training data (required)
    if "train" not in split_paths or timeframe not in split_paths["train"]:
        raise ValueError(f"No training data found for {timeframe}. Run data processing first.")

    train_data = processor.load_data({timeframe: split_paths["train"][timeframe]})
    train_df = train_data[timeframe]
    X_train, y_train = processor.prepare_ml_features(train_df, horizon=1)
    result['train'] = (X_train, y_train, train_df)
    logger.info(f"Loaded training data: {len(train_df)} rows, {X_train.shape[1]} features")

    # Load validation data (optional)
    if "validation" in split_paths and timeframe in split_paths["validation"]:
        val_data = processor.load_data({timeframe: split_paths["validation"][timeframe]})
        val_df = val_data[timeframe]
        X_val, y_val = processor.prepare_ml_features(val_df, horizon=1)
        result['validation'] = (X_val, y_val, val_df)
        logger.info(f"Loaded validation data: {len(val_df)} rows, {X_val.shape[1]} features")

    # Load test data (optional)
    if "test" in split_paths and timeframe in split_paths["test"]:
        test_data = processor.load_data({timeframe: split_paths["test"][timeframe]})
        test_df = test_data[timeframe]
        X_test, y_test = processor.prepare_ml_features(test_df, horizon=1)
        result['test'] = (X_test, y_test, test_df)
        logger.info(f"Loaded test data: {len(test_df)} rows, {X_test.shape[1]} features")

    return result


def apply_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                            validation_test_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                            method: str = "mutual_info") -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Apply feature selection to training data and transform validation/test data.

    Args:
        X_train: Training features
        y_train: Training target
        validation_test_data: Dictionary with validation/test data
        method: Feature selection method

    Returns:
        Dictionary of transformed datasets and list of selected features
    """
    feature_selector = FeatureSelector()

    try:
        # Convert validation_test_data to format expected by select_features_for_splits
        for_selection = {}
        for split_name, (X, y, _) in validation_test_data.items():
            for_selection[split_name] = (X, y)

        # Apply feature selection
        X_train_selected, transformed_sets, selected_features = feature_selector.select_features_for_splits(
            X_train, y_train, for_selection, method=method
        )

        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")

        # Create result dictionary
        result = {'train': X_train_selected}
        for split_name, X_selected in transformed_sets.items():
            result[split_name] = X_selected

        return result, selected_features

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        logger.info("Using all features")

        # If feature selection fails, return original features
        result = {'train': X_train}
        for split_name, (X, _, _) in validation_test_data.items():
            result[split_name] = X

        return result, X_train.columns.tolist()


def evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
    """
    Evaluate model performance on training and test data.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Test metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    test_metrics = {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'classification_report': class_report
    }
    metrics['test'] = test_metrics

    # Training metrics
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0)
    }
    metrics['train'] = train_metrics

    # Check for potential overfitting
    if train_metrics['train_f1'] - test_metrics['test_f1'] > 0.2:
        logger.warning("Possible overfitting detected: large gap between train and test F1 scores")

    return metrics


def load_optimized_parameters(model_type: str, config: Dict) -> Dict:
    """
    Load optimized parameters from saved file.

    Args:
        model_type: Type of model
        config: Configuration dictionary

    Returns:
        Dictionary of optimized parameters or empty dict if not found
    """
    optimized_file = config["model"].get("optimized_params_file", f"{model_type}_H1_direction_1_optimization.pkl")
    optimized_path = os.path.join("data_output", "trained_models", optimized_file)

    logger.info(f"Looking for optimization file at: {optimized_path}")

    if os.path.exists(optimized_path):
        logger.info(f"Optimized parameters file found at {optimized_path}")
        try:
            opt_results = joblib.load(optimized_path)
            best_params = opt_results.get("best_params", {})
            logger.info(f"Loaded optimized parameters: {best_params}")
            return best_params
        except Exception as e:
            logger.error(f"Error loading optimized parameters: {str(e)}")
    else:
        logger.info(f"No optimized parameters file found at {optimized_path}")

    return {}


class RandomForestTrainer:
    """Class for training and tuning Random Forest models."""

    @staticmethod
    def train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameter_tuning: bool = True, use_optimized_params: bool = False,
              use_cross_validation: bool = False, config: Dict = None, random_seed: int = 42) -> Tuple[Any, Dict]:
        """
        Train a Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            use_optimized_params: Whether to use pre-optimized parameters
            use_cross_validation: Whether to use cross-validation
            config: Configuration dictionary
            random_seed: Random seed for reproducibility

        Returns:
            Trained model and training metrics
        """
        metrics = {}

        # Setup time series cross-validation if needed
        tscv = None
        if use_cross_validation:
            logger.info("Setting up time series cross-validation for training data only")
            tscv = TimeSeriesSplit(n_splits=5)

        # Try to load optimized parameters if requested
        if hyperparameter_tuning and use_optimized_params:
            best_params = load_optimized_parameters(ModelType.RANDOM_FOREST.value, config)

            if best_params:
                # Train with optimized parameters
                model = RandomForestClassifier(
                    **best_params,
                    class_weight='balanced',
                    random_state=random_seed,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                return model, {}

        # If no optimized parameters or hyperparameter tuning not requested, use default approach
        if hyperparameter_tuning:
            # Default parameters
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=15,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=random_seed,
                n_jobs=-1
            )

            # Parameter grid for tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 8],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [10, 15]
            }

            if use_cross_validation:
                # Use cross-validation for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                metrics['cv_scores'] = grid_search.cv_results_
            else:
                # Manual hyperparameter search using validation set
                best_score = 0
                best_params = None

                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        for split in param_grid['min_samples_split']:
                            for leaf in param_grid['min_samples_leaf']:
                                # Train model with current parameters
                                temp_model = RandomForestClassifier(
                                    n_estimators=n_est,
                                    max_depth=depth,
                                    min_samples_split=split,
                                    min_samples_leaf=leaf,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    class_weight='balanced',
                                    random_state=random_seed,
                                    n_jobs=-1
                                )
                                temp_model.fit(X_train, y_train)

                                # Evaluate on validation set
                                y_pred = temp_model.predict(X_val)
                                score = f1_score(y_val, y_pred)

                                # Update best if improved
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'n_estimators': n_est,
                                        'max_depth': depth,
                                        'min_samples_split': split,
                                        'min_samples_leaf': leaf,
                                        'max_features': 'sqrt',
                                        'bootstrap': True
                                    }

                logger.info(f"Best parameters: {best_params}")

                # Train final model with best parameters
                model = RandomForestClassifier(
                    **best_params,
                    class_weight='balanced',
                    random_state=random_seed,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
        else:
            # No hyperparameter tuning - use default parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=15,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=random_seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        return model, metrics


class XGBoostTrainer:
    """Class for training and tuning XGBoost models."""

    @staticmethod
    def train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameter_tuning: bool = True, use_optimized_params: bool = False,
              use_cross_validation: bool = False, config: Dict = None, random_seed: int = 42) -> Tuple[Any, Dict]:
        """
        Train an XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            use_optimized_params: Whether to use pre-optimized parameters
            use_cross_validation: Whether to use cross-validation
            config: Configuration dictionary
            random_seed: Random seed for reproducibility

        Returns:
            Trained model and training metrics
        """
        metrics = {}

        # Setup time series cross-validation if needed
        tscv = None
        if use_cross_validation:
            logger.info("Setting up time series cross-validation for training data only")
            tscv = TimeSeriesSplit(n_splits=5)

        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
        logger.info(f"Class weight (0:1 ratio): {scale_pos_weight:.2f}")

        # Try to load optimized parameters if requested
        if hyperparameter_tuning and use_optimized_params:
            best_params = load_optimized_parameters(ModelType.XGBOOST.value, config)

            if best_params:
                # Ensure random_state is set
                if "random_state" not in best_params:
                    best_params["random_state"] = random_seed

                # Train with optimized parameters
                model = xgb.XGBClassifier(**best_params)
                model.fit(X_train, y_train)
                return model, {}

        # If no optimized parameters or hyperparameter tuning not requested, use default approach
        if hyperparameter_tuning:
            # Default parameters
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                gamma=1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_seed,
                n_jobs=-1
            )

            # Parameter grid for tuning
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1],
                'min_child_weight': [5, 10]
            }

            if use_cross_validation:
                # Use cross-validation for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                metrics['cv_scores'] = grid_search.cv_results_
            else:
                # Manual hyperparameter search using validation set
                best_score = 0
                best_params = None

                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        for lr in param_grid['learning_rate']:
                            for weight in param_grid['min_child_weight']:
                                # Train model with current parameters
                                temp_model = xgb.XGBClassifier(
                                    n_estimators=n_est,
                                    max_depth=depth,
                                    learning_rate=lr,
                                    min_child_weight=weight,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    gamma=1,
                                    scale_pos_weight=scale_pos_weight,
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    random_state=random_seed,
                                    n_jobs=-1
                                )
                                temp_model.fit(X_train, y_train)

                                # Evaluate on validation set
                                y_pred = temp_model.predict(X_val)
                                score = f1_score(y_val, y_pred)

                                # Update best if improved
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'n_estimators': n_est,
                                        'max_depth': depth,
                                        'learning_rate': lr,
                                        'min_child_weight': weight,
                                        'subsample': 0.8,
                                        'colsample_bytree': 0.8,
                                        'gamma': 1
                                    }

                logger.info(f"Best parameters: {best_params}")

                # Add additional parameters
                best_params['scale_pos_weight'] = scale_pos_weight
                best_params['use_label_encoder'] = False
                best_params['eval_metric'] = 'logloss'
                best_params['random_state'] = random_seed
                best_params['n_jobs'] = -1

                # Train final model with best parameters
                model = xgb.XGBClassifier(**best_params)
                model.fit(X_train, y_train)
        else:
            # No hyperparameter tuning - use default parameters
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                gamma=1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        return model, metrics


class EnsembleTrainer:
    """Class for training and tuning ensemble models."""

    @staticmethod
    def train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameter_tuning: bool = True, use_optimized_params: bool = False,
              use_cross_validation: bool = False, config: Dict = None, random_seed: int = 42) -> Tuple[Any, Dict]:
        """
        Train an ensemble model combining RandomForest and XGBoost.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            use_optimized_params: Whether to use pre-optimized parameters
            use_cross_validation: Whether to use cross-validation
            config: Configuration dictionary
            random_seed: Random seed for reproducibility

        Returns:
            Trained model and training metrics
        """
        metrics = {}
        storage = DataStorage()

        # Try to load optimized parameters if requested
        if hyperparameter_tuning and use_optimized_params:
            best_params = load_optimized_parameters(ModelType.ENSEMBLE.value, config)

            if best_params:
                rf_best_params = best_params.get("random_forest", {})
                xgb_best_params = best_params.get("xgboost", {})

                if rf_best_params and xgb_best_params:
                    logger.info(
                        f"Using optimized parameters for ensemble: RF: {rf_best_params}, XGB: {xgb_best_params}")

                    # Train RandomForest with optimized parameters
                    rf_model = RandomForestClassifier(
                        **rf_best_params,
                        class_weight='balanced',
                        random_state=random_seed,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)

                    # Train XGBoost with optimized parameters
                    if "random_state" not in xgb_best_params:
                        xgb_best_params["random_state"] = random_seed

                    xgb_model = xgb.XGBClassifier(**xgb_best_params)
                    xgb_model.fit(X_train, y_train)

                    # Create and train ensemble
                    from sklearn.ensemble import VotingClassifier
                    model = VotingClassifier(
                        estimators=[('rf', rf_model), ('xgb', xgb_model)],
                        voting='soft',
                        weights=[0.3, 0.7]
                    )
                    model.fit(X_train, y_train)

                    # Save individual models for inspection
                    timeframe = "H1"  # Default to H1 for gold
                    prediction_target = "direction"
                    prediction_horizon = 1

                    rf_path = storage.save_model(
                        rf_model,
                        f"ensemble_{timeframe}_{prediction_target}_{prediction_horizon}_random_forest"
                    )
                    xgb_path = storage.save_model(
                        xgb_model,
                        f"ensemble_{timeframe}_{prediction_target}_{prediction_horizon}_xgboost"
                    )

                    logger.info(f"Saved RF model to {rf_path}")
                    logger.info(f"Saved XGB model to {xgb_path}")

                    return model, metrics

        # Train individual models
        logger.info("Training ensemble components separately")

        # Train RandomForest
        rf_model, _ = RandomForestTrainer.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning,
            use_optimized_params=use_optimized_params,
            use_cross_validation=use_cross_validation,
            config=config,
            random_seed=random_seed
        )

        # Train XGBoost
        xgb_model, _ = XGBoostTrainer.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning,
            use_optimized_params=use_optimized_params,
            use_cross_validation=use_cross_validation,
            config=config,
            random_seed=random_seed
        )

        # Create and train ensemble
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft',
            weights=[0.3, 0.7]
        )
        model.fit(X_train, y_train)

        return model, metrics


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                config: Dict, model_type: str = "ensemble", random_seed: int = 42) -> Tuple[Any, Dict]:
    """
    Train a model with the specified configuration.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        config: Configuration dictionary
        model_type: Type of model to train
        random_seed: Random seed for reproducibility

    Returns:
        Trained model and training metrics
    """
    hyperparameter_tuning = config.get("model", {}).get("hyperparameter_tuning", True)
    use_optimized_params = hyperparameter_tuning and config.get("model", {}).get("use_bayes_optimizer", False)
    use_cross_validation = config.get("model", {}).get("cross_validation", False)

    logger.info(f"Training {model_type} model with settings:")
    logger.info(f"  hyperparameter_tuning: {hyperparameter_tuning}")
    logger.info(f"  use_optimized_params: {use_optimized_params}")
    logger.info(f"  use_cross_validation: {use_cross_validation}")
    logger.info(f"  random_seed: {random_seed}")

    if model_type == ModelType.RANDOM_FOREST.value:
        model, train_metrics = RandomForestTrainer.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning,
            use_optimized_params=use_optimized_params,
            use_cross_validation=use_cross_validation,
            config=config,
            random_seed=random_seed
        )
    elif model_type == ModelType.XGBOOST.value:
        model, train_metrics = XGBoostTrainer.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning,
            use_optimized_params=use_optimized_params,
            use_cross_validation=use_cross_validation,
            config=config,
            random_seed=random_seed
        )
    elif model_type == ModelType.ENSEMBLE.value:
        model, train_metrics = EnsembleTrainer.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning,
            use_optimized_params=use_optimized_params,
            use_cross_validation=use_cross_validation,
            config=config,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

    # Merge with training metrics
    if train_metrics:
        metrics.update(train_metrics)

    return model, metrics


def train_best_model(config: Dict, timeframe: str = "H1", num_runs: int = 5, random_seed_base: int = 42) -> Tuple[
    Any, Dict]:
    """
    Train multiple models with different random seeds and select the best one.

    Args:
        config: Configuration dictionary
        timeframe: Timeframe to use (e.g., "H1")
        num_runs: Number of training runs with different seeds
        random_seed_base: Base random seed

    Returns:
        Best model and its metrics
    """
    logger.info(f"Starting optimized model training with {num_runs} iterations")

    best_model = None
    best_metrics = None
    best_f1 = 0.0
    best_accuracy = 0.0
    best_run = 0

    # Load datasets
    datasets = load_split_datasets(timeframe)

    if 'train' not in datasets:
        raise ValueError(f"No training data found for {timeframe}")

    X_train, y_train, _ = datasets['train']

    # Use validation data if available, otherwise create from training data
    # Use validation data if available, otherwise create from training data
    if 'validation' in datasets:
        X_val, y_val, _ = datasets['validation']
    else:
        logger.warning("No validation data available, splitting training data")
        train_size = int(len(X_train) * 0.8)
        X_val = X_train.iloc[train_size:].copy()
        y_val = y_train.iloc[train_size:].copy()
        X_train = X_train.iloc[:train_size].copy()
        y_train = y_train.iloc[:train_size].copy()

    # Check for data leakage
    data_integrity = check_data_split_integrity(X_train.index, X_val.index)
    if not data_integrity:
        logger.warning("DATA INTEGRITY CHECK FAILED! There may be overlap between train/val sets")

    # Feature selection
    feature_data = {'train': (X_train, y_train, None)}
    if 'validation' in datasets:
        feature_data['validation'] = datasets['validation']
    if 'test' in datasets:
        feature_data['test'] = datasets['test']

    if config["model"].get("feature_selection", True):
        transformed_datasets, selected_features = apply_feature_selection(X_train, y_train, feature_data)
        X_train = transformed_datasets['train']
        X_val = transformed_datasets.get('validation', X_val)
    else:
        selected_features = X_train.columns.tolist()

    # Run training multiple times with different random seeds
    for run in range(num_runs):
        logger.info(f"Starting training run {run + 1}/{num_runs}")

        # Set random seed for this run
        run_seed = random_seed_base + run

        try:
            # Determine model type to use
            model_type = config.get("model", {}).get("type", "ensemble")

            # Train model
            model, metrics = train_model(
                X_train.copy(), y_train.copy(),
                X_val.copy(), y_val.copy(),
                config, model_type, run_seed
            )

            # Extract metrics
            if 'test' in metrics:
                f1 = metrics['test']['test_f1']
                accuracy = metrics['test']['test_accuracy']
                logger.info(f"Run {run + 1} results - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

                # Update best model if this one is better
                if f1 > best_f1:
                    best_model = model
                    best_metrics = metrics
                    best_f1 = f1
                    best_accuracy = accuracy
                    best_run = run + 1
                    logger.info(f"New best model found in run {best_run} with F1 score: {best_f1:.4f}")
            else:
                logger.warning(f"No test metrics found for run {run + 1}")

        except Exception as e:
            logger.error(f"Error in training run {run + 1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    if best_model is None:
        raise ValueError("All training runs failed. Check logs for details.")

    logger.info(f"Best model selected from run {best_run}/{num_runs}")
    logger.info(f"Best model metrics - F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    # Save the best model
    prediction_target = config.get("model", {}).get("prediction_target", "direction")
    prediction_horizon = config.get("model", {}).get("prediction_horizon", 1)
    model_type = config.get("model", {}).get("type", "ensemble")
    model_name = f"{model_type}_{timeframe}_{prediction_target}_{prediction_horizon}_best"

    metadata = {
        "features": selected_features,
        "metrics": best_metrics,
        "config": config,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_shape": X_train.shape,
        "timeframe": timeframe,
        "prediction_target": prediction_target,
        "prediction_horizon": prediction_horizon,
        "best_run": best_run,
        "total_runs": num_runs
    }

    storage = DataStorage()
    model_path = storage.save_model(best_model, model_name, metadata=metadata)
    logger.info(f"Best model saved to {model_path}")

    return best_model, best_metrics


def train_model_pipeline(config: Dict, timeframe: str = "H1") -> Tuple[Any, Dict]:
    """
    Complete pipeline for training an optimized model for gold trading.

    Args:
        config: Configuration dictionary
        timeframe: Timeframe to use (e.g., "H1")

    Returns:
        Trained model and its metrics
    """
    logger.info(f"Starting model training pipeline for {timeframe}")
    start_time = time.time()

    # Load datasets
    datasets = load_split_datasets(timeframe)

    if 'train' not in datasets:
        raise ValueError(f"No training data found for {timeframe}")

    X_train, y_train, _ = datasets['train']
    logger.info(f"Loaded training data: {X_train.shape}")

    # Use validation data if available, otherwise create from training data
    if 'validation' in datasets:
        X_val, y_val, _ = datasets['validation']
        logger.info(f"Loaded validation data: {X_val.shape}")
    else:
        logger.warning("No validation data available, splitting training data")
        train_size = int(len(X_train) * 0.8)
        X_val = X_train.iloc[train_size:].copy()
        y_val = y_train.iloc[train_size:].copy()
        X_train = X_train.iloc[:train_size].copy()
        y_train = y_train.iloc[:train_size].copy()
        logger.info(f"Split training data: {X_train.shape}, validation: {X_val.shape}")

    # Load test data if available
    X_test, y_test = None, None
    if 'test' in datasets:
        X_test, y_test, _ = datasets['test']
        logger.info(f"Loaded test data: {X_test.shape}")

    # Check for data leakage
    data_integrity = check_data_split_integrity(
        X_train.index, X_val.index, X_test.index if X_test is not None else None
    )
    if not data_integrity:
        logger.warning("DATA INTEGRITY CHECK FAILED! There may be overlap between datasets")

    try:
        # Update prediction horizon to 1 if needed
        if 'model' in config and 'prediction_horizon' in config['model']:
            prediction_horizon = config['model']['prediction_horizon']
            if prediction_horizon != 1:
                logger.info(f"Changing prediction horizon from {prediction_horizon} to 1 for better performance")
                config['model']['prediction_horizon'] = 1
        else:
            if 'model' not in config:
                config['model'] = {}
            config['model']['prediction_horizon'] = 1
            logger.info("Setting prediction horizon to 1 period")

        # Log class distribution
        train_class_balance = y_train.mean()
        logger.info(f"Train class balance: {train_class_balance:.4f}")

        val_class_balance = y_val.mean()
        logger.info(f"Validation class balance: {val_class_balance:.4f}")

        if y_test is not None:
            test_class_balance = y_test.mean()
            logger.info(f"Test class balance: {test_class_balance:.4f}")

        # Feature selection
        feature_data = {'train': (X_train, y_train, None), 'validation': (X_val, y_val, None)}
        if X_test is not None:
            feature_data['test'] = (X_test, y_test, None)

        if config.get("model", {}).get("feature_selection", True):
            transformed_datasets, selected_features = apply_feature_selection(X_train, y_train, feature_data)
            X_train = transformed_datasets['train']
            X_val = transformed_datasets['validation']
            if 'test' in transformed_datasets:
                X_test = transformed_datasets['test']
        else:
            selected_features = X_train.columns.tolist()
            logger.info(f"Using all {len(selected_features)} features")

        # Determine model type
        model_type = config.get("model", {}).get("type", "ensemble")

        # Train model
        model, metrics = train_model(
            X_train, y_train, X_val, y_val,
            config, model_type
        )

        # Final evaluation on test set if available
        if X_test is not None and y_test is not None:
            test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            metrics['final_test'] = test_metrics.get('test', {})
            logger.info(
                f"Final test metrics: "
                f"Accuracy={metrics['final_test'].get('test_accuracy', 0):.4f}, "
                f"F1={metrics['final_test'].get('test_f1', 0):.4f}"
            )

        # Save model
        prediction_target = config.get("model", {}).get("prediction_target", "direction")
        prediction_horizon = config.get("model", {}).get("prediction_horizon", 1)
        model_name = f"{model_type}_{timeframe}_{prediction_target}_{prediction_horizon}"

        metadata = {
            "features": selected_features,
            "metrics": metrics,
            "config": config,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_shape": X_train.shape,
            "timeframe": timeframe,
            "prediction_target": prediction_target,
            "prediction_horizon": prediction_horizon
        }

        storage = DataStorage()
        model_path = storage.save_model(model, model_name, metadata=metadata)
        logger.info(f"Model saved to {model_path}")

        # Calculate training time
        training_time = time.time() - start_time
        metrics["training_time"] = training_time
        metrics["n_features"] = len(selected_features)

        logger.info(f"Model training completed in {training_time:.2f} seconds")

        return model, metrics

    except Exception as e:
        logger.error(f"Error in train_model_pipeline: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise