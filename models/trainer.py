import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from config.constants import ModelType, PredictionTarget
from data.processor import DataProcessor
from data.storage import DataStorage
from models.feature_selection import FeatureSelector
from utils.logger import setup_logger

logger = setup_logger("ModelTrainer")


def train_model_pipeline(config: Dict, timeframe: str = "H1") -> Tuple[Any, Dict]:
    """Complete pipeline for training an optimized model for gold trading."""
    logger.info(f"Starting model training pipeline for {timeframe}")
    start_time = time.time()

    # Load processed data
    storage = DataStorage()
    processed_files = storage.find_latest_processed_data()

    if timeframe not in processed_files:
        raise ValueError(f"No processed data found for {timeframe}. Run data processing first.")

    processor = DataProcessor()
    data_dict = processor.load_data({timeframe: processed_files[timeframe]})
    df = data_dict[timeframe]

    # Check for and update prediction horizon - use 1 period
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

    # Prepare features and target with 1-period horizon
    X, y = processor.prepare_ml_features(df, horizon=1)
    logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")

    # Initialize config with default values if keys don't exist
    if 'data' not in config:
        config['data'] = {}
    if 'split_ratio' not in config.get('data', {}):
        config['data']['split_ratio'] = 0.8  # Default 80/20 split
        logger.info("Using default train/test split ratio of 0.8")

    # Split data chronologically
    train_size = int(len(X) * config["data"]["split_ratio"])
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    # Check for class imbalance
    train_class_balance = y_train.mean()
    test_class_balance = y_test.mean()
    logger.info(f"Class balance - Train: {train_class_balance:.4f}, Test: {test_class_balance:.4f}")

    # Ensure model config exists with defaults
    if 'model' not in config:
        config['model'] = {}
        logger.warning("Model configuration not found, using defaults")

    # Feature selection specifically for gold trading
    feature_selector = FeatureSelector()
    if config["model"].get("feature_selection", True):
        try:
            X_train, X_test, selected_features = feature_selector.select_features(
                X_train, y_train, X_test, method="mutual_info"
            )
            logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            logger.info("Using all features")
            selected_features = X_train.columns.tolist()
    else:
        selected_features = X_train.columns.tolist()

    # Determine model type to use
    model_type = config["model"].get("type", "ensemble")

    # Create and train model
    model, metrics = train_gold_trading_model(
        X_train, y_train, X_test, y_test,
        model_type=model_type,
        use_cross_validation=config["model"].get("cross_validation", True),
        hyperparameter_tuning=config["model"].get("hyperparameter_tuning", True)
    )

    # Save model and metadata
    prediction_target = config["model"].get("prediction_target", "direction")
    prediction_horizon = config["model"].get("prediction_horizon", 1)  # Use 1 as default now
    model_name = f"{model_type}_{timeframe}_{prediction_target}_{prediction_horizon}"

    metadata = {
        "features": selected_features,
        "metrics": metrics,
        "config": config,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_shape": X.shape,
        "timeframe": timeframe,
        "prediction_target": prediction_target,
        "prediction_horizon": prediction_horizon
    }

    model_path = storage.save_model(model, model_name, metadata=metadata)
    logger.info(f"Model saved to {model_path}")

    # Calculate training time
    training_time = time.time() - start_time
    metrics["training_time"] = training_time
    metrics["n_features"] = len(selected_features)

    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Test metrics: {metrics['test']}")

    return model, metrics


def train_gold_trading_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: str = "ensemble",
        use_cross_validation: bool = True,
        hyperparameter_tuning: bool = True
) -> Tuple[Any, Dict]:
    """Train an optimized model for gold trading."""
    logger.info(f"Training {model_type} model for gold trading")

    metrics = {}

    # Setup time series cross-validation if needed
    if use_cross_validation:
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

    # Check class distribution and balance
    class_counts = y_train.value_counts()
    logger.info(f"Train class distribution: {class_counts}")

    # Calculate appropriate class weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    logger.info(f"Class weight (0:1 ratio): {scale_pos_weight:.2f}")

    # Train different model types
    if model_type == ModelType.RANDOM_FOREST.value:
        # Gold-optimized Random Forest with parameters suitable for short-term prediction
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,  # Reduced from 10 for less overfitting on short-term
            min_samples_split=20,
            min_samples_leaf=15,  # Increased for better generalization
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        if hyperparameter_tuning:
            from sklearn.model_selection import GridSearchCV
            logger.info("Performing hyperparameter tuning for RandomForest")

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 8],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [10, 15]
            }

            if use_cross_validation:
                grid_search = GridSearchCV(
                    estimator=model,
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
                # Simplified parameter search
                best_score = 0
                best_params = None

                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        model = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=20,
                            min_samples_leaf=15,
                            class_weight='balanced',
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        f1 = f1_score(y_test, y_pred)

                        if f1 > best_score:
                            best_score = f1
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_split': 20,
                                'min_samples_leaf': 15
                            }

                logger.info(f"Best parameters: {best_params}")
                model = RandomForestClassifier(
                    **best_params,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )

        # Final training with best parameters
        model.fit(X_train, y_train)

    elif model_type == ModelType.XGBOOST.value:
        # Gold-optimized XGBoost with parameters for 1-period ahead prediction
        model = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for 1-period prediction
            max_depth=4,  # Reduced for less overfitting
            learning_rate=0.1,  # Faster learning for short-term patterns
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,  # More conservative for financial data
            gamma=1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,  # Avoid warning
            eval_metric='logloss',  # Binary classification metric
            random_state=42,
            n_jobs=-1
        )

        if hyperparameter_tuning:
            from sklearn.model_selection import GridSearchCV
            logger.info("Performing hyperparameter tuning for XGBoost")

            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1],
                'min_child_weight': [5, 10]
            }

            if use_cross_validation:
                grid_search = GridSearchCV(
                    estimator=model,
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
                # Simple parameter search
                best_score = 0
                best_params = None

                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        for lr in param_grid['learning_rate']:
                            model = xgb.XGBClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=0.8,
                                min_child_weight=10,
                                scale_pos_weight=scale_pos_weight,
                                use_label_encoder=False,
                                eval_metric='logloss',
                                random_state=42,
                                n_jobs=-1
                            )
                            # Use simplified fit without extra parameters
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            f1 = f1_score(y_test, y_pred)

                            if f1 > best_score:
                                best_score = f1
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': 0.8,
                                    'min_child_weight': 10
                                }

                logger.info(f"Best parameters: {best_params}")
                best_params['scale_pos_weight'] = scale_pos_weight
                best_params['use_label_encoder'] = False
                best_params['eval_metric'] = 'logloss'
                best_params['random_state'] = 42
                best_params['n_jobs'] = -1

                model = xgb.XGBClassifier(**best_params)

        # Final training with best parameters - using simplified fit
        model.fit(X_train, y_train)

    elif model_type == ModelType.ENSEMBLE.value:
        # Simplified ensemble approach specifically for 1-period gold prediction
        logger.info("Training ensemble model for gold trading")

        # Train Random Forest - simplified parameters for 1-period prediction
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        # Train XGBoost - simplified parameters for 1-period prediction
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_child_weight=10,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        # Use simplified fit without extra parameters
        xgb_model.fit(X_train, y_train)

        # Create ensemble model with weighted voting - more weight to XGBoost
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            voting='soft',  # Use probability-weighted voting
            weights=[0.3, 0.7]  # Give more weight to XGBoost for gold prediction
        )
        model.fit(X_train, y_train)

        # Save individual models for inspection
        storage = DataStorage()
        timeframe = "H1"  # Default to H1 for gold
        prediction_target = "direction"
        prediction_horizon = 1  # Use 1-period horizon

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

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Evaluate on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Get classification report for detailed metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Store metrics
    test_metrics = {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'classification_report': class_report
    }

    metrics['test'] = test_metrics

    # Calculate metrics for training set to check for overfitting
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0)
    }

    metrics['train'] = train_metrics

    # Check for overfitting
    if train_metrics['train_f1'] - test_metrics['test_f1'] > 0.2:
        logger.warning("Possible overfitting detected: large gap between train and test F1 scores")

    return model, metrics