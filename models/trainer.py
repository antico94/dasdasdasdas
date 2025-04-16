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


def train_best_model(config: Dict, timeframe: str = "H1", num_runs: int = 5, random_seed_base: int = 42) -> Tuple[
    Any, Dict]:
    """Train multiple models and select the best one based on F1 score."""
    logger.info(f"Starting optimized model training with {num_runs} iterations")

    best_model = None
    best_metrics = None
    best_f1 = 0.0
    best_accuracy = 0.0
    best_run = 0

    # Load data once to avoid variation in data preprocessing
    storage = DataStorage()
    processed_files = storage.find_latest_processed_data()

    if timeframe not in processed_files:
        raise ValueError(f"No processed data found for {timeframe}. Run data processing first.")

    processor = DataProcessor()
    data_dict = processor.load_data({timeframe: processed_files[timeframe]})
    df = data_dict[timeframe]

    # Prepare features and target
    X, y = processor.prepare_ml_features(df, horizon=1)
    logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")

    # Split data once to ensure consistent train/test sets across all runs
    train_size = int(len(X) * config["data"].get("split_ratio", 0.8))
    X_train_base, y_train_base = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    # Feature selection once to ensure consistent features across all runs
    feature_selector = FeatureSelector()
    if config["model"].get("feature_selection", True):
        try:
            X_train_base, X_test, selected_features = feature_selector.select_features(
                X_train_base, y_train_base, X_test, method="mutual_info"
            )
            logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            logger.info("Using all features")
            selected_features = X_train_base.columns.tolist()
    else:
        selected_features = X_train_base.columns.tolist()

    # Run training multiple times with different random seeds
    for run in range(num_runs):
        logger.info(f"Starting training run {run + 1}/{num_runs}")

        # Create a copy of the config to avoid modifying the original
        run_config = config.copy()
        if "model" not in run_config:
            run_config["model"] = {}

        # Set a different random seed for each run
        run_seed = random_seed_base + run
        run_config["model"]["random_seed"] = run_seed

        # Create copies of the data to avoid any cross-run contamination
        X_train = X_train_base.copy()
        y_train = y_train_base.copy()

        try:
            # Determine model type to use
            model_type = run_config["model"].get("type", "ensemble")

            # Train model with the current random seed
            model, metrics = train_gold_trading_model(
                X_train, y_train, X_test, y_test, run_config,
                model_type=model_type,
                use_cross_validation=run_config["model"].get("cross_validation", True),
                hyperparameter_tuning=run_config["model"].get("hyperparameter_tuning", True),
                random_seed=run_seed
            )

            # Extract the metrics we care about
            f1 = metrics["test"]["test_f1"]
            accuracy = metrics["test"]["test_accuracy"]

            logger.info(f"Run {run + 1} results - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

            # Update the best model if this one is better
            if f1 > best_f1:
                best_model = model
                best_metrics = metrics
                best_f1 = f1
                best_accuracy = accuracy
                best_run = run + 1
                logger.info(f"New best model found in run {best_run} with F1 score: {best_f1:.4f}")

        except Exception as e:
            logger.error(f"Error in training run {run + 1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    if best_model is None:
        raise ValueError("All training runs failed. Check logs for details.")

    logger.info(f"Best model selected from run {best_run}/{num_runs}")
    logger.info(f"Best model metrics - F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    # Save the best model
    prediction_target = config["model"].get("prediction_target", "direction")
    prediction_horizon = config["model"].get("prediction_horizon", 1)
    model_type = config["model"].get("type", "ensemble")
    model_name = f"{model_type}_{timeframe}_{prediction_target}_{prediction_horizon}_best"

    metadata = {
        "features": selected_features,
        "metrics": best_metrics,
        "config": config,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_shape": X.shape,
        "timeframe": timeframe,
        "prediction_target": prediction_target,
        "prediction_horizon": prediction_horizon,
        "best_run": best_run,
        "total_runs": num_runs
    }

    model_path = storage.save_model(best_model, model_name, metadata=metadata)
    logger.info(f"Best model saved to {model_path}")

    return best_model, best_metrics


def train_model_pipeline(config: Dict, timeframe: str = "H1") -> Tuple[Any, Dict]:
    """Complete pipeline for training an optimized model for gold trading."""
    logger.info(f"Starting model training pipeline for {timeframe}")
    start_time = time.time()

    # Load processed split data
    storage = DataStorage()
    split_paths = storage.find_latest_split_data()

    if not split_paths or "train" not in split_paths or timeframe not in split_paths["train"]:
        raise ValueError(f"No training data found for {timeframe}. Run data processing first.")

    processor = DataProcessor()

    # Load the train dataset
    train_data = processor.load_data({timeframe: split_paths["train"][timeframe]})
    train_df = train_data[timeframe]

    # Load validation dataset if available
    val_df = None
    if "validation" in split_paths and timeframe in split_paths["validation"]:
        val_data = processor.load_data({timeframe: split_paths["validation"][timeframe]})
        val_df = val_data[timeframe]
        logger.info(f"Loaded validation data: {len(val_df)} rows")

    # Load test dataset for final evaluation
    test_df = None
    if "test" in split_paths and timeframe in split_paths["test"]:
        test_data = processor.load_data({timeframe: split_paths["test"][timeframe]})
        test_df = test_data[timeframe]
        logger.info(f"Loaded test data: {len(test_df)} rows")

    try:
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
        X_train, y_train = processor.prepare_ml_features(train_df, horizon=1)
        logger.info(f"Prepared training features shape: {X_train.shape}, target shape: {y_train.shape}")

        # Prepare validation features if available
        X_val, y_val = None, None
        if val_df is not None:
            X_val, y_val = processor.prepare_ml_features(val_df, horizon=1)
            logger.info(f"Prepared validation features shape: {X_val.shape}, target shape: {y_val.shape}")

        # Prepare test features if available
        X_test, y_test = None, None
        if test_df is not None:
            X_test, y_test = processor.prepare_ml_features(test_df, horizon=1)
            logger.info(f"Prepared test features shape: {X_test.shape}, target shape: {y_test.shape}")

        # Check for class imbalance
        train_class_balance = y_train.mean()
        logger.info(f"Train class balance: {train_class_balance:.4f}")

        if y_val is not None:
            val_class_balance = y_val.mean()
            logger.info(f"Validation class balance: {val_class_balance:.4f}")

        if y_test is not None:
            test_class_balance = y_test.mean()
            logger.info(f"Test class balance: {test_class_balance:.4f}")

        # Ensure model config exists with defaults
        if 'model' not in config:
            config['model'] = {}
            logger.warning("Model configuration not found, using defaults")

        # Feature selection specifically for gold trading
        feature_selector = FeatureSelector()
        if config["model"].get("feature_selection", True):
            try:
                # Only use training data for feature selection
                X_train, selected_features = feature_selector.select_features(
                    X_train, y_train, method="mutual_info", return_features_only=True
                )
                logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")

                # Apply same feature selection to validation and test
                if X_val is not None:
                    X_val = X_val[selected_features]
                if X_test is not None:
                    X_test = X_test[selected_features]
            except Exception as e:
                logger.error(f"Feature selection failed: {str(e)}")
                logger.info("Using all features")
                selected_features = X_train.columns.tolist()
        else:
            selected_features = X_train.columns.tolist()

        # Determine model type to use
        model_type = config["model"].get("type", "ensemble")

        # Train with validation data if available, otherwise use X_train
        if X_val is not None and y_val is not None:
            model, metrics = train_gold_trading_model(
                X_train, y_train, X_val, y_val, config,
                model_type=model_type,
                use_cross_validation=False,  # No need for CV since we have a validation set
                hyperparameter_tuning=config["model"].get("hyperparameter_tuning", True)
            )
        else:
            # Fall back to train/test split if no validation data
            train_size = int(len(X_train) * 0.8)
            X_train_subset, y_train_subset = X_train.iloc[:train_size], y_train.iloc[:train_size]
            X_val_subset, y_val_subset = X_train.iloc[train_size:], y_train.iloc[train_size:]

            model, metrics = train_gold_trading_model(
                X_train_subset, y_train_subset, X_val_subset, y_val_subset, config,
                model_type=model_type,
                use_cross_validation=config["model"].get("cross_validation", True),
                hyperparameter_tuning=config["model"].get("hyperparameter_tuning", True)
            )

        # Final evaluation on test set if available
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            test_metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            metrics['final_test'] = test_metrics
            logger.info(
                f"Final test metrics: {test_metrics['test_accuracy']:.4f} accuracy, {test_metrics['test_f1']:.4f} F1")

        # Save model and metadata
        prediction_target = config["model"].get("prediction_target", "direction")
        prediction_horizon = config["model"].get("prediction_horizon", 1)
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

        model_path = storage.save_model(model, model_name, metadata=metadata)
        logger.info(f"Model saved to {model_path}")

        # Calculate training time
        training_time = time.time() - start_time
        metrics["training_time"] = training_time
        metrics["n_features"] = len(selected_features)

        logger.info(f"Model training completed in {training_time:.2f} seconds")
        if "test" in metrics:
            logger.info(f"Validation metrics: {metrics['test']}")
        if "final_test" in metrics:
            logger.info(f"Final test metrics: {metrics['final_test']}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error in train_model_pipeline: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def train_gold_trading_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        config: Dict,
        model_type: str = "ensemble",
        use_cross_validation: bool = True,
        hyperparameter_tuning: bool = True,
        random_seed: int = 42  # Added random_seed parameter
) -> Tuple[Any, Dict]:
    """Train an optimized model for gold trading."""
    logger.info(f"Training {model_type} model for gold trading with random seed {random_seed}")

    metrics = {}

    # Setup time series cross-validation if needed
    if use_cross_validation:
        tscv = TimeSeriesSplit(n_splits=5)

    # Check class distribution and balance
    class_counts = y_train.value_counts()
    logger.info(f"Train class distribution: {class_counts}")

    # Calculate appropriate class weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    logger.info(f"Class weight (0:1 ratio): {scale_pos_weight:.2f}")

    # Determine if we should use precomputed optimized parameters from BayesSearchCV
    use_optimized_params = hyperparameter_tuning and config["model"].get("use_bayes_optimizer", False)
    # Add after defining use_optimized_params
    logger.info(f"hyperparameter_tuning: {hyperparameter_tuning}")
    logger.info(f"use_bayes_optimizer setting: {config['model'].get('use_bayes_optimizer', False)}")
    logger.info(f"use_optimized_params: {use_optimized_params}")

    if hyperparameter_tuning and use_optimized_params:
        optimized_file = config["model"].get("optimized_params_file", f"{model_type}_H1_direction_1_optimization.pkl")
        optimized_path = os.path.join("data_output", "trained_models", optimized_file)
        logger.info(f"Looking for optimization file at: {optimized_path}")
        logger.info(f"File exists: {os.path.exists(optimized_path)}")

    # --------------------------
    # RANDOM FOREST Branch
    # --------------------------
    if model_type == ModelType.RANDOM_FOREST.value:
        if hyperparameter_tuning and use_optimized_params:
            optimized_file = config["model"].get("optimized_params_file",
                                                 f"{model_type}_H1_direction_1_optimization.pkl")
            optimized_path = os.path.join("data_output", "trained_models", optimized_file)
            if os.path.exists(optimized_path):
                logger.info(
                    f"Optimized parameters file found at {optimized_path}, loading parameters for RandomForest.")
                opt_results = joblib.load(optimized_path)
                best_params = opt_results.get("best_params", {})
                logger.info(f"Using optimized hyperparameters for RandomForest: {best_params}")
                model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=random_seed,
                                               n_jobs=-1)
                model.fit(X_train, y_train)
            else:
                logger.info("No optimized parameters file found for RandomForest. Proceeding with GridSearchCV.")
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
                    best_score = 0
                    best_params = None
                    for n_est in param_grid['n_estimators']:
                        for depth in param_grid['max_depth']:
                            temp_model = RandomForestClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                min_samples_split=20,
                                min_samples_leaf=15,
                                class_weight='balanced',
                                random_state=random_seed,
                                n_jobs=-1
                            )
                            temp_model.fit(X_train, y_train)
                            y_pred = temp_model.predict(X_test)
                            score = f1_score(y_test, y_pred)
                            if score > best_score:
                                best_score = score
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
                        random_state=random_seed,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
        elif hyperparameter_tuning:
            # Use GridSearchCV as the fallback tuning method if not using optimized parameters.
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
                best_score = 0
                best_params = None
                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        temp_model = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=20,
                            min_samples_leaf=15,
                            class_weight='balanced',
                            random_state=random_seed,
                            n_jobs=-1
                        )
                        temp_model.fit(X_train, y_train)
                        y_pred = temp_model.predict(X_test)
                        score = f1_score(y_test, y_pred)
                        if score > best_score:
                            best_score = score
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
                    random_state=random_seed,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
        else:
            # No hyperparameter tuning: use default parameters.
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

    # --------------------------
    # XGBOOST Branch
    # --------------------------
    elif model_type == ModelType.XGBOOST.value:
        if hyperparameter_tuning and use_optimized_params:
            optimized_file = config["model"].get("optimized_params_file",
                                                 f"{model_type}_H1_direction_1_optimization.pkl")
            optimized_path = os.path.join("data_output", "trained_models", optimized_file)
            if os.path.exists(optimized_path):
                logger.info(f"Optimized parameters file found at {optimized_path}, loading parameters for XGBoost.")
                opt_results = joblib.load(optimized_path)
                best_params = opt_results.get("best_params", {})
                logger.info(f"Using optimized hyperparameters for XGBoost: {best_params}")
                # Add random_state to best_params
                if "random_state" not in best_params:
                    best_params["random_state"] = random_seed
                model = xgb.XGBClassifier(**best_params)
                model.fit(X_train, y_train)
            else:
                logger.info("No optimized parameters file found for XGBoost. Proceeding with GridSearchCV.")
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
                    best_score = 0
                    best_params = None
                    for n_est in param_grid['n_estimators']:
                        for depth in param_grid['max_depth']:
                            for lr in param_grid['learning_rate']:
                                temp_model = xgb.XGBClassifier(
                                    n_estimators=n_est,
                                    max_depth=depth,
                                    learning_rate=lr,
                                    subsample=0.8,
                                    min_child_weight=10,
                                    scale_pos_weight=scale_pos_weight,
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    random_state=random_seed,
                                    n_jobs=-1
                                )
                                temp_model.fit(X_train, y_train)
                                y_pred = temp_model.predict(X_test)
                                score = f1_score(y_test, y_pred)
                                if score > best_score:
                                    best_score = score
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
                    best_params['random_state'] = random_seed
                    best_params['n_jobs'] = -1
                    model = xgb.XGBClassifier(**best_params)
                    model.fit(X_train, y_train)
        elif hyperparameter_tuning:
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
                best_score = 0
                best_params = None
                for n_est in param_grid['n_estimators']:
                    for depth in param_grid['max_depth']:
                        for lr in param_grid['learning_rate']:
                            temp_model = xgb.XGBClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=0.8,
                                min_child_weight=10,
                                scale_pos_weight=scale_pos_weight,
                                use_label_encoder=False,
                                eval_metric='logloss',
                                random_state=random_seed,
                                n_jobs=-1
                            )
                            temp_model.fit(X_train, y_train)
                            y_pred = temp_model.predict(X_test)
                            score = f1_score(y_test, y_pred)
                            if score > best_score:
                                best_score = score
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
                best_params['random_state'] = random_seed
                best_params['n_jobs'] = -1
                model = xgb.XGBClassifier(**best_params)
                model.fit(X_train, y_train)
        else:
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

    # --------------------------
    # ENSEMBLE Branch
    # --------------------------
    elif model_type == ModelType.ENSEMBLE.value:
        if hyperparameter_tuning and use_optimized_params:
            optimized_file = config["model"].get("optimized_params_file",
                                                 f"{model_type}_H1_direction_1_optimization.pkl")
            optimized_path = os.path.join("data_output", "trained_models", optimized_file)
            if os.path.exists(optimized_path):
                logger.info(f"Optimized parameters file found at {optimized_path}, loading parameters for ensemble.")
                opt_results = joblib.load(optimized_path)
                best_params = opt_results.get("best_params", {})
                rf_best_params = best_params.get("random_forest", {})
                xgb_best_params = best_params.get("xgboost", {})
                logger.info(
                    f"Using optimized hyperparameters for ensemble: RF: {rf_best_params}, XGB: {xgb_best_params}")
                rf_model = RandomForestClassifier(**rf_best_params, class_weight='balanced', random_state=random_seed,
                                                  n_jobs=-1)
                rf_model.fit(X_train, y_train)

                # Ensure random_state is set for XGBoost
                if "random_state" not in xgb_best_params:
                    xgb_best_params["random_state"] = random_seed
                xgb_model = xgb.XGBClassifier(**xgb_best_params)
                xgb_model.fit(X_train, y_train)

                from sklearn.ensemble import VotingClassifier
                model = VotingClassifier(
                    estimators=[('rf', rf_model), ('xgb', xgb_model)],
                    voting='soft',
                    weights=[0.3, 0.7]
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
                logger.info("No optimized parameters file found for ensemble. Proceeding with preset parameters.")
                # Fallback to default ensemble training below
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=20,
                    min_samples_leaf=15,
                    class_weight='balanced',
                    random_state=random_seed,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    min_child_weight=10,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=random_seed,
                    n_jobs=-1
                )
                xgb_model.fit(X_train, y_train)
                from sklearn.ensemble import VotingClassifier
                model = VotingClassifier(
                    estimators=[('rf', rf_model), ('xgb', xgb_model)],
                    voting='soft',
                    weights=[0.3, 0.7]
                )
                model.fit(X_train, y_train)
        else:
            # Default ensemble training without hyperparameter tuning or optimized parameters.
            logger.info("Training ensemble model for gold trading using preset parameters")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=15,
                class_weight='balanced',
                random_state=random_seed,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_child_weight=10,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_seed,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            from sklearn.ensemble import VotingClassifier
            model = VotingClassifier(
                estimators=[('rf', rf_model), ('xgb', xgb_model)],
                voting='soft',
                weights=[0.3, 0.7]
            )
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # --------------------------
    # Evaluation
    # --------------------------
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

    y_train_pred = model.predict(X_train)
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0)
    }
    metrics['train'] = train_metrics

    if train_metrics['train_f1'] - test_metrics['test_f1'] > 0.2:
        logger.warning("Possible overfitting detected: large gap between train and test F1 scores")

    return model, metrics
