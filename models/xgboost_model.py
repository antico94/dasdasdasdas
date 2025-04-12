from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from config.constants import PredictionTarget
from models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for classification or regression tasks."""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.target_type = config.get('model', {}).get('prediction_target', 'direction')
        self.is_classifier = self.target_type in [PredictionTarget.DIRECTION.value,
                                                  PredictionTarget.VOLATILITY.value]

        # Default hyperparameters
        self.hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'objective': 'binary:logistic' if self.is_classifier else 'reg:squarederror',
            'eval_metric': 'logloss' if self.is_classifier else 'rmse',
            'random_state': 42,
            'n_jobs': -1
        }

        # Update with user-provided hyperparameters if available
        if 'hyperparameters' in config.get('model', {}) and 'xgboost' in config['model']['hyperparameters']:
            self.hyperparams.update(config['model']['hyperparameters']['xgboost'])

        # Initialize the model
        if self.is_classifier:
            self.model = xgb.XGBClassifier(**self.hyperparams)
        else:
            self.model = xgb.XGBRegressor(**self.hyperparams)

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        """Train the model on the provided data."""
        fit_params = {}

        # Add evaluation set if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]

            # Try with early_stopping_rounds
            try:
                self.model.fit(
                    X_train, y_train,
                    **fit_params,
                    early_stopping_rounds=10,
                    verbose=False
                )
            except TypeError:
                # If early_stopping_rounds is not supported, try with callbacks
                try:
                    callbacks = [xgb.callback.EarlyStopping(rounds=10)]
                    self.model.fit(
                        X_train, y_train,
                        **fit_params,
                        callbacks=callbacks,
                        verbose=False
                    )
                except (TypeError, AttributeError):
                    # If neither work, just fit without early stopping
                    self.model.fit(X_train, y_train, **fit_params)
        else:
            # If no validation data, just do a simple fit
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        self._calculate_feature_importance(X_train.columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities for classification models."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        if self.is_classifier:
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba() is only available for classification models.")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.feature_importance

    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """Calculate and store feature importances."""
        # Use gain importance for XGBoost
        importance_type = 'gain'
        try:
            if hasattr(self.model, 'get_booster'):
                # Get feature importances - handle different XGBoost versions
                try:
                    importances = self.model.get_booster().get_score(importance_type=importance_type)
                    # Convert feature map to proper format
                    importance_dict = {
                        name: importances.get(f"f{i}", 0) if name not in importances else importances[name]
                        for i, name in enumerate(feature_names)
                    }
                except (KeyError, ValueError, AttributeError):
                    # Fallback to feature_importances_ attribute
                    importances = self.model.feature_importances_
                    importance_dict = dict(zip(feature_names, importances))
            else:
                importances = self.model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
        except (AttributeError, TypeError):
            # If feature importance extraction fails, create a placeholder with equal values
            importance_dict = {name: 1.0 / len(feature_names) for name in feature_names}

        # Normalize importances
        total = sum(importance_dict.values()) if sum(importance_dict.values()) > 0 else 1.0
        self.feature_importance = {k: v / total for k, v in importance_dict.items()}

    def get_hyperparameters(self) -> Dict:
        """Get hyperparameters used for training."""
        return self.hyperparams