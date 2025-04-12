from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from config.constants import PredictionTarget
from models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model for classification or regression tasks."""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.target_type = config.get('model', {}).get('prediction_target', 'direction')
        self.is_classifier = self.target_type in [PredictionTarget.DIRECTION.value,
                                                  PredictionTarget.VOLATILITY.value]

        # Default hyperparameters
        self.hyperparams = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt' if self.is_classifier else 'auto',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced' if self.is_classifier else None
        }

        # Update with user-provided hyperparameters if available
        if 'hyperparameters' in config.get('model', {}) and 'random_forest' in config['model']['hyperparameters']:
            self.hyperparams.update(config['model']['hyperparameters']['random_forest'])

        # Initialize the model
        if self.is_classifier:
            self.model = RandomForestClassifier(**self.hyperparams)
        else:
            self.model = RandomForestRegressor(**self.hyperparams)

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        """Train the model on the provided data."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate feature importance
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
        importances = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))

    def get_hyperparameters(self) -> Dict:
        """Get hyperparameters used for training."""
        return self.hyperparams