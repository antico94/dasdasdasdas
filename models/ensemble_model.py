from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor

from config.constants import PredictionTarget
from models.base import BaseModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple models for better predictions."""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.target_type = config.get('model', {}).get('prediction_target', 'direction')
        self.is_classifier = self.target_type in [PredictionTarget.DIRECTION.value,
                                                  PredictionTarget.VOLATILITY.value]

        # Sub-models to include in ensemble
        self.models_to_use = config.get('model', {}).get('ensemble_models',
                                                         ['random_forest', 'xgboost'])

        # Weights for each model (default to equal weights)
        self.model_weights = config.get('model', {}).get('ensemble_weights', None)

        # Create sub-models
        self.sub_models = []
        self.fitted_sub_models = []

        if 'random_forest' in self.models_to_use:
            self.sub_models.append(('random_forest', RandomForestModel(config)))

        if 'xgboost' in self.models_to_use:
            self.sub_models.append(('xgboost', XGBoostModel(config)))

        if 'lstm' in self.models_to_use:
            # LSTM will be handled separately as it's not compatible with sklearn's voting
            self.lstm_model = LSTMModel(config)
            # LSTM weight if using weighted predictions
            self.lstm_weight = 1.0
        else:
            self.lstm_model = None

        # Initialize ensemble model
        if len(self.sub_models) > 0:
            if self.is_classifier:
                self.model = VotingClassifier(
                    estimators=[(name, model.model) for name, model in self.sub_models],
                    voting='soft',
                    weights=self.model_weights
                )
            else:
                self.model = VotingRegressor(
                    estimators=[(name, model.model) for name, model in self.sub_models],
                    weights=self.model_weights
                )
        else:
            self.model = None

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        """Train all sub-models and the ensemble."""
        # Train LSTM separately if included
        if self.lstm_model is not None:
            print("Training LSTM model...")
            self.lstm_model.fit(X_train, y_train, X_val, y_val)

        # Train scikit-learn compatible models
        if len(self.sub_models) > 0:
            # First train individual models to get feature importances
            for name, model in self.sub_models:
                print(f"Training {name} model...")
                model.fit(X_train, y_train, X_val, y_val)
                self.fitted_sub_models.append((name, model))

            # Then train the ensemble
            print("Training ensemble model...")
            self.model.fit(X_train, y_train)

        self.is_fitted = True

        # Combine feature importances from sub-models
        self._combine_feature_importances(X_train.columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # If we have only LSTM, return its predictions
        if self.lstm_model is not None and not self.fitted_sub_models:
            return self.lstm_model.predict(X)

        # If we have only scikit-learn models, return ensemble predictions
        if not self.lstm_model and self.fitted_sub_models:
            return self.model.predict(X)

        # If we have both, combine predictions
        if self.lstm_model and self.fitted_sub_models:
            # Get predictions from each model type
            assert self.lstm_model is not None
            lstm_preds = self.lstm_model.predict(X)
            ensemble_preds = self.model.predict(X)

            # Handle NaN values from LSTM padding
            lstm_preds_clean = np.copy(lstm_preds)
            nan_mask = np.isnan(lstm_preds)
            lstm_preds_clean[nan_mask] = ensemble_preds[nan_mask]

            # For classifier, we need to ensure consistent label types
            if self.is_classifier:
                return lstm_preds_clean.astype(int)
            else:
                return lstm_preds_clean

        raise ValueError("No models available for prediction.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities for classification models."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        if not self.is_classifier:
            raise ValueError("predict_proba() is only available for classification models.")

        # If we have only LSTM, return its probabilities
        if self.lstm_model is not None and not self.fitted_sub_models:
            return self.lstm_model.predict_proba(X)

        # If we have only scikit-learn models, return ensemble probabilities
        if not self.lstm_model and self.fitted_sub_models:
            return self.model.predict_proba(X)

        # If we have both, combine probabilities
        if self.lstm_model and self.fitted_sub_models:
            assert self.lstm_model is not None
            lstm_probas = self.lstm_model.predict_proba(X)
            ensemble_probas = self.model.predict_proba(X)

            # Handle NaN values from LSTM padding
            lstm_probas_clean = np.copy(lstm_probas)
            nan_mask = np.isnan(lstm_probas[:, 0])
            lstm_probas_clean[nan_mask] = ensemble_probas[nan_mask]

            # Weighted average of probabilities
            if self.model_weights:
                lstm_weight = self.lstm_weight
                ensemble_weight = sum(self.model_weights) / len(self.model_weights)
                total_weight = lstm_weight + ensemble_weight

                weighted_probas = (
                        (lstm_probas_clean * lstm_weight + ensemble_probas * ensemble_weight) /
                        total_weight
                )
                return weighted_probas
            else:
                # Simple average if no weights provided
                return (lstm_probas_clean + ensemble_probas) / 2

        raise ValueError("No models available for prediction.")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return combined feature importances from all sub-models."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.feature_importance

    def _combine_feature_importances(self, feature_names: List[str]) -> None:
        """Combine feature importances from all sub-models."""
        combined_importance = {feature: 0.0 for feature in feature_names}

        # Get importances from each fitted sub-model
        for name, model in self.fitted_sub_models:
            model_importance = model.get_feature_importance()
            for feature, importance in model_importance.items():
                if feature in combined_importance:
                    combined_importance[feature] += importance

        # Add LSTM importances if available
        if self.lstm_model and self.lstm_model.is_fitted:
            lstm_importance = self.lstm_model.get_feature_importance()
            for feature, importance in lstm_importance.items():
                if feature in combined_importance:
                    combined_importance[feature] += importance

        # Normalize combined importances
        total_models = len(self.fitted_sub_models)
        if self.lstm_model and self.lstm_model.is_fitted:
            total_models += 1

        if total_models > 0:
            self.feature_importance = {
                feature: importance / total_models
                for feature, importance in combined_importance.items()
            }
        else:
            self.feature_importance = combined_importance

    def save(self, filepath: str) -> None:
        """Save the ensemble model to disk."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Save sub-models
        sub_model_paths = {}
        for name, model in self.fitted_sub_models:
            sub_model_path = filepath.replace('.joblib', f'_{name}.joblib')
            model.save(sub_model_path)
            sub_model_paths[name] = sub_model_path

        # Save LSTM if available
        if self.lstm_model and self.lstm_model.is_fitted:
            lstm_path = filepath.replace('.joblib', '_lstm.joblib')
            self.lstm_model.save(lstm_path)
            sub_model_paths['lstm'] = lstm_path

        # Save ensemble metadata
        import joblib
        save_dict = {
            'is_classifier': self.is_classifier,
            'target_type': self.target_type,
            'models_to_use': self.models_to_use,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            'sub_model_paths': sub_model_paths
        }
        joblib.dump(save_dict, filepath)

    def load(self, filepath: str) -> None:
        """Load the ensemble model from disk."""
        import joblib

        # Load metadata
        save_dict = joblib.load(filepath)

        self.is_classifier = save_dict['is_classifier']
        self.target_type = save_dict['target_type']
        self.models_to_use = save_dict['models_to_use']
        self.model_weights = save_dict['model_weights']
        self.feature_importance = save_dict['feature_importance']
        self.is_fitted = save_dict['is_fitted']

        # Load sub-models
        self.fitted_sub_models = []
        for name, path in save_dict['sub_model_paths'].items():
            if name == 'lstm':
                self.lstm_model = LSTMModel({})  # Empty config as we're loading a fitted model
                self.lstm_model.load(path)
            elif name == 'random_forest':
                model = RandomForestModel({})
                model.load(path)
                self.fitted_sub_models.append((name, model))
            elif name == 'xgboost':
                model = XGBoostModel({})
                model.load(path)
                self.fitted_sub_models.append((name, model))

        # Recreate ensemble model if scikit-learn models exist
        if self.fitted_sub_models:
            if self.is_classifier:
                self.model = VotingClassifier(
                    estimators=[(name, model.model) for name, model in self.fitted_sub_models],
                    voting='soft',
                    weights=self.model_weights
                )
            else:
                self.model = VotingRegressor(
                    estimators=[(name, model.model) for name, model in self.fitted_sub_models],
                    weights=self.model_weights
                )

    def get_hyperparameters(self) -> Dict:
        """Get hyperparameters from all sub-models."""
        hyperparams = {}

        for name, model in self.fitted_sub_models:
            hyperparams[name] = model.get_hyperparameters()

        if self.lstm_model:
            hyperparams['lstm'] = self.lstm_model.get_hyperparameters()

        return hyperparams
