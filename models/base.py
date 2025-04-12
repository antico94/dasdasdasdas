from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Base class for all trading models."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_importance = {}
        self.is_fitted = False

    @abstractmethod
    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities for classification models."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances."""
        pass

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        import joblib
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load the model from disk."""
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True

    def get_params(self) -> Dict:
        """Get model parameters."""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

    def get_hyperparameters(self) -> Dict:
        """Get hyperparameters used for training."""
        return self.get_params()