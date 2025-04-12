from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from config.constants import MIN_FEATURE_IMPORTANCE
from utils.logger import setup_logger


class FeatureSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.selected_features = None
        self.feature_importance = {}
        self.min_importance = config.get('model', {}).get('min_feature_importance', MIN_FEATURE_IMPORTANCE)
        self.logger = setup_logger("FeatureSelectorLogger")
        self.logger.debug(f"Initialized FeatureSelector with min_importance: {self.min_importance}")

    def select_features_from_importance(self, feature_importance: Dict[str, float]) -> List[str]:
        """Select features based on feature importance."""
        self.logger.info("Selecting features from importance dictionary.")
        self.feature_importance = feature_importance

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.logger.debug(f"Sorted features (feature: importance): {sorted_features}")

        # Select features above threshold
        selected = [feature for feature, importance in sorted_features if importance >= self.min_importance]
        self.logger.info(f"{len(selected)} features selected using threshold {self.min_importance}")

        # Ensure we have at least 5 features
        if len(selected) < 5:
            self.logger.warning("Less than 5 features met the threshold; selecting the top 5 features regardless.")
            selected = [feature for feature, _ in sorted_features[:5]]

        self.selected_features = selected
        self.logger.debug(f"Final selected features: {selected}")
        return selected

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int] = None) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        self.logger.info("Selecting features using RFE.")
        is_classifier = len(y.unique()) <= 2
        self.logger.debug(f"Data determined to be for a {'classifier' if is_classifier else 'regressor'} (unique targets: {len(y.unique())}).")

        # Default to 1/3 of features if not specified
        if n_features is None:
            n_features = max(int(X.shape[1] / 3), 5)
            self.logger.debug(f"n_features not provided, defaulting to {n_features}")

        # Create estimator
        if is_classifier:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.logger.debug("Using RandomForestClassifier for RFE.")
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            self.logger.debug("Using RandomForestRegressor for RFE.")

        # Run RFE
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        self.logger.info("RFE fitting completed.")

        # Get selected features
        selected_mask = selector.support_
        selected = X.columns[selected_mask].tolist()
        self.logger.info(f"RFE selected {len(selected)} features.")

        # Store feature importances (from ranking)
        rankings = selector.ranking_
        max_rank = np.max(rankings)
        self.feature_importance = {
            feature: (max_rank - rank + 1) / max_rank
            for feature, rank in zip(X.columns, rankings)
        }
        self.logger.debug(f"Computed feature importances from RFE: {self.feature_importance}")

        self.selected_features = selected
        self.logger.info(f"Final selected features via RFE: {selected}")
        return selected

    def select_features_from_model(self, X: pd.DataFrame, y: pd.Series, threshold: Optional[str] = 'mean') -> List[str]:
        """Select features using SelectFromModel."""
        self.logger.info("Selecting features using SelectFromModel.")
        is_classifier = len(y.unique()) <= 2
        self.logger.debug(f"Data determined to be for a {'classifier' if is_classifier else 'regressor'} (unique targets: {len(y.unique())}).")

        # Create estimator
        if is_classifier:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.logger.debug("Using RandomForestClassifier for SelectFromModel.")
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.logger.debug("Using RandomForestRegressor for SelectFromModel.")

        # Run SelectFromModel
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X, y)
        self.logger.info("SelectFromModel fitting completed.")

        # Get selected features
        selected_mask = selector.get_support()
        selected = X.columns[selected_mask].tolist()
        self.logger.info(f"SelectFromModel selected {len(selected)} features using threshold '{threshold}'.")

        # Store feature importances
        self.feature_importance = dict(zip(X.columns, selector.estimator_.feature_importances_))
        self.logger.debug(f"Feature importances from model: {self.feature_importance}")

        # Explicitly store the selected features
        self.selected_features = selected
        self.logger.debug(f"Final selected features from model: {selected}")
        return selected

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        self.logger.debug("Fetching stored feature importance scores.")
        return self.feature_importance
