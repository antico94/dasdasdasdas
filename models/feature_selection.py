import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import List, Tuple, Union, Dict

from utils.logger import setup_logger

logger = setup_logger("FeatureSelector")


class FeatureSelector:
    """Optimized feature selection for gold trading models."""

    def select_features(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            method: str = "mutual_info",
            k: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Select the most important features for gold trading prediction.

        Parameters:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            method: Feature selection method ('mutual_info', 'random_forest', 'xgboost')
            k: Number of features to select (if None, use automatic threshold)

        Returns:
            X_train_selected: Training data with selected features
            X_test_selected: Test data with selected features (if X_test provided)
            selected_features: List of selected feature names
        """
        logger.info(f"Selecting features using {method} method")

        if method == "mutual_info":
            # Mutual information measures dependence between variables
            # Good for capturing non-linear relationships in gold price data
            if k is None:
                k = min(int(X_train.shape[1] * 0.6), 30)  # Select up to 60% of features, max 30

            selector = SelectKBest(mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)

            # Get feature names
            mask = selector.get_support()
            selected_features = X_train.columns[mask].tolist()

            # Important: sort features by importance
            scores = selector.scores_[mask]
            features_scores = list(zip(selected_features, scores))
            features_scores.sort(key=lambda x: x[1], reverse=True)

            selected_features = [f[0] for f in features_scores]
            logger.info(f"Selected {len(selected_features)} features using mutual information")

            # Log top features and their scores
            for feature, score in features_scores[:10]:
                logger.info(f"Feature: {feature}, Score: {score:.4f}")

        elif method == "random_forest":
            # Random Forest feature importance - good for capturing complex interactions
            if k is None:
                k = min(int(X_train.shape[1] * 0.6), 30)  # Select up to 60% of features, max 30

            # Use Random Forest with parameters optimized for financial data
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            # Use SelectFromModel with a threshold
            selector = SelectFromModel(
                rf,
                threshold='mean',  # Use mean importance as threshold
                prefit=False
            )

            X_train_selected = selector.fit_transform(X_train, y_train)

            # Get feature names and importances
            mask = selector.get_support()
            selected_features = X_train.columns[mask].tolist()

            # Get and sort feature importances
            importances = selector.estimator_.feature_importances_[mask]
            features_importances = list(zip(selected_features, importances))
            features_importances.sort(key=lambda x: x[1], reverse=True)

            # Select top k features if we have more than k
            if len(features_importances) > k:
                features_importances = features_importances[:k]
                selected_features = [f[0] for f in features_importances]

            logger.info(f"Selected {len(selected_features)} features using Random Forest importance")

            # Log top features and their importances
            for feature, importance in features_importances[:10]:
                logger.info(f"Feature: {feature}, Importance: {importance:.4f}")

        elif method == "xgboost":
            # XGBoost feature importance - good for capturing non-linear relationships
            if k is None:
                k = min(int(X_train.shape[1] * 0.6), 30)  # Select up to 60% of features, max 30

            # Calculate class weights for imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0

            # Use XGBoost with parameters optimized for financial data
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_child_weight=5,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )

            # Use SelectFromModel with a threshold
            selector = SelectFromModel(
                xgb_model,
                threshold='mean',  # Use mean importance as threshold
                prefit=False
            )

            X_train_selected = selector.fit_transform(X_train, y_train)

            # Get feature names and importances
            mask = selector.get_support()
            selected_features = X_train.columns[mask].tolist()

            # Get and sort feature importances
            importances = selector.estimator_.feature_importances_[mask]
            features_importances = list(zip(selected_features, importances))
            features_importances.sort(key=lambda x: x[1], reverse=True)

            # Select top k features if we have more than k
            if len(features_importances) > k:
                features_importances = features_importances[:k]
                selected_features = [f[0] for f in features_importances]

            logger.info(f"Selected {len(selected_features)} features using XGBoost importance")

            # Log top features and their importances
            for feature, importance in features_importances[:10]:
                logger.info(f"Feature: {feature}, Importance: {importance:.4f}")

        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        # Convert to DataFrame with feature names
        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=selected_features,
            index=X_train.index
        )

        # Transform test data if provided
        if X_test is not None:
            if method == "mutual_info":
                # For mutual info, we need to select the columns manually
                X_test_selected = X_test[selected_features].copy()
            else:
                # For model-based methods, use the transform method
                X_test_selected = pd.DataFrame(
                    selector.transform(X_test),
                    columns=selected_features,
                    index=X_test.index
                )
        else:
            X_test_selected = None

        logger.info(f"Final feature list: {', '.join(selected_features[:10])}...")
        if len(selected_features) > 10:
            logger.info(f"...and {len(selected_features) - 10} more features")

        return X_train_selected, X_test_selected, selected_features

    def select_features_for_splits(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            validation_test_data: Dict[str, Tuple[pd.DataFrame, pd.Series]] = None,
            method: str = "mutual_info",
            k: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
        """
        Select features using training data and apply the same selection to validation/test data.

        Parameters:
            X_train: Training features
            y_train: Training target
            validation_test_data: Dictionary with 'validation' and/or 'test' keys, each containing (X, y) tuple
            method: Feature selection method ('mutual_info', 'random_forest', 'xgboost')
            k: Number of features to select (if None, use automatic threshold)

        Returns:
            X_train_selected: Training data with selected features
            transformed_sets: Dictionary with transformed validation and test sets
            selected_features: List of selected feature names
        """
        logger.info(f"Selecting features for train/validation/test splits using {method} method")

        # Selection is always based only on training data to prevent data leakage
        X_train_selected, _, selected_features = self.select_features(
            X_train, y_train, None, method, k
        )

        # Apply the same feature selection to validation and test data
        transformed_sets = {}

        if validation_test_data:
            for split_name, (X, y) in validation_test_data.items():
                if X is not None:
                    # Select the same features as in training data
                    X_selected = X[selected_features].copy()
                    transformed_sets[split_name] = X_selected
                    logger.info(f"Applied feature selection to {split_name} data: {X_selected.shape}")

        return X_train_selected, transformed_sets, selected_features

    def gold_specific_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance specifically optimized for gold trading.
        This uses a weighted ensemble approach combining multiple importance metrics.

        Parameters:
            X: Features DataFrame
            y: Target Series

        Returns:
            A DataFrame with features ranked by combined importance
        """
        logger.info("Calculating gold-specific feature importance")

        # 1. Mutual Information (captures non-linear relationships)
        mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        })
        mi_df['mi_rank'] = mi_df['mutual_info'].rank(ascending=False)

        # 2. Random Forest importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        rf_importances = rf.feature_importances_
        rf_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf_importances
        })
        rf_df['rf_rank'] = rf_df['rf_importance'].rank(ascending=False)

        # 3. XGBoost importance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb_model.fit(X, y)
        xgb_importances = xgb_model.feature_importances_
        xgb_df = pd.DataFrame({
            'feature': X.columns,
            'xgb_importance': xgb_importances
        })
        xgb_df['xgb_rank'] = xgb_df['xgb_importance'].rank(ascending=False)

        # Combine all importance metrics
        combined_df = mi_df.merge(rf_df, on='feature')
        combined_df = combined_df.merge(xgb_df, on='feature')

        # Calculate a weighted rank (giving more weight to XGBoost for financial data)
        combined_df['weighted_rank'] = (
                0.3 * combined_df['mi_rank'] +
                0.3 * combined_df['rf_rank'] +
                0.4 * combined_df['xgb_rank']
        )

        # Sort by weighted rank
        combined_df = combined_df.sort_values('weighted_rank')

        # Add a normalized combined score (1 = best)
        max_rank = combined_df['weighted_rank'].max()
        combined_df['combined_score'] = 1 - (combined_df['weighted_rank'] / max_rank)

        logger.info(f"Top 10 features for gold trading:")
        for i, (_, row) in enumerate(combined_df.head(10).iterrows()):
            logger.info(f"{i + 1}. {row['feature']} (Score: {row['combined_score']:.4f})")

        return combined_df