import pandas as pd
import numpy as np
from models.factory import ModelFactory
from data.processor import DataProcessor
from data.storage import DataStorage
from utils.logger import setup_logger
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


def analyze_feature_importance():
    """Analyze and select the most important features."""
    # Set up both logging and direct console output
    logger = setup_logger("FeatureAnalyzer")
    print("Starting feature importance analysis")

    # Load processed data
    storage = DataStorage()
    processed_data = storage.find_latest_processed_data()
    timeframe = "H1"

    if timeframe not in processed_data:
        print(f"No processed data found for timeframe {timeframe}")
        return

    data_path = processed_data[timeframe]
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded data shape: {df.shape}")

    # Prepare features for analysis
    processor = DataProcessor()
    horizon = 1
    X, y = processor.prepare_ml_features(df, horizon=horizon)
    print(f"Prepared features: {X.shape}, target: {y.shape}")

    # Split data into train/test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Method 1: Analyze model's feature importance
    try:
        model_path = "../data_output/trained_models/ensemble_H1_direction_1_20250415_075832.joblib"
        model = ModelFactory.load_model(model_path)

        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
            if importances:
                sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                print("\nTop 20 features from model:")
                for feature, importance in sorted_importances[:20]:
                    print(f"  {feature}: {importance:.6f}")
    except Exception as e:
        print(f"Error analyzing model feature importance: {str(e)}")

    # Method 2: Use statistical feature selection
    try:
        # Select top K features based on ANOVA F-value
        selector = SelectKBest(f_classif, k=20)
        X_new = selector.fit_transform(X_train, y_train)

        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices].tolist()

        # Get scores
        scores = selector.scores_
        feature_scores = list(zip(X_train.columns, scores))
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)

        print("\nTop 20 features by statistical selection:")
        for feature, score in sorted_features[:20]:
            print(f"  {feature}: {score:.6f}")

        # Test these features
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        # Train on selected features only
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train[selected_features], y_train)

        # Evaluate
        y_pred = clf.predict(X_test[selected_features])
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel with top 20 features - Accuracy: {accuracy:.4f}")

        # Compare with baseline (simple moving average crossover)
        sma_fast_period = 9
        sma_slow_period = 21

        # Calculate moving averages on the original data
        df['sma_fast'] = df['close'].rolling(window=sma_fast_period).mean()
        df['sma_slow'] = df['close'].rolling(window=sma_slow_period).mean()

        # Generate signals: 1 when fast crosses above slow, 0 when fast crosses below slow
        df['sma_signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'sma_signal'] = 1

        # Shift to avoid lookahead bias and match our prediction target
        df['sma_strategy'] = df['sma_signal'].shift(1)

        # Calculate accuracy on test period
        test_indices = y_test.index
        test_actual = y_test.values
        test_sma_pred = df.loc[test_indices, 'sma_strategy'].values

        # Handle NaNs
        valid_mask = ~np.isnan(test_sma_pred)
        baseline_accuracy = np.mean(test_actual[valid_mask] == test_sma_pred[valid_mask])

        print(f"Baseline SMA crossover - Accuracy: {baseline_accuracy:.4f}")

        # Train a simpler model with just a few key features
        key_features = [
            'close', 'high', 'low',  # Basic price data
            'sma_50', 'sma_200',  # Long-term trend
            'rsi_14',  # Momentum
            'BBU_20_2.0', 'BBL_20_2.0',  # Volatility
            'atr_14',  # Volatility
            'macd_cross_up', 'macd_cross_down'  # Trend signals
        ]

        # Filter to available features
        available_features = [f for f in key_features if f in X_train.columns]
        print(f"\nTraining with {len(available_features)} basic features")

        simple_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        simple_clf.fit(X_train[available_features], y_train)

        # Evaluate
        simple_pred = simple_clf.predict(X_test[available_features])
        simple_accuracy = accuracy_score(y_test, simple_pred)
        print(f"Simple model accuracy: {simple_accuracy:.4f}")

    except Exception as e:
        print(f"Error in statistical feature selection: {str(e)}")
        import traceback
        print(traceback.format_exc())

    print("\nFeature analysis complete")


if __name__ == "__main__":
    analyze_feature_importance()