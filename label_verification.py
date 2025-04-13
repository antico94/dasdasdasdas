import os
import pandas as pd
import numpy as np
from models.factory import ModelFactory
from data.processor import DataProcessor
from data.storage import DataStorage
from utils.logger import setup_logger
import sys


def verify_labels():
    logger = setup_logger("LabelVerifier")
    logger.info("Starting label verification")

    try:
        # 1. Load data
        storage = DataStorage()
        latest_processed_data = storage.find_latest_processed_data()
        timeframe = "H1"

        if timeframe not in latest_processed_data:
            logger.error(f"No processed data found for timeframe {timeframe}")
            return

        data_path = latest_processed_data[timeframe]
        logger.info(f"Loading data from: {data_path}")

        # Load just 100 rows for quick analysis
        df = pd.read_csv(data_path, index_col=0, parse_dates=True, nrows=100)
        logger.info(f"Loaded {len(df)} rows with {df.shape[1]} columns")

        # 2. Check target variable
        horizon = 12
        target_col = f'target_{horizon}'

        if target_col in df.columns:
            logger.info(f"Target column found: {target_col}")
            target_counts = df[target_col].value_counts()
            logger.info(f"Target distribution: {target_counts.to_dict()}")

            # Calculate actual price changes
            df['actual_change'] = df['close'].shift(-horizon) - df['close']

            # Count matching directions
            matching = ((df[target_col] == 1) & (df['actual_change'] > 0)) | \
                       ((df[target_col] == 0) & (df['actual_change'] <= 0))
            match_pct = matching.mean() * 100
            logger.info(f"Target matches actual direction: {match_pct:.2f}%")

            # Verify the target label logic
            logger.info("Analyzing target vs actual price movement:")
            for i in range(min(10, len(df))):
                if pd.notna(df['actual_change'].iloc[i]):
                    act_change = df['actual_change'].iloc[i]
                    target = df[target_col].iloc[i]
                    logger.info(f"Row {i}: Price change: {act_change:.4f}, Target: {target} (expect 1 if >0, 0 if ≤0)")
        else:
            logger.warning(f"Target column {target_col} not found")

        # 3. Load the model
        logger.info("Loading model")
        model_path = "models/ensemble_H1_direction_12.joblib"
        model = ModelFactory.load_model(model_path)
        logger.info(f"Loaded model type: {type(model).__name__}")

        # Remove analysis columns
        if 'actual_change' in df.columns:
            df = df.drop(columns=['actual_change'])

        # 4. Prepare features and make predictions
        logger.info("Preparing features for prediction")
        processor = DataProcessor()
        X, y = processor.prepare_ml_features(df, horizon=horizon)
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

        # Make predictions
        logger.info("Getting predictions")
        y_pred = model.predict(X)
        logger.info(f"Predictions shape: {y_pred.shape}")
        logger.info(f"Prediction distribution: {np.unique(y_pred, return_counts=True)}")

        # Check accuracy
        accuracy = (y_pred == y.values).mean()
        logger.info(f"Accuracy on sample: {accuracy:.4f}")

        # 5. Check probabilities
        if hasattr(model, 'predict_proba'):
            logger.info("Getting probability predictions")
            proba = model.predict_proba(X)
            logger.info(f"Probability shape: {proba.shape}")

            # Calculate mean probability for each class
            mean_probs = proba.mean(axis=0)
            logger.info(f"Mean probabilities: Class 0 (DOWN): {mean_probs[0]:.4f}, Class 1 (UP): {mean_probs[1]:.4f}")

            # Analyze a few examples
            logger.info("Detailed prediction examples:")
            for i in range(min(5, len(X))):
                actual = y.iloc[i]
                pred = y_pred[i]
                prob = proba[i]
                logger.info(f"Example {i}:")
                logger.info(f"  Actual: {actual} ({'UP' if actual == 1 else 'DOWN'})")
                logger.info(f"  Predicted: {pred} ({'UP' if pred == 1 else 'DOWN'})")
                logger.info(f"  Probabilities: DOWN={prob[0]:.4f}, UP={prob[1]:.4f}")

        logger.info("Label verification complete")

    except Exception as e:
        logger.error(f"Error in label verification: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    verify_labels()
    print("Script completed - check logs for details")