import os
import sys
import pandas as pd
import numpy as np
from models.factory import ModelFactory
from utils.logger import setup_logger
from data.storage import DataStorage
from data.processor import DataProcessor

logger = setup_logger("ModelTest")


def test_model_predict(model_path, timeframe='H1'):
    """Test model prediction with real data features"""
    logger.info(f"Loading model from {model_path}")

    try:
        # Load the model
        model = ModelFactory.load_model(model_path)
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model has predict: {hasattr(model, 'predict')}")
        logger.info(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")

        # Get actual feature names used during training by loading a small sample of real data
        logger.info("Loading real data sample to get feature names")
        storage = DataStorage()
        processed_data_paths = storage.find_latest_processed_data()

        if timeframe not in processed_data_paths:
            logger.error(f"No processed data found for timeframe {timeframe}")
            return

        # Load just 100 rows to get feature names
        data = pd.read_csv(processed_data_paths[timeframe], index_col=0, parse_dates=True, nrows=100)
        logger.info(f"Loaded data sample with shape: {data.shape}")

        # Extract feature names (exclude target columns)
        feature_cols = [col for col in data.columns if not col.startswith('target_')]
        logger.info(f"Using {len(feature_cols)} real features")

        # Create a small test dataset with the same features
        test_rows = 10
        test_data = data.iloc[:test_rows].copy()[feature_cols]
        logger.info(f"Test data shape: {test_data.shape}")

        # Try making predictions
        try:
            logger.info("Attempting to predict...")
            pred = model.predict(test_data)
            logger.info(f"Prediction successful! Shape: {pred.shape}")
            logger.info(f"Prediction values: {pred}")

            if hasattr(model, 'predict_proba'):
                logger.info("Attempting to get prediction probabilities...")
                proba = model.predict_proba(test_data)
                logger.info(f"Probability shape: {proba.shape}")
                logger.info(f"Probability values: {proba[:5]}")

                # Check if any predictions cross confidence threshold
                confidence_threshold = 0.65
                buy_signals = (proba[:, 1] >= confidence_threshold).sum()
                sell_signals = (proba[:, 1] <= (1 - confidence_threshold)).sum()
                logger.info(f"Using threshold {confidence_threshold}:")
                logger.info(f"  Buy signals: {buy_signals}/{test_rows}")
                logger.info(f"  Sell signals: {sell_signals}/{test_rows}")

                # Try lower thresholds
                for threshold in [0.6, 0.55, 0.5]:
                    buy_signals = (proba[:, 1] >= threshold).sum()
                    sell_signals = (proba[:, 1] <= (1 - threshold)).sum()
                    logger.info(f"With threshold {threshold}:")
                    logger.info(f"  Buy signals: {buy_signals}/{test_rows}")
                    logger.info(f"  Sell signals: {sell_signals}/{test_rows}")

        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")


if __name__ == "__main__":
    model_path = "models/ensemble_H1_direction_12.joblib"
    test_model_predict(model_path)