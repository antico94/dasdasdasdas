import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.factory import ModelFactory
from data.storage import DataStorage
from data.processor import DataProcessor
from utils.logger import setup_logger


def analyze_predictions(model_path, timeframe='H1'):
    """Focused analysis of model predictions"""
    logger = setup_logger("PredictionAnalyzer")
    logger.info(f"Analyzing predictions for model: {model_path}")

    # Load the model
    try:
        model = ModelFactory.load_model(model_path)
        logger.info(f"Successfully loaded model: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # Load data
    storage = DataStorage()
    processed_data_paths = storage.find_latest_processed_data()
    if timeframe not in processed_data_paths:
        logger.error(f"No processed data found for timeframe {timeframe}")
        return

    data_path = processed_data_paths[timeframe]
    logger.info(f"Loading data from: {data_path}")

    # Extract prediction horizon from model path
    try:
        horizon = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        horizon = 12  # Default

    # Load just 1000 rows for quick analysis
    try:
        # Get total rows first
        from utils.csv_tools import count_csv_rows
        total_rows = count_csv_rows(data_path)

        # Load the last 1000 rows (or all if less)
        nrows = min(1000, total_rows)
        skiprows = max(0, total_rows - nrows)

        logger.info(f"Loading {nrows} rows from position {skiprows} of {total_rows} total")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True, skiprows=range(1, skiprows + 1), nrows=nrows)
        logger.info(f"Loaded {len(data)} rows with {data.shape[1]} columns")

        # Process features for prediction
        processor = DataProcessor()
        X, y = processor.prepare_ml_features(data, horizon=horizon)
        logger.info(f"Prepared features shape: {X.shape}")

        # Make predictions
        logger.info("Generating predictions...")
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            if y_proba.shape[1] == 2:  # Binary classification
                positive_probs = y_proba[:, 1]

                # Print statistics
                logger.info(
                    f"Prediction stats: min={positive_probs.min():.4f}, max={positive_probs.max():.4f}, mean={positive_probs.mean():.4f}")

                # Show distribution
                percentiles = np.percentile(positive_probs, [10, 25, 50, 75, 90])
                logger.info(f"Percentiles (10/25/50/75/90): {percentiles}")

                # Analyze thresholds
                for threshold in [0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]:
                    buy_signals = (positive_probs >= threshold).sum()
                    sell_signals = (positive_probs <= (1 - threshold)).sum()
                    logger.info(
                        f"Threshold {threshold}: {buy_signals} buys ({buy_signals / len(positive_probs) * 100:.1f}%), {sell_signals} sells ({sell_signals / len(positive_probs) * 100:.1f}%)")

                # Save distribution plot
                plt.figure(figsize=(10, 6))
                plt.hist(positive_probs, bins=50)
                plt.axvline(x=0.5, color='k', linestyle='--')
                plt.axvline(x=0.65, color='r', linestyle='-', label='Current threshold (0.65)')
                plt.axvline(x=0.35, color='r', linestyle='-')
                plt.axvline(x=0.55, color='g', linestyle=':', label='Potential threshold (0.55)')
                plt.axvline(x=0.45, color='g', linestyle=':')
                plt.title('Prediction Probability Distribution')
                plt.xlabel('Probability of Price Going Up')
                plt.ylabel('Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig('prediction_distribution_detailed.png')
                logger.info("Saved detailed prediction distribution plot")
            else:
                logger.warning(f"Unexpected probability shape: {y_proba.shape}")
        else:
            logger.warning("Model does not support probability predictions")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    model_path = "models/ensemble_H1_direction_12.joblib"
    analyze_predictions(model_path)