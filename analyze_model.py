import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import yaml

# Add project root to path to import project modules
sys.path.append('.')

from data.storage import DataStorage
from data.processor import DataProcessor
from utils.logger import setup_logger


def analyze_model(model_path, timeframe='H1', confidence_threshold=0.65):
    """
    Analyze a trained model's predictions on test data to diagnose issues.

    Args:
        model_path: Path to the trained model file
        timeframe: Timeframe to use for testing (e.g., 'H1')
        confidence_threshold: Confidence threshold for trade signals
    """
    # Set up logging
    logger = setup_logger("ModelAnalyzer")
    logger.info(f"Analyzing model: {model_path}, timeframe: {timeframe}")

    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load the model
    logger.info(f"Loading model from {model_path}")
    try:
        from models.factory import ModelFactory
        model = ModelFactory.load_model(model_path)
        logger.info(f"Model type: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # Get model features if available
    if hasattr(model, 'get_feature_names'):
        model_features = model.get_feature_names()
        logger.info(f"Model uses {len(model_features)} features")
    else:
        model_features = None
        logger.warning("Model does not provide feature names")

    # Load test data
    storage = DataStorage()
    processed_data_paths = storage.find_latest_processed_data()

    if timeframe not in processed_data_paths:
        logger.error(f"No processed data found for timeframe {timeframe}")
        return

    logger.info(f"Loading data from {processed_data_paths[timeframe]}")
    data = pd.read_csv(processed_data_paths[timeframe], index_col=0, parse_dates=True)
    logger.info(f"Data shape: {data.shape}")

    # Extract the prediction horizon from the model filename
    try:
        horizon = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        horizon = 12  # Default
    logger.info(f"Using prediction horizon: {horizon}")

    # Split into train/test (use last 20% for testing)
    test_size = int(len(data) * 0.2)
    test_data = data.iloc[-test_size:].copy()
    logger.info(f"Test data shape: {test_data.shape}")

    # Prepare features and target
    processor = DataProcessor()
    X_test, y_test = processor.prepare_ml_features(test_data, horizon=horizon)
    logger.info(f"Prepared features shape: {X_test.shape}, target shape: {y_test.shape}")

    # Filter features if model expects specific ones
    if model_features is not None:
        common_features = [f for f in model_features if f in X_test.columns]
        missing_features = [f for f in model_features if f not in X_test.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features expected by model")
            logger.warning(f"First few missing: {missing_features[:5]}")

        if common_features:
            X_test = X_test[common_features]
            logger.info(f"Using {len(common_features)} common features")
        else:
            logger.error("No common features between model and test data!")
            return

    # Generate predictions
    logger.info("Generating predictions...")

    is_classifier = hasattr(model, 'predict_proba')

    if is_classifier:
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Analyze prediction distribution
            if y_pred_proba.shape[1] == 2:  # Binary classification
                positive_probs = y_pred_proba[:, 1]

                # Log probability distribution
                prob_bins = np.linspace(0, 1, 11)
                hist, _ = np.histogram(positive_probs, bins=prob_bins)
                bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

                logger.info("Prediction probability distribution:")
                for i, (center, count) in enumerate(zip(bin_centers, hist)):
                    logger.info(f"  {center:.1f}: {count} predictions ({count / len(positive_probs) * 100:.1f}%)")

                logger.info(f"Min probability: {positive_probs.min():.4f}")
                logger.info(f"Max probability: {positive_probs.max():.4f}")
                logger.info(f"Mean probability: {positive_probs.mean():.4f}")

                # Check threshold crossing
                above_threshold = (positive_probs >= confidence_threshold).sum()
                below_threshold = (positive_probs <= (1 - confidence_threshold)).sum()

                logger.info(f"Confidence threshold: {confidence_threshold}")
                logger.info(
                    f"Predictions above buy threshold: {above_threshold} ({above_threshold / len(positive_probs) * 100:.1f}%)")
                logger.info(
                    f"Predictions below sell threshold: {below_threshold} ({below_threshold / len(positive_probs) * 100:.1f}%)")

                # Analyze prediction vs actual
                logger.info(f"Prediction distribution: 0s={sum(y_pred == 0)}, 1s={sum(y_pred == 1)}")
                logger.info(f"Actual distribution: 0s={sum(y_test == 0)}, 1s={sum(y_test == 1)}")

                # Create results dataframe
                results = pd.DataFrame(index=X_test.index)
                results['actual'] = y_test.values
                results['predicted'] = y_pred
                results['probability'] = positive_probs
                results['signal'] = 0
                results.loc[results['probability'] >= confidence_threshold, 'signal'] = 1  # Buy
                results.loc[results['probability'] <= (1 - confidence_threshold), 'signal'] = -1  # Sell

                signal_counts = results['signal'].value_counts()
                logger.info(f"Signal distribution: {signal_counts.to_dict()}")

                # Plot probability distribution and threshold
                plt.figure(figsize=(10, 6))
                plt.hist(positive_probs, bins=20, alpha=0.7)
                plt.axvline(x=confidence_threshold, color='r', linestyle='--',
                            label=f'Buy Threshold ({confidence_threshold})')
                plt.axvline(x=1 - confidence_threshold, color='g', linestyle='--',
                            label=f'Sell Threshold ({1 - confidence_threshold})')
                plt.title('Prediction Probability Distribution')
                plt.xlabel('Probability of Class 1 (Up)')
                plt.ylabel('Count')
                plt.legend()
                plt.savefig('prediction_distribution.png')
                logger.info("Saved probability distribution plot to prediction_distribution.png")

                # Plot predictions over time
                plt.figure(figsize=(12, 8))
                plt.plot(results.index, results['probability'], label='Prediction Probability')
                plt.axhline(y=confidence_threshold, color='r', linestyle='--',
                            label=f'Buy Threshold ({confidence_threshold})')
                plt.axhline(y=1 - confidence_threshold, color='g', linestyle='--',
                            label=f'Sell Threshold ({1 - confidence_threshold})')
                plt.title('Prediction Probabilities Over Time')
                plt.xlabel('Date')
                plt.ylabel('Probability')
                plt.legend()
                plt.savefig('prediction_time_series.png')
                logger.info("Saved time series plot to prediction_time_series.png")

                # Test alternate thresholds
                logger.info("Testing alternate thresholds:")
                for threshold in [0.55, 0.6, 0.65, 0.7, 0.75]:
                    buy_signals = (positive_probs >= threshold).sum()
                    sell_signals = (positive_probs <= (1 - threshold)).sum()
                    total_signals = buy_signals + sell_signals
                    logger.info(
                        f"  Threshold {threshold}: {buy_signals} buys, {sell_signals} sells, {total_signals} total ({total_signals / len(positive_probs) * 100:.1f}%)")

                # Save results for further analysis
                results_path = os.path.join('analysis', 'model_analysis_results.csv')
                os.makedirs('analysis', exist_ok=True)
                results.to_csv(results_path)
                logger.info(f"Saved detailed results to {results_path}")

            else:
                logger.warning(f"Unexpected prediction shape: {y_pred_proba.shape}")
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
    else:
        # For regression models
        try:
            y_pred = model.predict(X_test)
            logger.info(
                f"Prediction statistics: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")

            # Create results dataframe
            results = pd.DataFrame(index=X_test.index)
            results['actual'] = y_test.values
            results['predicted'] = y_pred

            # Plot predictions vs actual
            plt.figure(figsize=(12, 8))
            plt.plot(results.index, results['actual'], label='Actual')
            plt.plot(results.index, results['predicted'], label='Predicted')
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig('regression_predictions.png')
            logger.info("Saved regression plot to regression_predictions.png")

            results_path = os.path.join('analysis', 'model_analysis_results.csv')
            os.makedirs('analysis', exist_ok=True)
            results.to_csv(results_path)
            logger.info(f"Saved detailed results to {results_path}")
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")

    logger.info("Analysis complete")


if __name__ == "__main__":
    # Default parameters
    model_path = "models/ensemble_H1_direction_12.joblib"
    timeframe = "H1"
    confidence_threshold = 0.65

    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        timeframe = sys.argv[2]
    if len(sys.argv) > 3:
        confidence_threshold = float(sys.argv[3])

    analyze_model(model_path, timeframe, confidence_threshold)