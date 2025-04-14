import pandas as pd
import numpy as np
from models.factory import ModelFactory
from utils.logger import setup_logger

logger = setup_logger("DiagnosticLogger")


def simple_diagnostic():
    """Step-by-step diagnostic to find where issues occur"""

    try:
        # Step 1: Load a small amount of test data
        logger.info("Step 1: Loading test data")
        data_path = "../data/historical/XAUUSD_H1_processed.csv"
        df = pd.read_csv(data_path, index_col=0, parse_dates=True, nrows=50)
        logger.info(f"Loaded {len(df)} rows with {df.shape[1]} columns")

        # Step 2: Basic preparation
        logger.info("Step 2: Basic data preparation")
        feature_cols = [col for col in df.columns if not col.startswith('target_')]
        target_cols = [col for col in df.columns if col.startswith('target_')]
        logger.info(f"Found {len(feature_cols)} feature columns")
        logger.info(f"Found {len(target_cols)} target columns: {target_cols}")

        # Step 3: Load model
        logger.info("Step 3: Loading model")
        model_path = "models/ensemble_H1_direction_12.joblib"
        model = ModelFactory.load_model(model_path)
        logger.info(f"Model loaded: {type(model).__name__}")

        # Step 4: Make basic prediction
        logger.info("Step 4: Basic prediction")
        X = df[feature_cols].iloc[:10]
        try:
            pred = model.predict(X)
            logger.info(f"Made {len(pred)} predictions: {pred}")
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                logger.info(f"Probability shape: {proba.shape}")
                logger.info(f"First few probabilities: {proba[:3]}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")

        # Step 5: Check threshold behavior
        if hasattr(model, 'predict_proba'):
            logger.info("Step 5: Testing thresholds")
            try:
                probas = model.predict_proba(X)[:, 1]  # Positive class probabilities
                logger.info(f"Probabilities range: {probas.min():.4f} to {probas.max():.4f}")

                for threshold in [0.7, 0.65, 0.6, 0.55, 0.5]:
                    buys = sum(probas >= threshold)
                    sells = sum(probas <= (1 - threshold))
                    logger.info(f"Threshold {threshold}: {buys} buys, {sells} sells")
            except Exception as e:
                logger.error(f"Error testing thresholds: {str(e)}")

        logger.info("Diagnostic complete")

    except Exception as e:
        logger.error(f"Error in diagnostic: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    simple_diagnostic()