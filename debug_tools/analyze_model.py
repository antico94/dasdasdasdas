import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
from data.storage import DataStorage
from data.processor import DataProcessor
from utils.logger import setup_logger
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


logger = setup_logger("ModelAnalyzer")


def load_model_and_metadata(model_path=None):
    """Load the model and its metadata."""
    import os  # ensure os is imported here
    storage = DataStorage()

    # If a model_path is provided and it's a directory, search for a model file inside it
    if model_path and os.path.isdir(model_path):
        model_files = [
            f for f in os.listdir(model_path)
            if f.endswith('.joblib') and
            not f.endswith('_random_forest.joblib') and
            not f.endswith('_xgboost.joblib')
        ]
        if not model_files:
            logger.error("No model files found in directory: " + model_path)
            return None, None
        # Sort by creation time (most recent first)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_path, x)), reverse=True)
        model_path = os.path.join(model_path, model_files[0])
        logger.info(f"Using latest model: {model_files[0]}")
    elif model_path is None:
        # If no model_path is provided, set a default (adjust as needed)
        # For example, if your default location is the "data/models" folder:
        models_dir = os.path.join(project_root, "data", "models")
        model_files = [
            f for f in os.listdir(models_dir)
            if f.endswith('.joblib') and
            not f.endswith('_random_forest.joblib') and
            not f.endswith('_xgboost.joblib')
        ]
        if not model_files:
            logger.error("No model files found in " + models_dir)
            return None, None
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
        model_path = os.path.join(models_dir, model_files[0])
        logger.info(f"Using latest model: {model_files[0]}")

    try:
        # Confirm the model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None

        # Load model and metadata
        result = storage.load_model(model_path)

        if isinstance(result, tuple) and len(result) == 2:
            model, metadata = result
            logger.info("Model loaded successfully with metadata")
            return model, metadata
        else:
            model = result
            logger.info("Model loaded successfully without metadata")
            return model, None

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None



def analyze_model_performance(model, metadata=None):
    """Analyze model performance metrics."""
    if metadata is None or 'metrics' not in metadata:
        logger.warning("No metrics found in metadata")
        return

    metrics = metadata['metrics']
    print("\n===== MODEL PERFORMANCE METRICS =====")

    if 'train' in metrics:
        print("\nTraining Performance:")
        train_metrics = metrics['train']
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

    if 'test' in metrics:
        print("\nTest Performance:")
        test_metrics = metrics['test']
        for k, v in test_metrics.items():
            if k != 'classification_report' and isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

        if 'classification_report' in test_metrics:
            print("\nDetailed Classification Report:")
            report = test_metrics['classification_report']

            # Format the report as a table
            headers = ['precision', 'recall', 'f1-score', 'support']
            print(f"{'':>10} {headers[0]:>10} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10}")
            print("-" * 55)

            for cls, metrics in report.items():
                if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                    if isinstance(metrics, dict):
                        print(
                            f"{cls:>10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1-score']:>10.4f} {metrics['support']:>10}")

            print("-" * 55)
            if 'accuracy' in report:
                print(f"{'accuracy':>10} {report['accuracy']:>10.4f} {' ':>10} {' ':>10} {' ':>10}")


def analyze_feature_importance(model, metadata=None):
    """Analyze feature importance."""
    if metadata is None or 'features' not in metadata:
        logger.warning("No feature information found in metadata")
        return

    features = metadata['features']

    try:
        # Get feature importance depending on model type
        if hasattr(model, 'feature_importances_'):  # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):  # For linear models
            importances = np.abs(model.coef_[0])
            # For ensemble, try to get feature importance from one of the constituent estimators
        elif hasattr(model, 'estimators_'):
            if hasattr(model, 'named_estimators_'):
                # Iterate through the dictionary of estimators
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        print(f"Using feature importances from {name}")
                        break
                else:
                    logger.warning("No estimator in named_estimators_ has feature_importances_")
                    return
            else:
                # Fallback: iterate through the list of estimators
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        print("Using feature importances from one of the estimators")
                        break
                else:
                    logger.warning("No estimator in estimators_ has feature_importances_")
                    return

            # Make sure we have the right number of features
            if len(importances) != len(features):
                logger.warning(
                    f"Mismatch between feature importances ({len(importances)}) and feature names ({len(features)})")
                return

        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\n===== FEATURE IMPORTANCE =====")
        for i, (feature, importance) in enumerate(
                zip(feature_importance_df['Feature'], feature_importance_df['Importance'])):
            print(f"{i + 1:2d}. {feature:30s}: {importance:.4f}")

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(project_root, "analysis")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(plots_dir, f"feature_importance_{timestamp}.png"))
        print(f"\nFeature importance plot saved to: {os.path.join(plots_dir, f'feature_importance_{timestamp}.png')}")
        plt.close()

    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")


def analyze_model_config(metadata=None):
    """Analyze model configuration."""
    if metadata is None or 'config' not in metadata:
        logger.warning("No configuration found in metadata")
        return

    config = metadata['config']

    print("\n===== MODEL CONFIGURATION =====")
    if 'model' in config:
        model_config = config['model']
        print("\nModel settings:")
        for k, v in model_config.items():
            print(f"  {k}: {v}")

    # Print training configuration
    print("\nTraining details:")
    if 'trained_at' in metadata:
        print(f"  Trained at: {metadata['trained_at']}")
    if 'data_shape' in metadata:
        print(f"  Data shape: {metadata['data_shape']}")
    if 'timeframe' in metadata:
        print(f"  Timeframe: {metadata['timeframe']}")
    if 'prediction_target' in metadata:
        print(f"  Prediction target: {metadata['prediction_target']}")
    if 'prediction_horizon' in metadata:
        print(f"  Prediction horizon: {metadata['prediction_horizon']}")


def validate_model_on_recent_data(model, metadata=None):
    """Validate model on the most recent data."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from data.storage import DataStorage
    from data.processor import DataProcessor
    from utils.logger import setup_logger

    logger = setup_logger("ModelValidator")

    # Initialize storage and processor
    storage = DataStorage()
    processor = DataProcessor()

    # Get the latest processed data files
    latest_data = storage.find_latest_processed_data()

    if not latest_data:
        logger.warning("No processed data found")
        return

    # Determine the timeframe from metadata or default to 'H1'
    timeframe = metadata.get("timeframe", "H1") if metadata else "H1"
    if timeframe not in latest_data:
        logger.warning(f"No data found for timeframe {timeframe}")
        return

    # Load the data using the processor
    data_dict = processor.load_data({timeframe: latest_data[timeframe]})
    df = data_dict[timeframe]

    # Get prediction horizon from metadata (default to 1)
    horizon = metadata.get("prediction_horizon", 1) if metadata else 1

    # Prepare features and target
    X, y = processor.prepare_ml_features(df, horizon=horizon)
    logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")

    # --- FIX: Subset the features to match those used during training ---
    expected_features = metadata.get("features", [])
    if expected_features:
        missing_features = [f for f in expected_features if f not in X.columns]
        if missing_features:
            logger.error(f"Missing expected features: {missing_features}. Cannot proceed with prediction.")
            return
        X = X[expected_features]
        logger.info(f"Subsetting features to expected list: {expected_features}")
    # ---------------------------------------------------------------------

    # Use the last 20% of the data for recent validation
    test_size = int(len(X) * 0.2)
    X_recent = X.iloc[-test_size:]
    y_recent = y.iloc[-test_size:]

    # Predict using the model
    try:
        y_pred = model.predict(X_recent)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return

    print("\n===== RECENT DATA VALIDATION =====")
    print(f"Using the most recent {test_size} data points ({timeframe} timeframe)")

    # Calculate performance metrics
    accuracy = accuracy_score(y_recent, y_pred)
    precision = precision_score(y_recent, y_pred, zero_division=0)
    recall = recall_score(y_recent, y_pred, zero_division=0)
    f1 = f1_score(y_recent, y_pred, zero_division=0)

    print("\nPerformance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Generate confusion matrix plot
    cm = confusion_matrix(y_recent, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on Recent Data')

    # Save plot to analysis folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nConfusion matrix saved to: {plot_path}")

    # Print class distribution of ground truth and predictions
    print("\nClass Distribution (Ground Truth):")
    class_counts = y_recent.value_counts()
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({count / len(y_recent) * 100:.2f}%)")

    print("\nSignal Distribution (Predictions):")
    signal_counts = pd.Series(y_pred).value_counts()
    for cls, count in signal_counts.items():
        print(f"  Signal {cls}: {count} predictions ({count / len(y_pred) * 100:.2f}%)")



def main():
    """Main function."""
    print("===== GOLD TRADING MODEL ANALYZER =====")
    print("This tool analyzes a trained model and its performance.")

    # Use the already computed project_root (which points to the true project root)
    model_path = os.path.join(project_root, "data", "models")

    # Load model and metadata
    model, metadata = load_model_and_metadata()

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Analyze model
    analyze_model_config(metadata)
    analyze_model_performance(model, metadata)
    analyze_feature_importance(model, metadata)
    validate_model_on_recent_data(model, metadata)

    print("\nAnalysis complete!")



if __name__ == "__main__":
    main()