import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from data.storage import DataStorage
from data.processor import DataProcessor
from utils.logger import setup_logger

logger = setup_logger("ModelInspector")


def load_model(model_path=None):
    """Load a trained model for inspection."""
    storage = DataStorage()

    if model_path is None:
        # Find the latest model
        models_dir = os.path.join(project_root, "models")
        if not os.path.exists(models_dir):
            logger.error("Models directory not found")
            return None, None

        model_files = [f for f in os.listdir(models_dir) if
                       f.endswith('.joblib') and not f.endswith('_random_forest.joblib') and not f.endswith(
                           '_xgboost.joblib')]

        if not model_files:
            logger.error("No model files found")
            return None, None

        # Sort by creation time (most recent first)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
        model_path = os.path.join(models_dir, model_files[0])
        print(f"Using latest model: {model_files[0]}")

    try:
        # Load model
        result = storage.load_model(model_path)

        if isinstance(result, tuple) and len(result) == 2:
            model, metadata = result
            logger.info(f"Model loaded successfully with metadata")
        else:
            model = result
            metadata = None
            logger.info(f"Model loaded successfully without metadata")

        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None


def load_test_data(timeframe='H1', horizon=1):
    """Load the most recent processed data for testing."""
    storage = DataStorage()
    processor = DataProcessor()

    # Get the latest processed data
    latest_data = storage.find_latest_processed_data()

    if not latest_data or timeframe not in latest_data:
        logger.error(f"No processed data found for {timeframe}")
        return None, None

    # Load data
    data_path = latest_data[timeframe]
    df = processor.load_data({timeframe: data_path})[timeframe]
    logger.info(f"Loaded {len(df)} rows of {timeframe} data from {data_path}")

    # Prepare features and target
    try:
        X, y = processor.prepare_ml_features(df, horizon=horizon)
        logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error preparing test data: {str(e)}")
        return None, None


def inspect_model_structure(model, metadata=None):
    """Inspect the structure of the model."""
    print("\n===== MODEL STRUCTURE =====")

    # Try to determine the model type
    model_type = type(model).__name__
    print(f"Model type: {model_type}")

    # Extract model parameters if available
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print("\nModel parameters:")
        for param, value in params.items():
            if not param.startswith('_'):  # Skip private parameters
                print(f"  {param}: {value}")

    # Check if it's an ensemble model
    if hasattr(model, 'estimators_'):
        estimators = model.estimators_

        if isinstance(estimators, list):
            print(f"\nEnsemble with {len(estimators)} estimators:")
            for i, estimator in enumerate(estimators):
                print(f"  Estimator {i + 1}: {type(estimator).__name__}")

        # For VotingClassifier
        if hasattr(model, 'estimators'):
            print("\nVoting Classifier with estimators:")
            for name, estimator in model.estimators:
                print(f"  {name}: {type(estimator).__name__}")

            if hasattr(model, 'weights') and model.weights is not None:
                print(f"  Weights: {model.weights}")

    # For tree-based models, show tree structure summary
    if hasattr(model, 'n_estimators') and hasattr(model, 'estimators_'):
        print(f"\nTree structure:")

        # For random forest or similar
        if hasattr(model.estimators_[0], 'tree_'):
            trees = [estimator.tree_ for estimator in model.estimators_]
            depths = [tree.max_depth for tree in trees if hasattr(tree, 'max_depth')]

            if depths:
                print(f"  Average tree depth: {np.mean(depths):.2f}")
                print(f"  Min/Max tree depth: {np.min(depths)}/{np.max(depths)}")

            node_counts = [tree.node_count for tree in trees if hasattr(tree, 'node_count')]
            if node_counts:
                print(f"  Average nodes per tree: {np.mean(node_counts):.2f}")

    # For XGBoost models
    if model_type == 'XGBClassifier':
        print("\nXGBoost model:")
        print(f"  Number of boosting rounds: {model.n_estimators}")
        if hasattr(model, 'best_iteration'):
            print(f"  Best iteration: {model.best_iteration}")
        if hasattr(model, 'feature_importances_'):
            print(f"  Number of features used: {len(model.feature_importances_)}")

    # From metadata
    if metadata:
        if 'config' in metadata and 'model' in metadata['config']:
            model_config = metadata['config']['model']
            print("\nModel configuration from metadata:")
            for key, value in model_config.items():
                print(f"  {key}: {value}")

        # Training details
        if 'trained_at' in metadata:
            print(f"\nTrained at: {metadata['trained_at']}")
        if 'timeframe' in metadata:
            print(f"Timeframe: {metadata['timeframe']}")
        if 'prediction_target' in metadata:
            print(f"Prediction target: {metadata['prediction_target']}")
        if 'prediction_horizon' in metadata:
            print(f"Prediction horizon: {metadata['prediction_horizon']}")


def test_model_on_data(model, X, y):
    """Test the model on the given data."""
    if model is None or X is None or y is None:
        logger.error("Model or test data not available")
        return

    print("\n===== MODEL EVALUATION =====")

    # Make predictions
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = (y_pred == y).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save plot
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to: {os.path.join(plots_dir, 'confusion_matrix.png')}")
    plt.close()

    # Check prediction probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)

        # Plot probability distribution
        plt.figure(figsize=(10, 6))

        if probas.shape[1] >= 2:
            # For binary classification
            sns.histplot(probas[:, 1], bins=20, kde=True)
            plt.axvline(x=0.5, color='r', linestyle='--', label='0.5 Threshold')
            plt.axvline(x=0.65, color='g', linestyle='--', label='0.65 Threshold')
            plt.title('Distribution of Prediction Probabilities')
            plt.xlabel('Probability of Class 1 (Up)')
            plt.ylabel('Frequency')
            plt.legend()

            # Save plot
            plt.savefig(os.path.join(plots_dir, "probability_distribution.png"))
            print(f"Probability distribution saved to: {os.path.join(plots_dir, 'probability_distribution.png')}")
            plt.close()

            # Analyze probability thresholds
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            print("\nImpact of probability thresholds:")
            print(f"  {'Threshold':>10} {'Predictions':>12} {'% of Data':>10}")
            print("  " + "-" * 35)

            for threshold in thresholds:
                pred_count = (probas[:, 1] >= threshold).sum()
                pct_of_data = pred_count / len(probas) * 100
                print(f"  {threshold:>10.2f} {pred_count:>12} {pct_of_data:>10.2f}%")


def analyze_feature_importance(model, X, metadata=None):
    """Analyze feature importance."""
    if model is None or X is None:
        logger.error("Model or feature data not available")
        return

    print("\n===== FEATURE IMPORTANCE =====")

    feature_names = X.columns.tolist()

    # Check if metadata has feature list
    if metadata and 'features' in metadata:
        feature_names = metadata['features']
        if len(feature_names) != X.shape[1]:
            logger.warning(f"Mismatch between feature count in metadata ({len(feature_names)}) and data ({X.shape[1]})")
            # Use the first n features from metadata
            feature_names = feature_names[:X.shape[1]] if len(feature_names) > X.shape[1] else feature_names

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    elif hasattr(model, 'estimators_') and hasattr(model, 'weights'):
        # For VotingClassifier - use feature importance from first base estimator that has it
        for estimator in model.estimators:
            if hasattr(estimator[1], 'feature_importances_'):
                importances = estimator[1].feature_importances_
                print(f"Using feature importances from {estimator[0]}")
                break
        else:
            logger.warning("No feature importances found in any estimator")
            return
    else:
        logger.warning("Model doesn't have accessible feature importances")
        return

    # Create DataFrame for visualization
    if len(importances) == len(feature_names):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Print feature importance
        print("\nTop features by importance:")
        for i, (feature, importance) in enumerate(zip(importance_df['Feature'][:15], importance_df['Importance'][:15])):
            print(f"{i + 1:2d}. {feature:<30}: {importance:.6f}")

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()  # Display the highest importance at the top

        # Save plot
        plots_dir = os.path.join(project_root, "analysis")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
        print(f"Feature importance plot saved to: {os.path.join(plots_dir, 'feature_importance.png')}")
        plt.close()
    else:
        logger.warning(
            f"Feature importance length ({len(importances)}) doesn't match feature count ({len(feature_names)})")


def check_signal_generation(model, X, horizon=1):
    """Check how the model generates trading signals."""
    if model is None or X is None:
        logger.error("Model or feature data not available")
        return

    print("\n===== SIGNAL GENERATION =====")

    # Get the most recent data points
    recent_X = X.tail(50)

    # Make predictions
    predictions = model.predict(recent_X)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(recent_X)

        # For binary classification
        if probabilities.shape[1] >= 2:
            up_probabilities = probabilities[:, 1]

            # Create result DataFrame
            results = pd.DataFrame({
                'date': recent_X.index,
                'prediction': predictions,
                'probability': up_probabilities
            })

            # Check high-confidence signals
            for threshold in [0.5, 0.6, 0.65, 0.7, 0.75]:
                buy_signals = ((results['prediction'] == 1) & (results['probability'] >= threshold)).sum()
                sell_signals = ((results['prediction'] == 0) & (results['probability'] >= threshold)).sum()
                total_signals = buy_signals + sell_signals

                print(f"\nWith {threshold} confidence threshold:")
                print(f"  Buy signals: {buy_signals}")
                print(f"  Sell signals: {sell_signals}")
                print(f"  Total signals: {total_signals} ({total_signals / len(results) * 100:.2f}%)")

            # Print the most recent signals
            print("\nMost recent 10 predictions:")
            most_recent = results.tail(10)
            for idx, row in most_recent.iterrows():
                signal = "BUY" if row['prediction'] == 1 else "SELL"
                print(f"  {idx}: {signal} (confidence: {row['probability']:.4f})")

            # Suggest optimal threshold
            avg_prob = results['probability'].mean()
            std_prob = results['probability'].std()
            suggested_threshold = min(0.5 + 1.5 * std_prob, 0.75)
            suggested_threshold = max(suggested_threshold, 0.55)  # Lower bound
            suggested_threshold = round(suggested_threshold * 20) / 20  # Round to nearest 0.05

            print(f"\nSignal statistics:")
            print(f"  Average probability: {avg_prob:.4f}")
            print(f"  Probability std dev: {std_prob:.4f}")
            print(f"  Suggested threshold: {suggested_threshold:.2f}")

            # Check suggested threshold impact
            high_conf = ((results['prediction'] == 1) & (results['probability'] >= suggested_threshold)).sum() + \
                        ((results['prediction'] == 0) & (results['probability'] >= suggested_threshold)).sum()
            print(
                f"  With {suggested_threshold} threshold: {high_conf} signals ({high_conf / len(results) * 100:.2f}%)")
    else:
        # If no probabilities available, just count predictions
        up_count = (predictions == 1).sum()
        down_count = (predictions == 0).sum()

        print(f"\nPrediction distribution (last 50 periods):")
        print(f"  Up signals: {up_count} ({up_count / len(predictions) * 100:.2f}%)")
        print(f"  Down signals: {down_count} ({down_count / len(predictions) * 100:.2f}%)")


def main():
    """Main function."""
    print("===== GOLD TRADING MODEL INSPECTOR =====")
    print("This tool inspects and tests a trained model.")

    # Check for model path from command line
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\nUsing specified model: {model_path}")

    # Load model
    model, metadata = load_model(model_path)

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Get prediction horizon from metadata
    horizon = 1  # Default to 1-period prediction
    timeframe = 'H1'  # Default to H1 timeframe

    if metadata:
        if 'prediction_horizon' in metadata:
            horizon = metadata['prediction_horizon']
        if 'timeframe' in metadata:
            timeframe = metadata['timeframe']

    # Load test data
    X, y = load_test_data(timeframe, horizon)

    # Run analyses
    inspect_model_structure(model, metadata)

    if X is not None and y is not None:
        test_model_on_data(model, X, y)
        analyze_feature_importance(model, X, metadata)
        check_signal_generation(model, X, horizon)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()