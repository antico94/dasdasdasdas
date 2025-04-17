import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import joblib
import sys

# Fix matplotlib backend to avoid GUI dependencies
matplotlib.use('Agg')

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from data.storage import DataStorage
from data.processor import DataProcessor
from models.factory import ModelFactory
from utils.logger import setup_logger

logger = setup_logger("PredictionAnalyzer")


def load_model_and_data(model_path=None, timeframe='H1', date_range=None, use_test_data=True):
    """
    Load a trained model and data for the specified timeframe.

    Parameters:
        model_path: Path to the model file
        timeframe: Timeframe to analyze (e.g., 'H1')
        date_range: Optional date range tuple (start_date, end_date)
        use_test_data: Whether to use test data (True) or validation data (False)

    Returns:
        Tuple of (model, model_info, data_info)
    """
    storage = DataStorage()
    processor = DataProcessor()

    # Find the latest model if not specified
    if model_path is None:
        models_dir = os.path.join(project_root, "data_output", "trained_models")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if
                           f.endswith('.joblib') and not f.endswith('_random_forest.joblib')
                           and not f.endswith('_xgboost.joblib')
                           and not f.endswith('_metadata.pkl')]
            if model_files:
                model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
                logger.info(f"Using latest model: {model_files[0]}")
            else:
                logger.error("No model files found")
                return None, None, None
        else:
            logger.error("Models directory not found")
            return None, None, None

    # Load model using ModelFactory
    try:
        logger.info(f"Loading model from {model_path}")
        model = ModelFactory.load_model(model_path)

        model_info = {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'creation_date': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        }

        # Check if model is a tuple with metadata
        if isinstance(model, tuple) and len(model) == 2:
            model, metadata = model
            model_info['metadata'] = metadata
            logger.info(f"Model loaded successfully with metadata")
        else:
            # Try to load metadata from separate file
            metadata_path = model_path.replace(".joblib", "_metadata.pkl")
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    model_info['metadata'] = metadata
                    logger.info(f"Loaded model metadata from separate file")
                except Exception as e:
                    logger.warning(f"Error loading metadata file: {str(e)}")
                    metadata = None
                    model_info['metadata'] = {}
            else:
                metadata = None
                model_info['metadata'] = {}
                logger.info(f"No metadata found for model")

        # Get the timeframe from metadata if available
        if metadata and 'timeframe' in metadata:
            timeframe = metadata['timeframe']
            logger.info(f"Using timeframe from metadata: {timeframe}")
            model_info['timeframe'] = timeframe
        else:
            model_info['timeframe'] = timeframe

        # Get prediction horizon from metadata if available
        horizon = 1  # Default to 1-period prediction
        if metadata and 'prediction_horizon' in metadata:
            horizon = metadata['prediction_horizon']
            logger.info(f"Using prediction horizon from metadata: {horizon}")
            model_info['prediction_horizon'] = horizon
        else:
            model_info['prediction_horizon'] = horizon

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

    # Load test data from the test split
    split_paths = storage.find_latest_split_data()

    # Decide whether to use test or validation data
    split_type = "test" if use_test_data else "validation"

    if split_type not in split_paths or timeframe not in split_paths[split_type]:
        logger.error(f"No {split_type} data found for timeframe {timeframe}")
        return model, model_info, None

    logger.info(f"Loading {split_type} data from {split_paths[split_type][timeframe]}")
    try:
        data_dict = processor.load_data({timeframe: split_paths[split_type][timeframe]})
        test_data = data_dict[timeframe]
        logger.info(f"Loaded {split_type} data shape: {test_data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return model, model_info, None

    # Filter by date range if specified
    if date_range is not None:
        start_date, end_date = date_range
        logger.info(f"Filtering data from {start_date} to {end_date}")
        test_data = test_data.loc[start_date:end_date]
        logger.info(f"Filtered data contains {len(test_data)} rows")

        if len(test_data) == 0:
            logger.error(f"No data available for the specified date range")
            return model, model_info, None

    # Add data info to model_info
    model_info['data_file'] = split_paths[split_type][timeframe]
    model_info['data_rows'] = len(test_data)
    model_info['data_period'] = {
        'start': test_data.index.min().strftime('%Y-%m-%d %H:%M:%S') if len(test_data) > 0 else "N/A",
        'end': test_data.index.max().strftime('%Y-%m-%d %H:%M:%S') if len(test_data) > 0 else "N/A"
    }

    # Add date range to model_info if specified
    if date_range is not None:
        model_info['selected_date_range'] = date_range

    return model, model_info, (test_data, horizon)


def generate_predictions(model, df, horizon=1):
    """Generate predictions using the model on the given data."""
    processor = DataProcessor()
    logger = processor.logger  # Use the same logger from DataProcessor

    try:
        # Prepare features
        X, y = processor.prepare_ml_features(df, horizon=horizon)
        logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")

        # Debug: Print class distribution
        actual_class_dist = y.value_counts(normalize=True)
        logger.info(f"Actual class distribution in test data: {actual_class_dist}")
        logger.info(f"Class 1 (UP) proportion: {actual_class_dist.get(1, 0):.4f}")
        logger.info(f"Class 0 (DOWN) proportion: {actual_class_dist.get(0, 0):.4f}")

        # Get expected features from model metadata
        expected_features = []

        # Check if model has metadata attribute
        if hasattr(model, "metadata") and model.metadata:
            expected_features = model.metadata.get("features", [])

        # If model has a get_feature_importance method, get features from there
        elif hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            expected_features = list(feature_importance.keys())

        if expected_features:
            # Check for missing features
            missing_features = [feat for feat in expected_features if feat not in X.columns]

            if missing_features:
                logger.warning(f"Adding missing features with default values: {missing_features}")

                # Add missing features with default values
                for feature in missing_features:
                    X[feature] = 0.0  # Use 0.0 as default value

            # Ensure features are in the right order
            X = X[expected_features]
            logger.info(f"Using {len(expected_features)} features expected by the model")
        else:
            logger.warning("No expected features found in model metadata. Using all available features.")

        # Generate predictions
        logger.info("Generating predictions...")
        y_pred = model.predict(X)

        # Debug: Print prediction distribution
        pred_class_dist = pd.Series(y_pred).value_counts(normalize=True)
        logger.info(f"Prediction distribution: {pred_class_dist}")
        logger.info(f"Class 1 (UP) predictions: {pred_class_dist.get(1, 0):.4f}")
        logger.info(f"Class 0 (DOWN) predictions: {pred_class_dist.get(0, 0):.4f}")

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)

                # Debug: Print probability distribution
                logger.info(f"UP probability stats: {pd.Series(probas[:, 1]).describe()}")
                logger.info(f"DOWN probability stats: {pd.Series(probas[:, 0]).describe()}")

                # Add predictions to dataframe
                predictions_df = pd.DataFrame(index=X.index)
                predictions_df['actual'] = y
                predictions_df['predicted'] = y_pred
                if probas is not None and probas.shape[1] >= 2:
                    predictions_df['probability_up'] = probas[:, 1]
                    predictions_df['probability_down'] = probas[:, 0]
                    predictions_df['confidence'] = np.max(probas, axis=1)
                return predictions_df
            except Exception as e:
                logger.error(f"Error generating prediction probabilities: {str(e)}")
                # If predict_proba fails, create dataframe without probabilities
                predictions_df = pd.DataFrame(index=X.index)
                predictions_df['actual'] = y
                predictions_df['predicted'] = y_pred
                return predictions_df
        else:
            # Add predictions to dataframe without probabilities
            predictions_df = pd.DataFrame(index=X.index)
            predictions_df['actual'] = y
            predictions_df['predicted'] = y_pred
            return predictions_df

    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def analyze_predictions_over_time(predictions_df, df):
    """Analyze how predictions change over time."""
    if predictions_df is None or len(predictions_df) == 0:
        logger.error("No predictions to analyze")
        return None

    results = {}

    # Merge with price data
    predictions_df = predictions_df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            predictions_df[col] = df.loc[predictions_df.index, col]

    # Calculate daily accuracy (fix deprecated warning)
    predictions_df['date'] = predictions_df.index.date
    daily_accuracy = predictions_df.groupby('date', observed=True).apply(
        lambda x: (x['actual'] == x['predicted']).mean() if len(x) > 0 else np.nan
    ).dropna()

    results['daily_accuracy'] = daily_accuracy.to_dict()

    # Plot daily accuracy
    plt.figure(figsize=(12, 6))
    daily_accuracy.plot(kind='line', marker='o')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.title('Prediction Accuracy by Day')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Save plot
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_accuracy_path = os.path.join(plots_dir, f"daily_accuracy_{timestamp}.png")
    plt.savefig(daily_accuracy_path)
    results['daily_accuracy_plot'] = daily_accuracy_path
    plt.close()

    # Analyze prediction distribution over time
    if 'probability_up' in predictions_df.columns:
        # Plot prediction probabilities over time with price
        plt.figure(figsize=(14, 10))

        # Create a subplot for price
        ax1 = plt.subplot(211)
        ax1.plot(predictions_df.index, predictions_df['close'], 'k-', label='Close Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Gold Price and Prediction Probabilities')
        ax1.grid(True, alpha=0.3)

        # Create a subplot for probabilities
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(predictions_df.index, predictions_df['probability_up'], 'g-', label='Prob. Up')
        ax2.plot(predictions_df.index, predictions_df['probability_down'], 'r-', label='Prob. Down')
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        prob_plot_path = os.path.join(plots_dir, f"prediction_probabilities_{timestamp}.png")
        plt.savefig(prob_plot_path)
        results['probability_plot'] = prob_plot_path
        plt.close()

    return results


def analyze_prediction_accuracy(predictions_df):
    """Analyze prediction accuracy and confidence."""
    if predictions_df is None or len(predictions_df) == 0:
        logger.error("No predictions to analyze")
        return None

    results = {}

    # Overall accuracy
    overall_accuracy = (predictions_df['actual'] == predictions_df['predicted']).mean()
    results['overall_accuracy'] = overall_accuracy

    # Class-specific accuracy
    class_accuracy = {}
    for cls in predictions_df['actual'].unique():
        cls_mask = predictions_df['actual'] == cls
        if cls_mask.sum() > 0:
            cls_acc = (predictions_df.loc[cls_mask, 'predicted'] == cls).mean()
            class_accuracy[int(cls)] = {
                'accuracy': cls_acc,
                'samples': int(cls_mask.sum())
            }

    results['class_accuracy'] = class_accuracy

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(predictions_df['actual'], predictions_df['predicted'])
    results['confusion_matrix'] = cm.tolist()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save plot
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_plot_path = os.path.join(plots_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_plot_path)
    results['confusion_matrix_plot'] = cm_plot_path
    plt.close()

    # Analyze confidence levels if available
    if 'confidence' in predictions_df.columns:
        confidence_results = {}

        # Confidence stats
        conf_stats = predictions_df['confidence'].describe()
        confidence_results['stats'] = {
            'min': float(conf_stats['min']),
            'max': float(conf_stats['max']),
            'mean': float(conf_stats['mean']),
            'std': float(conf_stats['std']),
            'median': float(conf_stats['50%'])
        }

        # Accuracy by confidence level (fix deprecated warning)
        conf_bins = np.linspace(0.5, 1, num=6)
        conf_groups = pd.cut(predictions_df['confidence'], bins=conf_bins)
        accuracy_by_conf = predictions_df.groupby(conf_groups, observed=True).apply(
            lambda x: (x['actual'] == x['predicted']).mean() if len(x) > 0 else np.nan
        ).dropna()

        samples_by_conf = predictions_df.groupby(conf_groups, observed=True).size()

        # Store results
        confidence_results['by_level'] = {}
        for conf_range, acc in accuracy_by_conf.items():
            samples = samples_by_conf.get(conf_range, 0)
            confidence_results['by_level'][str(conf_range)] = {
                'accuracy': float(acc),
                'samples': int(samples),
                'percentage': float(samples / len(predictions_df) * 100)
            }

        results['confidence'] = confidence_results

        # Plot accuracy by confidence
        plt.figure(figsize=(10, 6))
        ax = accuracy_by_conf.plot(kind='bar', color='skyblue')

        # Add sample counts
        for i, v in enumerate(accuracy_by_conf):
            samples = samples_by_conf.iloc[i]
            ax.text(i, v + 0.02, f"{samples} samples", ha='center')

        plt.title('Accuracy by Confidence Level')
        plt.xlabel('Confidence Range')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, axis='y', alpha=0.3)

        # Save plot
        conf_plot_path = os.path.join(plots_dir, f"accuracy_by_confidence_{timestamp}.png")
        plt.savefig(conf_plot_path)
        results['confidence_plot'] = conf_plot_path
        plt.close()

    return results


def analyze_consecutive_predictions(predictions_df):
    """Analyze consecutive predictions of the same type."""
    if predictions_df is None or len(predictions_df) == 0:
        logger.error("No predictions to analyze")
        return None

    results = {}

    # Calculate runs of consecutive predictions
    predictions_df = predictions_df.copy()
    predictions_df['pred_change'] = predictions_df['predicted'].diff().ne(0).cumsum()
    runs = predictions_df.groupby(['predicted', 'pred_change']).size().reset_index(name='run_length')

    # Analyze run lengths
    run_stats = {}
    for pred_type in sorted(runs['predicted'].unique()):
        pred_runs = runs[runs['predicted'] == pred_type]
        if len(pred_runs) > 0:
            avg_run = pred_runs['run_length'].mean()
            max_run = pred_runs['run_length'].max()

            # Store in results
            run_stats[int(pred_type)] = {
                'avg_length': float(avg_run),
                'max_length': int(max_run),
                'num_runs': int(len(pred_runs))
            }

            # Distribution of run lengths
            run_counts = pred_runs['run_length'].value_counts().sort_index()

            # Store distribution in results
            distribution = {}
            for length, count in run_counts.items():
                distribution[int(length)] = int(count)

            run_stats[int(pred_type)]['distribution'] = distribution

    results['run_stats'] = run_stats

    # Plot distribution of run lengths
    plt.figure(figsize=(12, 6))

    for pred_type in sorted(runs['predicted'].unique()):
        pred_runs = runs[runs['predicted'] == pred_type]
        run_counts = pred_runs['run_length'].value_counts().sort_index()
        plt.bar(run_counts.index - 0.2 + 0.4 * pred_type, run_counts.values,
                width=0.4, alpha=0.7, label=f'Predicted {pred_type}')

    plt.title('Distribution of Consecutive Prediction Runs')
    plt.xlabel('Run Length (Number of Consecutive Periods)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()

    # Save plot
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_plot_path = os.path.join(plots_dir, f"consecutive_predictions_{timestamp}.png")
    plt.savefig(runs_plot_path)
    results['runs_plot'] = runs_plot_path
    plt.close()

    return results


def analyze_price_movement_vs_prediction(predictions_df, df):
    """Analyze how price movements relate to predictions."""
    if predictions_df is None or len(predictions_df) == 0 or df is None:
        logger.error("No predictions or price data to analyze")
        return None

    results = {}

    # Merge with price data if needed
    if 'close' not in predictions_df.columns:
        predictions_df = predictions_df.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                predictions_df[col] = df.loc[predictions_df.index, col]

    # Calculate actual price change
    predictions_df['actual_change_pct'] = predictions_df['close'].pct_change(1) * 100

    # Calculate average price changes by prediction
    price_changes_by_pred = predictions_df.groupby('predicted')['actual_change_pct'].agg(['mean', 'std', 'count'])

    # Store in results
    price_change_stats = {}
    for pred, stats in price_changes_by_pred.iterrows():
        price_change_stats[int(pred)] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'count': int(stats['count'])
        }

    results['price_changes'] = price_change_stats

    # Plot price changes by prediction type
    plt.figure(figsize=(10, 6))

    for pred in sorted(predictions_df['predicted'].unique()):
        subset = predictions_df[predictions_df['predicted'] == pred]['actual_change_pct'].dropna()
        if len(subset) > 0:
            sns.histplot(subset, bins=20, alpha=0.6, label=f'Predicted {pred}', kde=True)

    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Distribution of Price Changes by Prediction Type')
    plt.xlabel('Actual Price Change (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plots_dir = os.path.join(project_root, "analysis")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    price_plot_path = os.path.join(plots_dir, f"price_change_by_prediction_{timestamp}.png")
    plt.savefig(price_plot_path)
    results['price_change_plot'] = price_plot_path
    plt.close()

    # Analyze prediction accuracy by price volatility
    predictions_df['volatility'] = df.loc[predictions_df.index]['high'] / df.loc[predictions_df.index]['low'] - 1

    # Check if we have sufficient volatility data to create bins
    if not predictions_df['volatility'].isna().all() and len(predictions_df['volatility'].unique()) > 1:
        try:
            volatility_bins = pd.qcut(predictions_df['volatility'], 5)
            accuracy_by_volatility = predictions_df.groupby(volatility_bins, observed=True).apply(
                lambda x: (x['actual'] == x['predicted']).mean() if len(x) > 0 else np.nan
            ).dropna()

            # Store volatility results
            volatility_results = {}
            for vol_range, acc in accuracy_by_volatility.items():
                samples = predictions_df.groupby(volatility_bins, observed=True).size().get(vol_range, 0)
                volatility_results[str(vol_range)] = {
                    'accuracy': float(acc),
                    'samples': int(samples)
                }

            results['volatility'] = volatility_results
        except Exception as e:
            logger.warning(f"Error calculating accuracy by volatility: {str(e)}")
            results['volatility'] = {"error": str(e)}
    else:
        logger.warning("Insufficient volatility data for analysis")
        results['volatility'] = {"error": "Insufficient volatility data"}

    return results


def check_trading_opportunities(predictions_df, confidence_threshold=0.6):
    """Find high-confidence trading opportunities in the most recent data."""
    if predictions_df is None or len(predictions_df) == 0:
        logger.error("No predictions to analyze")
        return None

    if 'confidence' not in predictions_df.columns:
        logger.warning("Confidence values not available for trade analysis")
        return None

    results = {}
    results['confidence_threshold'] = confidence_threshold

    # Look at the most recent predictions
    recent_df = predictions_df.sort_index().tail(50)

    # Find high-confidence signals
    high_conf_up = recent_df[(recent_df['predicted'] == 1) &
                             (recent_df['confidence'] >= confidence_threshold)]
    high_conf_down = recent_df[(recent_df['predicted'] == 0) &
                               (recent_df['confidence'] >= confidence_threshold)]

    results['signals_summary'] = {
        'up': {
            'count': len(high_conf_up),
            'percentage': float(len(high_conf_up) / len(recent_df) * 100) if len(recent_df) > 0 else 0
        },
        'down': {
            'count': len(high_conf_down),
            'percentage': float(len(high_conf_down) / len(recent_df) * 100) if len(recent_df) > 0 else 0
        }
    }

    # Look at the most recent 5 periods
    very_recent = recent_df.tail(5)

    recent_signals = []
    for idx, row in very_recent.iterrows():
        direction = "UP" if row['predicted'] == 1 else "DOWN"
        conf = row.get('confidence', None)

        signal_info = {
            'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
            'direction': direction,
            'confidence': float(conf) if conf is not None else None,
        }

        if not pd.isna(row['actual']):
            signal_info['correct'] = bool(row['actual'] == row['predicted'])
        else:
            signal_info['correct'] = None

        recent_signals.append(signal_info)

    results['recent_signals'] = recent_signals

    # Most recent trade signal
    if len(recent_df) > 0:
        last_signal = recent_df.iloc[-1]
        direction = "UP" if last_signal['predicted'] == 1 else "DOWN"
        conf = last_signal.get('confidence', None)

        results['latest_signal'] = {
            'timestamp': recent_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'direction': direction,
            'confidence': float(conf) if conf is not None else None,
            'is_trading_opportunity': bool(conf is not None and conf >= confidence_threshold)
        }

    return results


def analyze_predictions(model_path: str, timeframe: str = 'H1', date_range: Optional[Tuple[str, str]] = None,
                        use_test_data: bool = True, confidence_threshold: float = 0.65) -> Dict:
    """Main function to analyze model predictions."""
    logger.info(f"Starting prediction analysis for model: {model_path}")
    logger.info(
        f"Parameters: timeframe={timeframe}, use_test_data={use_test_data}, confidence_threshold={confidence_threshold}")

    # Load model and data
    model, model_info, data_info = load_model_and_data(
        model_path=model_path,
        timeframe=timeframe,
        date_range=date_range,
        use_test_data=use_test_data
    )

    if model is None:
        logger.error("Failed to load model")
        return {'error': 'Failed to load model'}

    if data_info is None:
        logger.error("Failed to load data")
        return {'error': 'Failed to load data'}

    df, horizon = data_info

    # Generate predictions
    predictions_df = generate_predictions(model, df, horizon)

    if predictions_df is None or len(predictions_df) == 0:
        logger.error("Failed to generate predictions")
        return {'error': 'Failed to generate predictions'}

    # Run analyses and collect results
    results = {'model_info': model_info, 'time_analysis': analyze_predictions_over_time(predictions_df, df),
               'accuracy_analysis': analyze_prediction_accuracy(predictions_df),
               'consecutive_analysis': analyze_consecutive_predictions(predictions_df),
               'price_analysis': analyze_price_movement_vs_prediction(predictions_df, df),
               'trading_opportunities': check_trading_opportunities(
                   predictions_df,
                   confidence_threshold=confidence_threshold
               )}

    # Create summary statistics
    summary = {
        "Model": os.path.basename(model_path),
        "Timeframe": timeframe,
        "Data Period": f"{model_info['data_period']['start']} to {model_info['data_period']['end']}",
        "Overall Accuracy": f"{results['accuracy_analysis']['overall_accuracy']:.4f}",
    }

    if 1 in results['accuracy_analysis']['class_accuracy']:
        summary["Win Rate Class 1 (UP)"] = f"{results['accuracy_analysis']['class_accuracy'][1]['accuracy']:.4f}"

    if 'confidence' in results['accuracy_analysis']:
        summary["Avg Confidence"] = f"{results['accuracy_analysis']['confidence']['stats']['mean']:.4f}"

    if 'trading_opportunities' in results and results['trading_opportunities']:
        summary["Recent Signals"] = f"UP: {results['trading_opportunities']['signals_summary']['up']['count']}, " \
                                    f"DOWN: {results['trading_opportunities']['signals_summary']['down']['count']}"

        # Add latest signal info
        if 'latest_signal' in results['trading_opportunities']:
            latest = results['trading_opportunities']['latest_signal']
            summary["Latest Signal"] = f"{latest['direction']} ({latest['confidence']:.2f})"
            summary["Trade Opportunity"] = "Yes" if latest['is_trading_opportunity'] else "No"

    results['summary'] = summary

    # Generate HTML report if report_generator exists
    try:
        from debug_tools.report_generator_analyze_predictions import generate_html_report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"prediction_analysis_{os.path.basename(model_path)}_{timestamp}.html"
        report_dir = os.path.join(project_root, "analysis")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, report_name)
        generate_html_report(results, report_path)
        logger.info(f"HTML report generated: {report_path}")
        results['report_path'] = report_path
    except ImportError:
        logger.warning("Report generator module not found. No HTML report will be generated.")
        results['report_path'] = None

    return results