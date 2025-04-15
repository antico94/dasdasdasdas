import os
import sys
import joblib
import numpy as np
import pandas as pd
# Fix matplotlib backend to avoid GUI dependencies
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import questionary
from questionary import Choice, Separator

matplotlib.use('Agg')


# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from data.storage import DataStorage
from data.processor import DataProcessor
from utils.logger import setup_logger

# Import the report generator if it exists
try:
    from debug_tools.report_generator_analyze_predictions import generate_html_report

    has_report_generator = True
except ImportError:
    has_report_generator = False
    print("Warning: report_generator.py not found. Reports will not be generated.")
    print("Please make sure report_generator.py is in the debug_tools directory.")

logger = setup_logger("PredictionAnalyzer")


def select_model():
    """Let the user select a model using questionary interface."""
    # Find model files
    models_dir = os.path.join(project_root, "data", "models")
    if not os.path.exists(models_dir):
        logger.error("Models directory not found at: " + models_dir)
        return None

    model_files = [f for f in os.listdir(models_dir) if
                   f.endswith('.joblib') and not f.endswith('_random_forest.joblib')
                   and not f.endswith('_xgboost.joblib')
                   and not f.endswith('_metadata.pkl')]

    if not model_files:
        logger.error("No model files found in: " + models_dir)
        return None

    # Sort by creation time (most recent first)
    model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)

    # Create choices for questionary
    choices = []
    for i, model_file in enumerate(model_files):
        file_path = os.path.join(models_dir, model_file)
        created_time = datetime.fromtimestamp(os.path.getctime(file_path))
        choices.append(Choice(
            f"{model_file} (created: {created_time.strftime('%Y-%m-%d %H:%M:%S')})",
            file_path
        ))

    # Add "Use latest model" as the first option
    if model_files:
        choices.insert(0, Choice(
            f"Use latest model: {model_files[0]}",
            os.path.join(models_dir, model_files[0])
        ))

    # Ask user to select a model
    selected_model = questionary.select(
        "Select a model to analyze:",
        choices=choices
    ).ask()

    if selected_model:
        print(f"Selected model: {os.path.basename(selected_model)}")
        return selected_model
    else:
        # If user cancels, use the latest model
        if model_files:
            default_model = os.path.join(models_dir, model_files[0])
            print(f"Using latest model: {model_files[0]}")
            return default_model
        return None


def _is_valid_date(date_str):
    """Validate date string format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return 'Invalid date format. Please use YYYY-MM-DD'


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_banner():
    """Display banner for the prediction analyzer."""
    clear_screen()
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗          ║
║   ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝          ║
║   ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║             ║
║   ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║             ║
║   ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║             ║
║   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝             ║
║                                                                  ║
║   XAUUSD Gold Trading Model Analysis Tool                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_model_and_data(model_path=None, timeframe='H1', date_range=None):
    """Load model and latest processed data for the specified timeframe."""
    storage = DataStorage()
    processor = DataProcessor()

    # Find the latest model if not specified
    if model_path is None:
        model_path = select_model()
        if model_path is None:
            logger.error("No model selected. Exiting.")
            return None, None, None

    # Load model
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None, None

        # Load model and metadata
        print(f"Loading model from {model_path}")
        result = storage.load_model(model_path)

        model_info = {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'creation_date': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        }

        if isinstance(result, tuple) and len(result) == 2:
            model, metadata = result
            model.metadata = metadata
            logger.info(f"Model loaded successfully with metadata")
            model_info['metadata'] = metadata
            if metadata:
                print(f"Loaded model metadata: {list(metadata.keys())}")
        else:
            model = result
            metadata = None
            model_info['metadata'] = {}
            logger.info(f"Model loaded successfully without metadata")

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

    # Load the latest processed data
    latest_data = storage.find_latest_processed_data()

    if not latest_data or timeframe not in latest_data:
        logger.error(f"No processed data found for timeframe {timeframe}")
        return model, model_info, None

    data_dict = processor.load_data({timeframe: latest_data[timeframe]})
    df = data_dict[timeframe]
    logger.info(f"Loaded {len(df)} rows of {timeframe} data")

    # Filter by date range if specified
    if date_range is not None:
        start_date, end_date = date_range
        logger.info(f"Filtering data from {start_date} to {end_date}")
        df = df.loc[start_date:end_date]
        logger.info(f"Filtered data contains {len(df)} rows")

        if len(df) == 0:
            logger.error(f"No data available for the specified date range")
            return model, model_info, None

    # Add data info to model_info
    model_info['data_file'] = latest_data[timeframe]
    model_info['data_rows'] = len(df)
    model_info['data_period'] = {
        'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S') if len(df) > 0 else "N/A",
        'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S') if len(df) > 0 else "N/A"
    }

    # Add date range to model_info if specified
    if date_range is not None:
        model_info['selected_date_range'] = date_range

    return model, model_info, (df, horizon)


def generate_predictions(model, df, horizon=1):
    """Generate predictions using the model on the given data."""
    processor = DataProcessor()
    logger = processor.logger  # Use the same logger from DataProcessor

    try:
        # Prepare features
        X, y = processor.prepare_ml_features(df, horizon=horizon)
        logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")

        # Get expected features from model metadata
        expected_features = getattr(model, "metadata", {}).get("features", [])

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

        # Generate predictions
        y_pred = model.predict(X)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            # Add predictions to dataframe
            predictions_df = pd.DataFrame(index=X.index)
            predictions_df['actual'] = y
            predictions_df['predicted'] = y_pred
            if probas is not None and probas.shape[1] >= 2:
                predictions_df['probability_up'] = probas[:, 1]
                predictions_df['probability_down'] = probas[:, 0]
                predictions_df['confidence'] = np.max(probas, axis=1)
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


def main_menu():
    """Show main menu for prediction analyzer."""
    show_banner()

    # Default configuration
    default_config = {
        'model_path': None,  # Will be set during load if None
        'timeframe': 'H1',
        'confidence_threshold': 0.65,
        'date_range': None,  # Will be None for all data, or a tuple of (start_date, end_date)
        'generate_report': True,
        'report_path': None  # Will be auto-generated if None
    }

    # Defaults display
    print("\nCurrent Configuration:")
    print(f"- Model: Will use latest model")
    print(f"- Timeframe: {default_config['timeframe']}")
    print(f"- Confidence threshold: {default_config['confidence_threshold']}")
    print(f"- Date range: All available data")

    use_defaults = questionary.select(
        'How would you like to proceed?',
        choices=[
            Choice('Run analysis with current configuration', 'use_defaults'),
            Choice('Change configuration settings', 'change_config')
        ]
    ).ask()

    if use_defaults == 'use_defaults':
        # Ensure default config has a valid report path
        report_name = f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        default_config['report_path'] = os.path.join(project_root, "analysis", report_name)
        os.makedirs(os.path.join(project_root, "analysis"), exist_ok=True)
        return default_config

    # Custom configuration
    config = default_config.copy()

    # Let user select the model
    model_path = select_model()
    if model_path:
        config['model_path'] = model_path

    # Select timeframe
    config['timeframe'] = questionary.select(
        'Select timeframe for analysis:',
        choices=[
            Choice('M5 (5 minutes)', 'M5'),
            Choice('M15 (15 minutes)', 'M15'),
            Choice('H1 (1 hour)', 'H1'),
            Choice('D1 (Daily)', 'D1'),
        ],
        default=default_config['timeframe']
    ).ask()

    # Set confidence threshold
    threshold_input = questionary.text(
        'Enter confidence threshold for trading opportunities (0.5-1.0):',
        default=str(default_config['confidence_threshold']),
        validate=lambda val: (val.replace('.', '', 1).isdigit() and 0.5 <= float(
            val) <= 1.0) or 'Enter a number between 0.5 and 1.0'
    ).ask()
    config['confidence_threshold'] = float(threshold_input)

    # Date range selection
    date_range_type = questionary.select(
        'Which data range would you like to analyze?',
        choices=[
            Choice('All available data', 'all'),
            Choice('Last N days', 'last_n_days'),
            Choice('Custom date range', 'custom'),
            Choice('Random period', 'random')
        ]
    ).ask()

    if date_range_type == 'all':
        config['date_range'] = None
    elif date_range_type == 'last_n_days':
        days = int(questionary.text(
            'How many days to analyze?',
            default='30',
            validate=lambda val: val.isdigit() and int(val) > 0
        ).ask())
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        config['date_range'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    elif date_range_type == 'custom':
        start_date = questionary.text(
            'Enter start date (YYYY-MM-DD):',
            validate=lambda val: _is_valid_date(val)
        ).ask()
        end_date = questionary.text(
            'Enter end date (YYYY-MM-DD):',
            validate=lambda val: _is_valid_date(val) and datetime.strptime(val, '%Y-%m-%d') >= datetime.strptime(
                start_date, '%Y-%m-%d'),
            validate_while_typing=False
        ).ask()
        config['date_range'] = (start_date, end_date)
    elif date_range_type == 'random':
        # Generate random period for testing
        storage = DataStorage()
        processor = DataProcessor()

        total_days = int(questionary.text(
            'Total days for random period:',
            default='30',
            validate=lambda val: val.isdigit() and int(val) > 0
        ).ask())

        # Get data date range to determine valid random period
        try:
            data_files = storage.find_latest_processed_data()
            if config['timeframe'] in data_files:
                data_file = data_files[config['timeframe']]
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                min_date = df.index.min()
                max_date = df.index.max()
                date_range = (max_date - min_date).days

                if date_range <= total_days:
                    print(
                        f"Warning: Requested random period ({total_days} days) is longer than available data ({date_range} days).")
                    print(
                        f"Using all available data instead: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                    config['date_range'] = (min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
                else:
                    # Generate random start date within available range
                    from random import randint
                    random_start_day = randint(0, date_range - total_days)
                    random_start = min_date + timedelta(days=random_start_day)
                    random_end = random_start + timedelta(days=total_days)
                    config['date_range'] = (random_start.strftime('%Y-%m-%d'), random_end.strftime('%Y-%m-%d'))
                    print(
                        f"Random period selected: {random_start.strftime('%Y-%m-%d')} to {random_end.strftime('%Y-%m-%d')}")
            else:
                print(f"No data found for timeframe {config['timeframe']}. Using all available data.")
                config['date_range'] = None
        except Exception as e:
            print(f"Error determining data range: {str(e)}. Using all available data.")
            config['date_range'] = None

    # Report path (always generate report)
    config['generate_report'] = True
    report_name = f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    default_path = os.path.join(project_root, "analysis", report_name)

    custom_path = questionary.confirm(
        'Would you like to specify a custom path for the report?',
        default=False
    ).ask()

    if custom_path:
        config['report_path'] = questionary.text(
            'Enter path for report file:',
            default=default_path
        ).ask()
    else:
        config['report_path'] = default_path

    # Ensure report directory exists
    report_dir = os.path.dirname(config['report_path'])
    os.makedirs(report_dir, exist_ok=True)

    return config


def run_analysis(config):
    """Run the complete analysis with the specified configuration."""
    try:
        # If model path is not specified, select the latest model
        storage = DataStorage()
        processor = DataProcessor()

        if config['model_path'] is None:
            models_dir = os.path.join(project_root, "data", "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if
                               f.endswith('.joblib') and not f.endswith('_random_forest.joblib')
                               and not f.endswith('_xgboost.joblib')
                               and not f.endswith('_metadata.pkl')]

                if model_files:
                    model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                    config['model_path'] = os.path.join(models_dir, model_files[0])
                    logger.info(f"Using latest model: {model_files[0]}")
                else:
                    logger.error("No model files found")
                    return False
            else:
                logger.error("Models directory not found")
                return False

        # Load model and data
        model, model_info, data_info = load_model_and_data(
            model_path=config['model_path'],
            timeframe=config['timeframe'],
            date_range=config['date_range']
        )

        if model is None:
            logger.error("Failed to load model")
            return False

        if data_info is None:
            logger.error("Failed to load data")
            return False

        df, horizon = data_info

        # Generate predictions
        predictions_df = generate_predictions(model, df, horizon)

        if predictions_df is None or len(predictions_df) == 0:
            logger.error("Failed to generate predictions")
            return False

        # Run analyses and collect results
        results = {}
        results['model_info'] = model_info
        results['time_analysis'] = analyze_predictions_over_time(predictions_df, df)
        results['accuracy_analysis'] = analyze_prediction_accuracy(predictions_df)
        results['consecutive_analysis'] = analyze_consecutive_predictions(predictions_df)
        results['price_analysis'] = analyze_price_movement_vs_prediction(predictions_df, df)
        results['trading_opportunities'] = check_trading_opportunities(
            predictions_df,
            confidence_threshold=config['confidence_threshold']
        )

        # Generate HTML report
        if config['generate_report'] and has_report_generator:
            report_path = config['report_path']
            try:
                from debug_tools.report_generator_analyze_predictions import generate_html_report
                generate_html_report(results, report_path)
                logger.info(f"HTML report generated: {report_path}")
                print(f"\nAnalysis complete! Report generated at: {report_path}")
            except Exception as e:
                logger.error(f"Error generating HTML report: {str(e)}")
                print(f"\nError generating HTML report: {str(e)}")
                return False
        elif config['generate_report'] and not has_report_generator:
            logger.error("Report generation requested but report_generator.py not found")
            print("\nWarning: Report generation requested but report_generator.py not found.")
            print("Please make sure report_generator.py is in the debug_tools directory.")
            return False

        return True

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError during analysis: {str(e)}")
        return False


def main():
    """Main function for the prediction analyzer."""
    try:
        # Show main menu and get configuration
        config = main_menu()

        # Run analysis with the selected configuration
        success = run_analysis(config)

        if not success:
            print("\nAnalysis encountered errors. Please check the log file for details.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nUnexpected error: {str(e)}")


if __name__ == "__main__":
    main()
