import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data.storage import DataStorage


def analyze_backtest_results():
    """Analyze the most recent backtest results to identify issues."""
    storage = DataStorage()
    base_path = os.path.join(storage.base_path, "models/data/results")

    # Find the most recent backtest file
    backtest_files = list(Path(base_path).glob("*backtest_*.pkl"))
    if not backtest_files:
        print("No backtest results found.")
        return

    latest_backtest = max(backtest_files, key=os.path.getctime)
    print(f"Analyzing backtest results from: {latest_backtest}")

    # Load the backtest results
    with open(latest_backtest, 'rb') as f:
        backtest_data = pickle.load(f)

    # Extract relevant components (structure may vary based on your implementation)
    if isinstance(backtest_data, dict):
        metrics = backtest_data.get('metrics', {})
        trades = backtest_data.get('trades', pd.DataFrame())
        equity_curve = backtest_data.get('equity_curve', pd.DataFrame())
        predictions = backtest_data.get('predictions', pd.DataFrame())
    else:
        print("Unexpected backtest data format. Attempting to recover what we can...")
        # Try to recover what we can
        metrics, trades, equity_curve, predictions = None, None, None, None

        if hasattr(backtest_data, 'metrics'):
            metrics = backtest_data.metrics
        if hasattr(backtest_data, 'trades'):
            trades = backtest_data.trades
        if hasattr(backtest_data, 'equity_curve'):
            equity_curve = backtest_data.equity_curve
        if hasattr(backtest_data, 'predictions'):
            predictions = backtest_data.predictions

    print("\n=== Backtest Metrics ===")
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}")

    # Analyze trades
    analyze_trades(trades)

    # Analyze predictions if available
    if predictions is not None and not isinstance(predictions, pd.DataFrame):
        if isinstance(predictions, dict):
            # Convert dict to DataFrame if needed
            predictions = pd.DataFrame(predictions)
        else:
            predictions = None

    if predictions is not None:
        analyze_predictions(predictions)

    # Try to load the model and extract feature importance
    try:
        model_files = list(Path(base_path).glob("ensemble_*.joblib"))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            analyze_model_features(latest_model)
    except Exception as e:
        print(f"Error analyzing model features: {str(e)}")


def analyze_trades(trades):
    """Analyze trading performance in detail."""
    if trades is None or len(trades) == 0:
        print("\n=== No trades to analyze ===")
        return

    print(f"\n=== Trade Analysis ({len(trades)} trades) ===")

    # Convert to DataFrame if it's not already
    if not isinstance(trades, pd.DataFrame):
        try:
            trades = pd.DataFrame(trades)
        except:
            print("Could not convert trades to DataFrame for analysis")
            return

    # Basic trade statistics
    win_rate = (trades['profit'] > 0).mean() if 'profit' in trades.columns else None
    avg_win = trades.loc[trades['profit'] > 0, 'profit'].mean() if 'profit' in trades.columns else None
    avg_loss = trades.loc[trades['profit'] < 0, 'profit'].mean() if 'profit' in trades.columns else None
    profit_factor = abs(
        trades.loc[trades['profit'] > 0, 'profit'].sum() /
        trades.loc[trades['profit'] < 0, 'profit'].sum()
    ) if 'profit' in trades.columns else None

    print(f"Win Rate: {win_rate:.2f}" if win_rate is not None else "Win Rate: N/A")
    print(f"Average Win: {avg_win:.2f}" if avg_win is not None else "Average Win: N/A")
    print(f"Average Loss: {avg_loss:.2f}" if avg_loss is not None else "Average Loss: N/A")
    print(f"Profit Factor: {profit_factor:.2f}" if profit_factor is not None else "Profit Factor: N/A")

    # Check for columns we expect to have
    expected_columns = {'entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'direction'}
    available_columns = set(trades.columns)
    print(f"\nAvailable trade data columns: {', '.join(available_columns)}")

    # Time analysis of trades
    if 'entry_time' in trades.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(trades['entry_time']):
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])

        # Group trades by hour to see if there's a pattern
        trades['hour'] = trades['entry_time'].dt.hour
        hourly_performance = trades.groupby('hour')['profit'].agg(['mean', 'sum', 'count'])
        print("\nPerformance by hour of day:")
        print(hourly_performance)

        # Plot the hourly performance
        plt.figure(figsize=(12, 6))
        hourly_performance['sum'].plot(kind='bar', color='blue', alpha=0.7)
        plt.title("Profit/Loss by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Total Profit/Loss")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("hourly_performance.png")
        print(f"Saved hourly performance chart to hourly_performance.png")

    # Direction analysis
    if 'direction' in trades.columns:
        direction_performance = trades.groupby('direction')['profit'].agg(['mean', 'sum', 'count'])
        print("\nPerformance by trade direction:")
        print(direction_performance)

    # Consecutive losses
    if 'profit' in trades.columns:
        trades['profitable'] = trades['profit'] > 0
        trades['streak'] = (trades['profitable'] != trades['profitable'].shift()).cumsum()
        loss_streaks = trades[~trades['profitable']].groupby('streak')['profit'].count()
        if not loss_streaks.empty:
            max_consecutive_losses = loss_streaks.max()
            print(f"\nMaximum consecutive losses: {max_consecutive_losses}")

    # Plot profit distribution
    if 'profit' in trades.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(trades['profit'], kde=True, bins=20)
        plt.title("Distribution of Trade Profits")
        plt.xlabel("Profit/Loss")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("profit_distribution.png")
        print(f"Saved profit distribution chart to profit_distribution.png")


def analyze_predictions(predictions):
    """Analyze model predictions."""
    print("\n=== Prediction Analysis ===")

    # Check if we have the necessary columns
    if 'actual' in predictions.columns and 'predicted' in predictions.columns:
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        conf_matrix = confusion_matrix(predictions['actual'], predictions['predicted'])
        report = classification_report(predictions['actual'], predictions['predicted'])

        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(report)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print(f"Saved confusion matrix to confusion_matrix.png")
    else:
        print("Prediction data doesn't contain 'actual' and 'predicted' columns for analysis")


def analyze_model_features(model_path):
    """Extract and analyze feature importance from the model."""
    print(f"\n=== Model Feature Analysis ===")
    print(f"Loading model from {model_path}")

    import joblib
    model = joblib.load(model_path)

    # Try different approaches to get feature importance based on model type
    feature_importance = None
    feature_names = None

    try:
        # For ensemble models, there might be a get_feature_importance method
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if isinstance(feature_importance, tuple) and len(feature_importance) == 2:
                feature_names, importance_values = feature_importance
        # For RandomForest or XGBoost models
        elif hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
        # For ensemble models that have a .estimators_ attribute (scikit-learn)
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            # Try to get from the first estimator
            if hasattr(model.estimators_[0], 'feature_importances_'):
                feature_importance = model.estimators_[0].feature_importances_
    except Exception as e:
        print(f"Error extracting feature importance: {str(e)}")

    if feature_importance is not None:
        print("\nFeature Importance:")

        # If we have both names and values
        if feature_names is not None and len(feature_names) == len(feature_importance):
            # Create a DataFrame for better display
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            print(fi_df.head(20))  # Show top 20 features

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
            plt.title("Top 20 Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            print(f"Saved feature importance chart to feature_importance.png")
        else:
            # We have importance values but no names
            print(feature_importance)
    else:
        print("Could not extract feature importance from the model")


if __name__ == "__main__":
    analyze_backtest_results()
