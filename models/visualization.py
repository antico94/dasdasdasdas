import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config.constants import PredictionTarget, TradeAction


class ModelVisualizer:
    """Visualization tools for model evaluation and trading results."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_type = config.get('model', {}).get('prediction_target', PredictionTarget.DIRECTION.value)
        self.is_classifier = self.target_type in [
            PredictionTarget.DIRECTION.value,
            PredictionTarget.VOLATILITY.value
        ]

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_feature_importance(
            self,
            feature_importance: Dict[str, float],
            top_n: int = 20,
            title: str = "Feature Importance"
    ) -> None:
        """Plot feature importance from a trained model."""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top N features
        top_features = sorted_features[:top_n]

        # Create DataFrame for plotting
        df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])

        # Create bar plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=df)

        # Add title and labels
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # Add values to bars
        for i, v in enumerate(df['Importance']):
            ax.text(v + 0.001, i, f"{v:.4f}", va='center')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(
            self,
            y_true: Union[List, np.ndarray],
            y_pred: Union[List, np.ndarray],
            title: str = "Confusion Matrix"
    ) -> None:
        """Plot confusion matrix from classification results."""
        if not self.is_classifier:
            print("Confusion matrix is only available for classification models.")
            return

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create display
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Down', 'Up'] if self.target_type == PredictionTarget.DIRECTION.value else ['Low', 'High']
        )

        # Plot
        plt.figure(figsize=(8, 6))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_equity_curve(
            self,
            results: pd.DataFrame,
            title: str = "Equity Curve"
    ) -> None:
        """Plot equity curve and drawdown."""
        if 'equity' not in results.columns or 'balance' not in results.columns:
            print("Equity data not found in results.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot equity curve
        ax1.plot(results.index, results['balance'], label='Balance', color='blue')
        ax1.plot(results.index, results['equity'], label='Equity', color='green', alpha=0.7)
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel('Account Value', fontsize=12)
        ax1.legend()
        ax1.grid(True)

        # Calculate and plot drawdown
        equity = results['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100  # Convert to percentage

        ax2.fill_between(results.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_trades(
            self,
            results: pd.DataFrame,
            title: str = "Trading Signals and Price Action"
    ) -> None:
        """Plot price chart with trade entries and exits."""
        if 'close' not in results.columns or 'signal' not in results.columns:
            print("Required data not found in results.")
            return

        plt.figure(figsize=(14, 7))

        # Plot price
        plt.plot(results.index, results['close'], label='Close Price', color='blue')

        # Plot buy signals
        buy_signals = results[results['signal'] == TradeAction.BUY.value]
        if len(buy_signals) > 0:
            plt.scatter(
                buy_signals.index,
                buy_signals['close'],
                marker='^',
                color='green',
                s=100,
                label='Buy Signal'
            )

        # Plot sell signals
        sell_signals = results[results['signal'] == TradeAction.SELL.value]
        if len(sell_signals) > 0:
            plt.scatter(
                sell_signals.index,
                sell_signals['close'],
                marker='v',
                color='red',
                s=100,
                label='Sell Signal'
            )

        # Add entry and exit points if available
        if 'entry_price' in results.columns and 'exit_price' in results.columns:
            entries = results[~results['entry_price'].isna()]
            exits = results[~results['exit_price'].isna()]

            if len(entries) > 0:
                plt.scatter(
                    entries.index,
                    entries['entry_price'],
                    marker='o',
                    color='lime',
                    s=80,
                    label='Entry'
                )

            if len(exits) > 0:
                plt.scatter(
                    exits.index,
                    exits['exit_price'],
                    marker='x',
                    color='darkred',
                    s=80,
                    label='Exit'
                )

        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trade_distribution(
            self,
            trades: List[Dict],
            title: str = "Trade Distribution"
    ) -> None:
        """Plot distribution of trade returns."""
        if not trades:
            print("No trades available to visualize.")
            return

        # Extract trade returns
        returns = [trade['return'] * 100 for trade in trades]  # Convert to percentage

        plt.figure(figsize=(12, 6))

        # Create histogram
        sns.histplot(returns, bins=20, kde=True)

        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--')

        # Add statistics
        plt.title(title, fontsize=16)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Add text with statistics
        stats_text = (
            f"Total Trades: {len(trades)}\n"
            f"Winning Trades: {sum(1 for r in returns if r > 0)}\n"
            f"Losing Trades: {sum(1 for r in returns if r <= 0)}\n"
            f"Average Return: {np.mean(returns):.2f}%\n"
            f"Median Return: {np.median(returns):.2f}%\n"
            f"Max Return: {max(returns):.2f}%\n"
            f"Min Return: {min(returns):.2f}%"
        )
        plt.text(
            0.02, 0.95, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )

        plt.tight_layout()
        plt.show()

    def plot_monthly_returns(
            self,
            results: pd.DataFrame,
            title: str = "Monthly Returns"
    ) -> None:
        """Plot heatmap of monthly returns."""
        if 'balance' not in results.columns:
            print("Balance data not found in results.")
            return

        # Calculate daily returns
        daily_returns = results['balance'].pct_change().fillna(0)

        # Resample to monthly returns
        monthly_returns = (daily_returns + 1).resample('M').prod() - 1

        # Convert to DataFrame with year and month
        monthly_df = pd.DataFrame(monthly_returns)
        monthly_df = monthly_df.reset_index()
        monthly_df['Year'] = monthly_df['time'].dt.year
        monthly_df['Month'] = monthly_df['time'].dt.month

        # Pivot for heatmap
        pivot_df = monthly_df.pivot('Year', 'Month', 'balance')

        # Convert to percentage
        pivot_df = pivot_df * 100

        # Plot heatmap
        plt.figure(figsize=(14, 8))

        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            linewidths=1,
            cbar_kws={'label': 'Return (%)'}
        )

        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)

        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(
            self,
            results: pd.DataFrame,
            title: str = "Predictions vs Actual"
    ) -> None:
        """Plot model predictions against actual values."""
        if 'prediction' not in results.columns:
            print("Prediction data not found in results.")
            return

        if self.is_classifier:
            # For classification, plot prediction vs actual direction
            target_col = next((col for col in results.columns if col.startswith('target_')), None)

            if target_col is None:
                print("Target column not found in results.")
                return

            # Create accuracy plot
            plt.figure(figsize=(14, 7))

            # Plot actual vs predicted
            correct = results[results['prediction'] == results[target_col]]
            incorrect = results[results['prediction'] != results[target_col]]

            # Plot price
            plt.plot(results.index, results['close'], color='blue', alpha=0.5)

            # Plot correct/incorrect predictions
            if len(correct) > 0:
                plt.scatter(
                    correct.index,
                    correct['close'],
                    marker='o',
                    color='green',
                    label='Correct Prediction'
                )

            if len(incorrect) > 0:
                plt.scatter(
                    incorrect.index,
                    incorrect['close'],
                    marker='x',
                    color='red',
                    label='Incorrect Prediction'
                )

            plt.title(title, fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend()
            plt.grid(True)

            # Add accuracy statistics
            accuracy = len(correct) / (len(correct) + len(incorrect)) if len(correct) + len(incorrect) > 0 else 0
            plt.text(
                0.02, 0.95, f"Accuracy: {accuracy:.2%}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top'
            )

        else:
            # For regression, plot predicted vs actual values
            target_col = next((col for col in results.columns if col.startswith('target_')), None)

            if target_col is None:
                print("Target column not found in results.")
                return

            plt.figure(figsize=(14, 7))

            # Plot actual values
            plt.plot(results.index, results[target_col], label='Actual', color='blue')

            # Plot predictions
            plt.plot(results.index, results['prediction'], label='Predicted', color='red', alpha=0.7)

            plt.title(title, fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True)

            # Add error statistics
            error = results['prediction'] - results[target_col]
            rmse = np.sqrt(np.mean(error ** 2))
            mae = np.mean(np.abs(error))

            stats_text = (
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}"
            )
            plt.text(
                0.02, 0.95, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top'
            )

        plt.tight_layout()
        plt.show()

    def plot_optimization_results(
            self,
            results: Dict,
            title: str = "Hyperparameter Optimization Results"
    ) -> None:
        """Plot optimization results."""
        model_types = [t for t in results.keys() if t in
                       ['random_forest', 'xgboost', 'lstm', 'ensemble']]

        for model_type in model_types:
            model_results = results[model_type]

            if 'all_results' not in model_results:
                continue

            # For sklearn optimizers
            if hasattr(model_results['all_results'], 'keys'):
                param_names = [p for p in model_results['all_results'].keys()
                               if p.startswith('param_')]
                scores = model_results['all_results']['mean_test_score']

                # Create subplots for each parameter
                n_params = len(param_names)

                if n_params == 0:
                    continue

                fig, axes = plt.subplots(1, n_params, figsize=(n_params * 5, 5))
                if n_params == 1:
                    axes = [axes]

                for i, param in enumerate(param_names):
                    param_values = model_results['all_results'][param]

                    # Create parameter-score plot
                    axes[i].scatter(param_values, scores)
                    axes[i].set_xlabel(param.replace('param_', ''))
                    axes[i].set_ylabel('Score')
                    axes[i].set_title(f"{param.replace('param_', '')} vs Score")

                    # Add best parameter
                    best_value = model_results['best_params'][param.replace('param_', '')]
                    axes[i].axvline(x=best_value, color='red', linestyle='--',
                                    label=f'Best: {best_value}')
                    axes[i].legend()

                plt.suptitle(f"{title} - {model_type}", fontsize=16)
                plt.tight_layout()
                plt.show()

            # For custom optimizers (like LSTM)
            elif isinstance(model_results['all_results'], list):
                params_list = [r['params'] for r in model_results['all_results']]
                scores = [r['score'] for r in model_results['all_results']]

                # Find unique parameter keys
                param_keys = set()
                for params in params_list:
                    for key in params.keys():
                        param_keys.add(key)

                # Create a plot for each parameter
                for param_key in param_keys:
                    plt.figure(figsize=(10, 5))

                    # Extract parameter values and corresponding scores
                    param_values = []
                    param_scores = []

                    for params, score in zip(params_list, scores):
                        if param_key in params:
                            # Handle different parameter types
                            param_value = params[param_key]
                            if isinstance(param_value, list):
                                param_value = str(param_value)  # Convert lists to string for plotting

                            param_values.append(param_value)
                            param_scores.append(score)

                    # Plot if we have data
                    if param_values:
                        # For numeric parameters
                        if all(isinstance(v, (int, float)) for v in param_values):
                            plt.scatter(param_values, param_scores)

                            # Add best parameter
                            best_value = model_results['best_params'][param_key]
                            plt.axvline(x=best_value, color='red', linestyle='--',
                                        label=f'Best: {best_value}')
                        else:
                            # For categorical/string parameters
                            unique_values = list(set(param_values))
                            value_to_idx = {val: idx for idx, val in enumerate(unique_values)}

                            # Group scores by parameter value
                            value_scores = {val: [] for val in unique_values}
                            for val, score in zip(param_values, param_scores):
                                value_scores[val].append(score)

                            # Calculate mean score for each value
                            mean_scores = [np.mean(value_scores[val]) for val in unique_values]

                            # Plot bar chart
                            plt.bar(range(len(unique_values)), mean_scores, tick_label=unique_values)

                            # Highlight best value
                            best_value = str(model_results['best_params'][param_key])
                            best_idx = value_to_idx.get(best_value, -1)
                            if best_idx >= 0:
                                plt.bar([best_idx], [mean_scores[best_idx]], color='red',
                                        label=f'Best: {best_value}')

                        plt.xlabel(param_key)
                        plt.ylabel('Score')
                        plt.title(f"{param_key} vs Score")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()


def visualize_backtest_results(backtest_path: str) -> None:
    """Visualize backtest results from saved file."""
    import pickle

    with open(backtest_path, 'rb') as f:
        backtest_data = pickle.load(f)

    metrics = backtest_data.get('metrics', {})
    results_dict = backtest_data.get('results', {})
    trades = backtest_data.get('trades', [])

    # Convert results dict back to DataFrame
    if results_dict:
        results = pd.DataFrame(results_dict)
        if 'time' in results:
            results.set_index('time', inplace=True)
    else:
        results = pd.DataFrame()

    # Create visualizer
    config = metrics.get('config', {})
    visualizer = ModelVisualizer({'model': config})

    # Print summary
    print("\nBacktest Results Summary:")
    print("------------------------")
    print(f"Initial Balance: ${metrics.get('initial_balance', 0):.2f}")
    print(f"Final Balance: ${metrics.get('final_balance', 0):.2f}")
    print(f"Total Return: {metrics.get('return_pct', 0):.2f}%")
    print(f"Number of Trades: {metrics.get('n_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

    # Plot visualizations
    if not results.empty:
        visualizer.plot_equity_curve(results, title="Equity Curve")
        visualizer.plot_trades(results, title="Trading Signals")

    if trades:
        visualizer.plot_trade_distribution(trades, title="Trade Distribution")

    if not results.empty:
        visualizer.plot_monthly_returns(results, title="Monthly Returns")
        visualizer.plot_predictions_vs_actual(results, title="Predictions vs Actual")

    # If we have confusion matrix data
    model_metrics = metrics.get('model_metrics', {})
    if 'confusion_matrix' in model_metrics:
        cm = model_metrics['confusion_matrix']
        visualizer.plot_confusion_matrix(
            np.zeros(1),  # Dummy data, not used
            np.zeros(1),  # Dummy data, not used
            title="Confusion Matrix"
        )
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Down', 'Up'] if config.get('target_type') == PredictionTarget.DIRECTION.value
            else ['Low', 'High']
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix", fontsize=16)
        plt.tight_layout()
        plt.show()