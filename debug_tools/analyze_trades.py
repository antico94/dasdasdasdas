import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from utils.logger import setup_logger

logger = setup_logger("TradeAnalyzer")


def load_backtest_results(backtest_file=None):
    """Load backtest results from file."""
    from data.storage import DataStorage
    storage = DataStorage()

    if backtest_file is None:
        # Look for the latest backtest results in the correct directory
        backtest_dir = os.path.join(project_root, "data", "results")
        if not os.path.exists(backtest_dir):
            logger.error("Backtest results directory not found: " + backtest_dir)
            return None

        # Find files that contain "_backtest_" in the name and end with .pkl
        backtest_files = [f for f in os.listdir(backtest_dir) if "_backtest_" in f and f.endswith('.pkl')]

        if not backtest_files:
            logger.error("No backtest files found in " + backtest_dir)
            return None

        # Sort by creation time (most recent first)
        backtest_files.sort(key=lambda x: os.path.getctime(os.path.join(backtest_dir, x)), reverse=True)
        backtest_file = os.path.join(backtest_dir, backtest_files[0])
        logger.info(f"Using latest backtest file: {backtest_files[0]}")

    try:
        # Check if file exists
        if not os.path.exists(backtest_file):
            logger.error(f"Backtest file not found: {backtest_file}")
            return None

        # Load results
        with open(backtest_file, 'rb') as f:
            results = pickle.load(f)

        logger.info(f"Backtest results loaded successfully from {backtest_file}")
        return results
    except Exception as e:
        logger.error(f"Error loading backtest results: {str(e)}")
        return None


def check_backtest_debug_info(results):
    """Check if there's debug info in the backtest results that would help explain why no trades executed."""
    if results is None:
        return

    print("\n===== BACKTEST DEBUG INFORMATION =====")

    # Check what's in the results dictionary
    print("\nBacktest results contents:")
    for key in results.keys():
        value = results[key]
        value_type = type(value)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            value_info = f"{value_type.__name__} with shape {value.shape}"
        elif isinstance(value, (list, tuple, set, dict)):
            value_info = f"{value_type.__name__} with {len(value)} items"
        else:
            value_info = f"{value_type.__name__}: {value}"
        print(f"  {key}: {value_info}")

    # Look for model prediction information
    if 'predictions' in results:
        predictions = results['predictions']
        if predictions is None:
            print("\nPredictions data is None.")
        elif isinstance(predictions, pd.DataFrame):
            print("\nPrediction statistics:")
            if 'prediction' in predictions.columns:
                prediction_counts = predictions['prediction'].value_counts()
                print(f"  Prediction counts: {prediction_counts.to_dict()}")
            if 'probability' in predictions.columns:
                prob_stats = predictions['probability'].describe()
                print(f"  Probability stats: min={prob_stats['min']:.4f}, max={prob_stats['max']:.4f}, mean={prob_stats['mean']:.4f}")
                high_conf = predictions[predictions['probability'] >= 0.65]
                print(f"  High-confidence predictions (>= 0.65): {len(high_conf)} ({len(high_conf) / len(predictions) * 100:.2f}%)")

    # Check for signal generation information
    if 'signals' in results:
        signals = results['signals']
        if signals is None:
            print("\nSignals data is None.")
        elif isinstance(signals, pd.DataFrame):
            print("\nSignal generation information:")
            signal_counts = (signals != 0).sum()
            print(f"  Total non-zero signals: {signal_counts.sum()}")
            if 'buy_signal' in signals.columns and 'sell_signal' in signals.columns:
                buy_count = (signals['buy_signal'] > 0).sum()
                sell_count = (signals['sell_signal'] > 0).sum()
                print(f"  Buy signals: {buy_count}")
                print(f"  Sell signals: {sell_count}")

    # Check trade log information
    if 'trade_log' in results:
        trade_log = results['trade_log']
        if isinstance(trade_log, pd.DataFrame) and len(trade_log) == 0:
            print("\nTrade filtering information:")
            if 'metrics' in results and 'min_confidence' in results['metrics']:
                print(f"  Minimum confidence threshold: {results['metrics']['min_confidence']}")
            if 'metrics' in results and 'max_open_positions' in results['metrics']:
                print(f"  Maximum open positions: {results['metrics']['max_open_positions']}")
            print("\nPossible reasons for no trades:")
            print("  1. No predictions passed the confidence threshold")
            print("  2. Strategy restrictions (e.g., session filters, risk limits)")
            print("  3. Prediction horizon (1-period) not aligned with trading logic")
            print("  4. Signal generation criteria too restrictive")

    if 'debug_logs' in results:
        debug_logs = results['debug_logs']
        print("\nDebug logs:")
        for entry in debug_logs:
            print(f"  {entry}")


def analyze_signal_statistics(results):
    """Analyze signal statistics from backtest results."""
    if results is None or 'predictions' not in results or results['predictions'] is None:
        logger.warning("No prediction data found in backtest results")
        return

    predictions = results['predictions']
    if not isinstance(predictions, pd.DataFrame):
        logger.warning("Predictions data is not in expected format")
        return

    print("\n===== SIGNAL STATISTICS =====")
    if 'prediction' in predictions.columns:
        pred_counts = predictions['prediction'].value_counts(normalize=True) * 100
        print("\nPrediction distribution:")
        for pred, pct in pred_counts.items():
            print(f"  {pred}: {pct:.2f}%")
    if 'probability' in predictions.columns:
        prob_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prob_groups = pd.cut(predictions['probability'], bins=prob_bins)
        prob_counts = predictions.groupby(prob_groups).size()
        print("\nProbability distribution:")
        for prob_range, count in prob_counts.items():
            print(f"  {prob_range}: {count} predictions ({count / len(predictions) * 100:.2f}%)")
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['probability'], bins=20, kde=True)
        plt.axvline(x=0.65, color='r', linestyle='--', label='Typical Threshold (0.65)')
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plots_dir = os.path.join(project_root, "analysis")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(plots_dir, f"probability_distribution_{timestamp}.png"))
        print(f"\nProbability distribution plot saved to: {os.path.join(plots_dir, f'probability_distribution_{timestamp}.png')}")
        plt.close()

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        print("\nImpact of different confidence thresholds:")
        print(f"  {'Threshold':>10} {'Buy Signals':>12} {'Sell Signals':>12} {'Total Signals':>12} {'% of Data':>10}")
        print("  " + "-" * 60)
        for threshold in thresholds:
            if 'prediction' in predictions.columns:
                buy_signals = ((predictions['prediction'] == 1) & (predictions['probability'] >= threshold)).sum()
                sell_signals = ((predictions['prediction'] == 0) & (predictions['probability'] >= threshold)).sum()
            else:
                buy_signals = ((predictions['probability'] >= 0.5) & (predictions['probability'] >= threshold)).sum()
                sell_signals = ((predictions['probability'] < 0.5) & (1 - predictions['probability'] >= threshold)).sum()
            total_signals = buy_signals + sell_signals
            pct_of_data = total_signals / len(predictions) * 100
            print(f"  {threshold:>10.2f} {buy_signals:>12} {sell_signals:>12} {total_signals:>12} {pct_of_data:>10.2f}%")
        suggested_threshold = 0.6
        for threshold in [0.65, 0.6, 0.7, 0.55, 0.75]:
            if 'prediction' in predictions.columns:
                total_signals = ((predictions['prediction'] == 1) & (predictions['probability'] >= threshold)).sum() + \
                                ((predictions['prediction'] == 0) & (predictions['probability'] >= threshold)).sum()
            else:
                total_signals = ((predictions['probability'] >= 0.5) & (predictions['probability'] >= threshold)).sum() + \
                                ((predictions['probability'] < 0.5) & (1 - predictions['probability'] >= threshold)).sum()
            pct_of_data = total_signals / len(predictions) * 100
            if 5 <= pct_of_data <= 15:
                suggested_threshold = threshold
                break
        print(f"\nSuggested confidence threshold: {suggested_threshold} (would generate signals for approximately " +
              f"{((predictions['probability'] >= suggested_threshold).sum() / len(predictions) * 100):.2f}% of data points)")


def simulate_trading_with_threshold(results, confidence_threshold=0.65):
    """Simulate trading with a specific confidence threshold."""
    if results is None or 'predictions' not in results or results['predictions'] is None:
        logger.warning("No prediction data found in backtest results")
        return

    predictions = results['predictions']
    if not isinstance(predictions, pd.DataFrame):
        logger.warning("Predictions data is not in expected format")
        return

    # Get price data
    if 'prices' in results:
        price_data = results['prices']
    else:
        logger.warning("No price data found in backtest results")
        return

    print(f"\n===== SIMULATED TRADING (Threshold: {confidence_threshold}) =====")
    trading_df = predictions.copy()
    if isinstance(price_data, pd.DataFrame):
        for col in ['open', 'high', 'low', 'close']:
            if col in price_data.columns:
                trading_df[col] = price_data.loc[trading_df.index, col]
    if 'prediction' in trading_df.columns and 'probability' in trading_df.columns:
        trading_df['buy_signal'] = (trading_df['prediction'] == 1) & (trading_df['probability'] >= confidence_threshold)
        trading_df['sell_signal'] = (trading_df['prediction'] == 0) & (trading_df['probability'] >= confidence_threshold)
    else:
        trading_df['buy_signal'] = (trading_df['probability'] >= 0.5) & (trading_df['probability'] >= confidence_threshold)
        trading_df['sell_signal'] = (trading_df['probability'] < 0.5) & (1 - trading_df['probability'] >= confidence_threshold)

    buy_signals = trading_df['buy_signal'].sum()
    sell_signals = trading_df['sell_signal'].sum()
    print(f"\nWith {confidence_threshold} threshold:")
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print(f"  Total signals: {buy_signals + sell_signals}")
    print(f"  Percentage of data points with signals: {(buy_signals + sell_signals) / len(trading_df) * 100:.2f}%")
    if 'future_return' in results:
        future_returns = results['future_return']
        if isinstance(future_returns, pd.Series):
            trading_df['future_return'] = future_returns.loc[trading_df.index]
            trade_results = []
            for signal_type in ['buy_signal', 'sell_signal']:
                signal_returns = trading_df.loc[trading_df[signal_type], 'future_return']
                if len(signal_returns) > 0:
                    if signal_type == 'sell_signal':
                        signal_returns = -signal_returns
                    avg_return = signal_returns.mean()
                    win_rate = (signal_returns > 0).mean()
                    trade_results.append({
                        'signal_type': 'Buy' if signal_type == 'buy_signal' else 'Sell',
                        'count': len(signal_returns),
                        'avg_return': avg_return,
                        'win_rate': win_rate,
                        'total_return': signal_returns.sum()
                    })
            if trade_results:
                print("\nSimulated trade outcomes:")
                for result in trade_results:
                    print(f"  {result['signal_type']} trades ({result['count']}):")
                    print(f"    Average return: {result['avg_return']:.4f}%")
                    print(f"    Win rate: {result['win_rate']:.4f}")
                    print(f"    Total return: {result['total_return']:.4f}%")
                total_trades = sum(r['count'] for r in trade_results)
                total_return = sum(r['total_return'] for r in trade_results)
                if total_trades > 0:
                    overall_win_rate = sum(r['win_rate'] * r['count'] for r in trade_results) / total_trades
                    print("\n  Overall:")
                    print(f"    Total trades: {total_trades}")
                    print(f"    Total return: {total_return:.4f}%")
                    print(f"    Average return per trade: {total_return / total_trades:.4f}%")
                    print(f"    Overall win rate: {overall_win_rate:.4f}")


def diagnose_backtest_issues(results):
    """Diagnose common issues with backtesting."""
    print("\n===== BACKTEST DIAGNOSTICS =====")
    if results is None:
        print("No backtest results to analyze")
        return

    if 'trade_log' in results:
        trade_log = results['trade_log']
        trades_exist = isinstance(trade_log, pd.DataFrame) and len(trade_log) > 0
    else:
        trades_exist = False

    if trades_exist:
        print("\nTrades were executed in this backtest.")
    else:
        print("\nNo trades were executed in this backtest.")
        issues = []
        if 'predictions' in results:
            predictions = results['predictions']
            if predictions is None or not isinstance(predictions, pd.DataFrame):
                issues.append("No valid predictions data available")
            else:
                if 'probability' in predictions.columns:
                    high_prob = (predictions['probability'] >= 0.65).sum()
                    if high_prob == 0:
                        issues.append("No predictions with high enough confidence (>= 0.65)")
                    else:
                        non_high_prob_pct = 1 - (high_prob / len(predictions))
                        if non_high_prob_pct > 0.98:
                            issues.append(f"Very few high-confidence predictions ({high_prob}/{len(predictions)}, {high_prob / len(predictions) * 100:.2f}%)")
        if 'metrics' in results and 'min_confidence' in results['metrics']:
            min_conf = results['metrics']['min_confidence']
            if min_conf > 0.7:
                issues.append(f"Confidence threshold might be too high ({min_conf})")
        prediction_horizon = None
        if 'model_info' in results and 'prediction_horizon' in results['model_info']:
            prediction_horizon = results['model_info']['prediction_horizon']
        trading_horizon = None
        if 'metrics' in results and 'holding_periods' in results['metrics']:
            if 'min' in results['metrics']['holding_periods']:
                trading_horizon = results['metrics']['holding_periods']['min']
        if prediction_horizon is not None and trading_horizon is not None and prediction_horizon != trading_horizon:
            issues.append(f"Mismatch between prediction horizon ({prediction_horizon}) and trading horizon ({trading_horizon})")
        if 'metrics' in results and 'session_filters' in results['metrics']:
            session_filters = results['metrics']['session_filters']
            if session_filters and any(f.get('active', False) for f in session_filters):
                issues.append("Session filters are active and may be restricting trades")
        if issues:
            print("\nPotential issues detected:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\nSuggested fixes:")
            print("  1. Reduce confidence threshold (try 0.55 or 0.6)")
            print("  2. Ensure prediction horizon (1) matches trading logic")
            print("  3. Temporarily disable session filters for testing")
            print("  4. Check if model predictions have good separation (up vs down)")
        else:
            print("\nNo specific issues detected. Consider:")
            print("  1. Check if the trading strategy logic is correctly implemented")
            print("  2. Verify if the model predictions are being correctly used for trade decisions")
            print("  3. Add debugging output in the backtest to trace signal generation")
            print("  4. Update backtest to support 1-period prediction horizon")


def main():
    """Main function."""
    print("===== GOLD TRADING BACKTEST ANALYZER =====")
    print("This tool analyzes backtest results to diagnose issues.")

    # Check for backtest file from command line
    backtest_file = None
    if len(sys.argv) > 1:
        backtest_file = sys.argv[1]
        print(f"\nUsing specified backtest file: {backtest_file}")

    # Load backtest results
    results = load_backtest_results(backtest_file)

    if results is None:
        print("Failed to load backtest results. Exiting.")
        return

    # Run analyses
    check_backtest_debug_info(results)
    analyze_signal_statistics(results)

    # Try different confidence thresholds
    for threshold in [0.6, 0.55, 0.5]:
        simulate_trading_with_threshold(results, confidence_threshold=threshold)

    diagnose_backtest_issues(results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
