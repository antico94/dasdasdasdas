import os
from models.evaluator import backtest_model_horizon_aware
from data.storage import DataStorage
import yaml


def run_horizon_backtest():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set backtest parameters
    model_path = "models/ensemble_H1_direction_12.joblib"
    timeframe = "H1"
    initial_balance = 10000.0
    risk_per_trade = 0.02

    # Run the horizon-aware backtest
    metrics, results, trades = backtest_model_horizon_aware(
        config,
        model_path,
        timeframe,
        initial_balance,
        risk_per_trade
    )

    # Print results
    print("=================")
    print("Status: Success")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Balance: ${metrics['initial_balance']:.2f}")
    print(f"Final Balance: ${metrics['final_balance']:.2f}")
    print(f"Total Return: {metrics['return_pct']:.2f}%")
    print(f"Number of Trades: {metrics['n_trades']}")

    if metrics['n_trades'] > 0:
        print(f"Win Rate: {metrics['win_rate']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    # Return results for further analysis
    return metrics, results, trades


if __name__ == "__main__":
    run_horizon_backtest()