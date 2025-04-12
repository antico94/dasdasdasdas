import os
import sys
from typing import Dict, List

import questionary
from questionary import Choice, Separator

from config.constants import AppMode


class TradingBotCLI:
    def __init__(self):
        self.main_choices = [
            Choice("Fetch Data", AppMode.FETCH_DATA.value),
            Choice("Process Data", AppMode.PROCESS_DATA.value),
            Choice("Train Model", AppMode.TRAIN_MODEL.value),
            Choice("Backtest Strategy", AppMode.BACKTEST.value),
            Choice("Live Trading", AppMode.LIVE_TRADE.value),
            Separator(),
            Choice("Optimize Hyperparameters", AppMode.OPTIMIZE.value),
            Choice("Visualize Results", AppMode.VISUALIZE.value),
            Separator(),
            Choice("Exit", "exit")
        ]

    def _clear_screen(self) -> None:
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_banner(self) -> None:
        """Display banner for the trading bot."""
        self._clear_screen()
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗  ██╗ █████╗ ██╗   ██╗██╗   ██╗███████╗██████╗             ║
║   ╚██╗██╔╝██╔══██╗██║   ██║██║   ██║██╔════╝██╔══██╗            ║
║    ╚███╔╝ ███████║██║   ██║██║   ██║███████╗██║  ██║            ║
║    ██╔██╗ ██╔══██║██║   ██║██║   ██║╚════██║██║  ██║            ║
║   ██╔╝ ██╗██║  ██║╚██████╔╝╚██████╔╝███████║██████╔╝            ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚═════╝             ║
║                                                                  ║
║   XAUUSD Trading Bot with Machine Learning                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def main_menu(self) -> str:
        """Show main menu and return selected choice."""
        self.show_banner()
        return questionary.select(
            'Select an action:',
            choices=self.main_choices
        ).ask() or 'exit'

    def fetch_data_menu(self) -> Dict:
        """Menu for data fetching options."""
        default_config = {
            'use_mt5': True,
            'lookback_days': 365,
            'timeframes': ['M5', 'M15', 'H1', 'D1'],
            'fetch_external': True
        }

        print("\nCurrent Configuration:")
        print(f"- MetaTrader 5 connection: {'Yes' if default_config['use_mt5'] else 'No'}")
        print(f"- Lookback period: {default_config['lookback_days']} days")
        print(f"- Timeframes: {', '.join(default_config['timeframes'])}")
        print(f"- Fetch external data: {'Yes' if default_config['fetch_external'] else 'No'}")

        use_defaults = questionary.select(
            'How would you like to proceed?',
            choices=[
                Choice('Start fetching with current configuration', 'use_defaults'),
                Choice('Change configuration settings', 'change_config')
            ]
        ).ask()

        if use_defaults == 'use_defaults':
            return default_config

        # Custom configuration
        results = {}

        results['use_mt5'] = questionary.confirm(
            'Do you want to connect to MetaTrader 5?',
            default=default_config['use_mt5']
        ).ask()

        if results['use_mt5']:
            # Check if MT5 config needs to be updated
            from data.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            valid, message = fetcher.check_connection_params()

            if not valid:
                if questionary.confirm(
                        f"{message}\nDo you want to update MT5 connection settings now?",
                        default=True
                ).ask():
                    results['update_mt5_config'] = True
                    results['mt5_server'] = questionary.text(
                        'Enter MT5 server:',
                        validate=lambda val: len(val) > 0
                    ).ask()

                    results['mt5_login'] = questionary.text(
                        'Enter MT5 login:',
                        validate=lambda val: val.isdigit() and int(val) > 0
                    ).ask()

                    results['mt5_password'] = questionary.password(
                        'Enter MT5 password:',
                        validate=lambda val: len(val) > 0
                    ).ask()
                else:
                    results['update_mt5_config'] = False

        results['lookback_days'] = int(questionary.text(
            'How many days of historical data to fetch?',
            default=str(default_config['lookback_days']),
            validate=lambda val: val.isdigit() and int(val) > 0
        ).ask())

        results['timeframes'] = questionary.checkbox(
            'Select timeframes to fetch:',
            choices=[
                Choice('M5 (5 minutes)', 'M5', checked='M5' in default_config['timeframes']),
                Choice('M15 (15 minutes)', 'M15', checked='M15' in default_config['timeframes']),
                Choice('H1 (1 hour)', 'H1', checked='H1' in default_config['timeframes']),
                Choice('D1 (Daily)', 'D1', checked='D1' in default_config['timeframes']),
            ],
            validate=lambda answer: len(answer) > 0
        ).ask()

        results['fetch_external'] = questionary.confirm(
            'Fetch external data (USD index, interest rates, etc.)?',
            default=default_config['fetch_external']
        ).ask()

        return results

    def process_data_menu(self) -> Dict:
        """Menu for data processing options."""
        default_config = {
            'indicator_types': ['trend', 'momentum', 'volatility', 'volume', 'support_resistance', 'custom'],
            'normalization': 'standard',
            'target_type': 'direction',
            'prediction_horizon': 12,
            'train_test_split': 0.8
        }

        print("\nCurrent Configuration:")
        print(f"- Indicator types: {', '.join(default_config['indicator_types'])}")
        print(f"- Normalization: {default_config['normalization']}")
        print(f"- Target variable: {default_config['target_type']}")
        print(f"- Prediction horizon: {default_config['prediction_horizon']} periods")
        print(
            f"- Train/test split: {int(default_config['train_test_split'] * 100)}%/{int((1 - default_config['train_test_split']) * 100)}%")

        use_defaults = questionary.select(
            'How would you like to proceed?',
            choices=[
                Choice('Process data with current configuration', 'use_defaults'),
                Choice('Change configuration settings', 'change_config')
            ]
        ).ask()

        if use_defaults == 'use_defaults':
            return default_config

        # Custom configuration
        results = {}

        results['indicator_types'] = questionary.checkbox(
            'Select indicator types to include:',
            choices=[
                Choice('Trend Indicators', 'trend', checked='trend' in default_config['indicator_types']),
                Choice('Momentum Indicators', 'momentum', checked='momentum' in default_config['indicator_types']),
                Choice('Volatility Indicators', 'volatility',
                       checked='volatility' in default_config['indicator_types']),
                Choice('Volume Indicators', 'volume', checked='volume' in default_config['indicator_types']),
                Choice('Support/Resistance Levels', 'support_resistance',
                       checked='support_resistance' in default_config['indicator_types']),
                Choice('Custom Gold-specific Indicators', 'custom',
                       checked='custom' in default_config['indicator_types']),
            ],
            validate=lambda answer: len(answer) > 0
        ).ask()

        results['normalization'] = questionary.select(
            'Select normalization method:',
            choices=[
                Choice('Standardization (zero mean, unit variance)', 'standard'),
                Choice('Min-Max Scaling (-1 to 1)', 'minmax'),
                Choice('No normalization', 'none'),
            ],
            default=default_config['normalization']
        ).ask()

        results['target_type'] = questionary.select(
            'Select target variable type:',
            choices=[
                Choice('Price Direction (binary classification)', 'direction'),
                Choice('Future Return (regression)', 'return'),
                Choice('Future Price (regression)', 'price'),
                Choice('Volatility (binary classification)', 'volatility'),
            ],
            default=default_config['target_type']
        ).ask()

        results['prediction_horizon'] = int(questionary.text(
            'Prediction horizon (number of periods ahead):',
            default=str(default_config['prediction_horizon']),
            validate=lambda val: val.isdigit() and int(val) > 0
        ).ask())

        results['train_test_split'] = float(questionary.text(
            'Train/test split ratio (0-1):',
            default=str(default_config['train_test_split']),
            validate=lambda val: (val.replace('.', '', 1).isdigit() and 0 < float(val) < 1)
        ).ask())

        return results

    def train_model_menu(self) -> Dict:
        """Menu for model training options."""
        default_config = {
            'model_type': 'ensemble',
            'timeframe': 'H1',
            'hyperparameter_tuning': True,
            'feature_selection': True,
            'cross_validation': True
        }

        print("\nCurrent Configuration:")
        print(f"- Model type: {default_config['model_type']}")
        print(f"- Timeframe: {default_config['timeframe']}")
        print(f"- Hyperparameter tuning: {'Yes' if default_config['hyperparameter_tuning'] else 'No'}")
        print(f"- Feature selection: {'Yes' if default_config['feature_selection'] else 'No'}")
        print(f"- Time-series cross-validation: {'Yes' if default_config['cross_validation'] else 'No'}")

        use_defaults = questionary.select(
            'How would you like to proceed?',
            choices=[
                Choice('Train model with current configuration', 'use_defaults'),
                Choice('Change configuration settings', 'change_config')
            ]
        ).ask()

        if use_defaults == 'use_defaults':
            return default_config

        # Custom configuration
        results = {}

        results['model_type'] = questionary.select(
            'Select model type:',
            choices=[
                Choice('Random Forest', 'rf'),
                Choice('XGBoost', 'xgboost'),
                Choice('LSTM Neural Network', 'lstm'),
                Choice('Ensemble (Multiple Models)', 'ensemble'),
            ],
            default=default_config['model_type']
        ).ask()

        results['timeframe'] = questionary.select(
            'Select timeframe for training:',
            choices=[
                Choice('M5 (5 minutes)', 'M5'),
                Choice('M15 (15 minutes)', 'M15'),
                Choice('H1 (1 hour)', 'H1'),
                Choice('D1 (Daily)', 'D1'),
            ],
            default=default_config['timeframe']
        ).ask()

        results['hyperparameter_tuning'] = questionary.confirm(
            'Perform hyperparameter tuning?',
            default=default_config['hyperparameter_tuning']
        ).ask()

        results['feature_selection'] = questionary.confirm(
            'Perform feature selection?',
            default=default_config['feature_selection']
        ).ask()

        results['cross_validation'] = questionary.confirm(
            'Use time-series cross-validation?',
            default=default_config['cross_validation']
        ).ask()

        return results

    def backtest_menu(self) -> Dict:
        """Menu for backtesting options."""
        default_config = {
            'timeframe': 'H1',
            'initial_balance': 10000,
            'risk_per_trade': 0.02,
            'include_spread': True,
            'include_slippage': True
        }

        model_choices = self._get_available_models()
        if not model_choices:
            print("\nNo models available for backtesting. Please train a model first.")
            # Return complete config with None for model_file
            default_config['model_file'] = None
            return default_config

        default_config['model_file'] = model_choices[0].value

        print("\nCurrent Configuration:")
        print(
            f"- Model: {os.path.basename(default_config['model_file']) if default_config.get('model_file') else 'None'}")
        print(f"- Timeframe: {default_config['timeframe']}")
        print(f"- Initial balance: ${default_config['initial_balance']}")
        print(f"- Risk per trade: {int(default_config['risk_per_trade'] * 100)}%")
        print(f"- Include spread costs: {'Yes' if default_config['include_spread'] else 'No'}")
        print(f"- Include slippage: {'Yes' if default_config['include_slippage'] else 'No'}")

        use_defaults = questionary.select(
            'How would you like to proceed?',
            choices=[
                Choice('Run backtest with current configuration', 'use_defaults'),
                Choice('Change configuration settings', 'change_config')
            ]
        ).ask()

        if use_defaults == 'use_defaults':
            return default_config

        # Custom configuration
        results = {}

        results['model_file'] = questionary.select(
            'Select model for backtesting:',
            choices=model_choices
        ).ask()

        results['timeframe'] = questionary.select(
            'Select timeframe for backtesting:',
            choices=[
                Choice('M5 (5 minutes)', 'M5'),
                Choice('M15 (15 minutes)', 'M15'),
                Choice('H1 (1 hour)', 'H1'),
                Choice('D1 (Daily)', 'D1'),
            ],
            default=default_config['timeframe']
        ).ask()

        results['initial_balance'] = float(questionary.text(
            'Initial account balance (USD):',
            default=str(int(default_config['initial_balance'])),
            validate=lambda val: val.isdigit() and int(val) > 0
        ).ask())

        risk_per_trade = float(questionary.text(
            'Risk per trade (% of balance):',
            default=str(int(default_config['risk_per_trade'] * 100)),
            validate=lambda val: (val.replace('.', '', 1).isdigit() and 0 < float(val) < 100)
        ).ask())
        results['risk_per_trade'] = risk_per_trade / 100

        results['include_spread'] = questionary.confirm(
            'Include spread costs in backtest?',
            default=default_config['include_spread']
        ).ask()

        results['include_slippage'] = questionary.confirm(
            'Include slippage in backtest?',
            default=default_config['include_slippage']
        ).ask()

        return results

    def live_trading_menu(self) -> Dict:
        """Menu for live trading options."""
        default_config = {
            'timeframe': 'H1',
            'risk_per_trade': 0.01,
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'confirm_trading': False
        }

        model_choices = self._get_available_models()
        if not model_choices or (len(model_choices) == 1 and model_choices[0].value is None):
            return {'model_file': None, 'confirm_trading': False}

        default_config['model_file'] = model_choices[0].value

        print("\nCurrent Configuration:")
        print(
            f"- Model: {os.path.basename(default_config['model_file']) if default_config.get('model_file') else 'None'}")
        print(f"- Timeframe: {default_config['timeframe']}")
        print(f"- Risk per trade: {int(default_config['risk_per_trade'] * 100)}%")
        print(f"- Stop loss: {int(default_config['stop_loss'] * 100)}%")
        print(f"- Take profit: {int(default_config['take_profit'] * 100)}%")

        use_defaults = questionary.select(
            'How would you like to proceed?',
            choices=[
                Choice('Start trading with current configuration', 'use_defaults'),
                Choice('Change configuration settings', 'change_config')
            ]
        ).ask()

        if use_defaults == 'use_defaults':
            default_config['confirm_trading'] = questionary.confirm(
                'WARNING: This will execute real trades with real money. Continue?',
                default=False
            ).ask()
            return default_config

        # Custom configuration
        results = {}

        results['model_file'] = questionary.select(
            'Select model for live trading:',
            choices=model_choices
        ).ask()

        results['timeframe'] = questionary.select(
            'Select trading timeframe:',
            choices=[
                Choice('M5 (5 minutes)', 'M5'),
                Choice('M15 (15 minutes)', 'M15'),
                Choice('H1 (1 hour)', 'H1'),
                Choice('D1 (Daily)', 'D1'),
            ],
            default=default_config['timeframe']
        ).ask()

        risk_per_trade = float(questionary.text(
            'Risk per trade (% of balance):',
            default=str(int(default_config['risk_per_trade'] * 100)),
            validate=lambda val: (val.replace('.', '', 1).isdigit() and 0 < float(val) < 100)
        ).ask())
        results['risk_per_trade'] = risk_per_trade / 100

        stop_loss = float(questionary.text(
            'Stop loss (% of price):',
            default=str(int(default_config['stop_loss'] * 100)),
            validate=lambda val: (val.replace('.', '', 1).isdigit() and 0 < float(val) < 100)
        ).ask())
        results['stop_loss'] = stop_loss / 100

        take_profit = float(questionary.text(
            'Take profit (% of price):',
            default=str(int(default_config['take_profit'] * 100)),
            validate=lambda val: (val.replace('.', '', 1).isdigit() and 0 < float(val) < 100)
        ).ask())
        results['take_profit'] = take_profit / 100

        results['confirm_trading'] = questionary.confirm(
            'WARNING: This will execute real trades with real money. Continue?',
            default=False
        ).ask()

        return results

    import os
    from questionary import Choice

    def _get_available_models(self) -> List[Choice]:
        """Get list of available trained models."""
        # Determine the project root based on the location of this file.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")

        if not os.path.exists(models_dir):
            return []  # Return empty list if no models directory is found

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

        if not model_files:
            return []  # Return empty list if no model files are present

        return [Choice(f, os.path.join(models_dir, f)) for f in model_files]

    def confirm_action(self, action: str) -> bool:
        """Confirm an action with the user."""
        return questionary.confirm(
            f'Are you sure you want to {action}?',
            default=True
        ).ask()

    def select(self, message: str, choices: List, default=None) -> str:
        """Generic select method for questionary."""
        return questionary.select(
            message,
            choices=choices,
            default=default
        ).ask()

    def show_progress(self, message: str, total: int = 100) -> None:
        """Show progress message."""
        from tqdm import tqdm
        import time

        with tqdm(total=total, desc=message) as pbar:
            for i in range(total):
                time.sleep(0.01)  # Simulate work
                pbar.update(1)

    def show_results(self, title: str, results: Dict) -> None:
        """Display results to the user."""
        self._clear_screen()
        print(f"\n{title}\n{'=' * len(title)}")

        for key, value in results.items():
            print(f"{key}: {value}")

        input("\nPress Enter to continue...")


def main():
    """Test CLI functionality."""
    cli = TradingBotCLI()

    while True:
        action = cli.main_menu()

        if action == 'exit':
            print("Exiting...")
            break

        elif action == AppMode.FETCH_DATA.value:
            options = cli.fetch_data_menu()
            print("Fetch data options:", options)

        elif action == AppMode.PROCESS_DATA.value:
            options = cli.process_data_menu()
            print("Process data options:", options)

        elif action == AppMode.TRAIN_MODEL.value:
            options = cli.train_model_menu()
            print("Train model options:", options)

        elif action == AppMode.BACKTEST.value:
            options = cli.backtest_menu()
            print("Backtest options:", options)

        elif action == AppMode.LIVE_TRADE.value:
            options = cli.live_trading_menu()
            print("Live trading options:", options)

        cli.show_progress("Processing")

        # Sample results
        results = {
            "Accuracy": "76.5%",
            "Profit Factor": 1.85,
            "Sharpe Ratio": 1.32,
            "Max Drawdown": "12.4%"
        }
        cli.show_results("Backtest Results", results)


if __name__ == "__main__":
    main()
