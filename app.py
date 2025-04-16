import os

import pandas as pd
import yaml
import joblib

from config.constants import AppMode
from data.fetcher import MT5DataFetcher
from data.processor import DataProcessor
from data.storage import DataStorage
from models.trainer import train_model_pipeline
from ui.cli import TradingBotCLI
from utils.logger import setup_logger

# Initialize a unified logger for the application
logger = setup_logger("App.py - Logger")


class TradingBotApp:
    def __init__(self):
        self.cli = TradingBotCLI()
        self.data_storage = DataStorage()
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_root, "config", "config.yaml")

        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            logger.info("Loaded configuration from %s", config_path)
        except Exception as e:
            logger.error("Failed to load configuration: %s", str(e))
            self.config = {}
        logger.info("TradingBotApp initialized.")

    def run(self):
        """Main application entry point."""
        logger.info("Application started. Entering main loop.")
        while True:
            action = self.cli.main_menu()
            logger.debug("User selected action: %s", action)

            if action == 'exit':
                logger.info("Exiting application.")
                print("Exiting...")
                break
            elif action == AppMode.FETCH_DATA.value:
                self._handle_fetch_data()
            elif action == AppMode.PROCESS_DATA.value:
                self._handle_process_data()
            elif action == AppMode.TRAIN_MODEL.value:
                self._handle_train_model()
            elif action == AppMode.BACKTEST.value:
                self._handle_backtest()
            elif action == AppMode.LIVE_TRADE.value:
                self._handle_live_trade()
            elif action == AppMode.OPTIMIZE.value:
                self._handle_optimize()
            elif action == AppMode.VISUALIZE.value:
                self._handle_visualize()

    def _handle_fetch_data(self):
        """Handle data fetching action."""
        options = self.cli.fetch_data_menu()
        logger.info("Fetch data options: %s", options)

        if not options['use_mt5']:
            self.cli.show_results("Error", {"Status": "MT5 connection required for data fetching"})
            return

        if options.get('update_mt5_config', False):
            self._update_mt5_config(
                options['mt5_server'],
                int(options['mt5_login']),
                options['mt5_password']
            )

        if not self.cli.confirm_action("fetch data from MetaTrader 5"):
            logger.info("User cancelled MT5 data fetching.")
            return

        try:
            fetcher = MT5DataFetcher()
            if fetcher.initialize():
                try:
                    fetcher.config["data"]["timeframes"] = options['timeframes']
                    fetcher.config["data"]["lookback_days"] = options['lookback_days']
                    self.cli.show_progress("Fetching and splitting price data", options['lookback_days'])

                    # Use the new function to fetch, split and save data
                    paths = fetcher.fetch_split_and_save()

                    if not paths:
                        self.cli.show_results("Error",
                                              {"Status": "No data retrieved. Check MT5 connection and symbol."})
                        return

                    if options['fetch_external']:
                        self.cli.show_progress("Fetching external data", 100)
                        external_data = fetcher.fetch_external_data()
                        fetcher.save_external_data(external_data)

                    # Display results
                    results = {"Status": "Success", "Data split and saved to:":
                        f"Train: {next(iter(paths.get('train', {}).values()), 'None')}"}

                    # Add more details about the splits
                    for split_type in ['train', 'validation', 'test']:
                        if split_type in paths and paths[split_type]:
                            for tf, path in paths[split_type].items():
                                if tf in options['timeframes']:
                                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                                    results[f"{split_type.capitalize()} {tf} rows"] = len(df)

                    self.cli.show_results("Data Fetching Complete", results)
                finally:
                    fetcher.shutdown()
            else:
                self.cli.show_results("Error",
                                      {"Status": "Failed to connect to MT5. Check your settings in config.yaml"})
        except Exception as e:
            logger.exception("Error during data fetching: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _update_mt5_config(self, server: str, login: int, password: str):
        """Update MT5 configuration in config.yaml file."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_root, "config", "config.yaml")
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            config["mt5"]["server"] = server
            config["mt5"]["login"] = login
            config["mt5"]["password"] = password
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)
            logger.info("Updated MT5 configuration in %s", config_path)
        except Exception as e:
            logger.error("Failed to update MT5 config: %s", str(e))

    def _handle_process_data(self):
        """Handle data processing action."""
        options = self.cli.process_data_menu()
        logger.info("Process data options: %s", options)

        if not self.cli.confirm_action("process the historical data"):
            logger.info("User cancelled data processing.")
            return

        try:
            # Find the latest split data
            storage = DataStorage()
            split_paths = storage.find_latest_split_data()

            if not split_paths or not any(split_paths.values()):
                self.cli.show_results("Error", {"Status": "No split data found. Please fetch data first."})
                return

            # Find latest external data
            latest_external_data_paths = {}
            for data_source in ["usd_index", "us_10y_yield"]:
                path = storage.find_latest_file(f"{data_source}_*.csv")
                if path:
                    latest_external_data_paths[data_source] = path

            processor = DataProcessor()
            processor.config["model"]["prediction_target"] = options['target_type']
            processor.config["model"]["prediction_horizon"] = options['prediction_horizon']

            self.cli.show_progress("Loading data", 50)

            # Load the split data
            split_data = processor.load_split_data(split_paths)
            external_data = processor.load_external_data(latest_external_data_paths)

            self.cli.show_progress("Processing data", 100)

            # Process each split separately
            processed_split_data = processor.process_split_data(split_data, external_data)

            if options['normalization'] != 'none':
                self.cli.show_progress("Normalizing data", 50)
                processed_split_data = processor.normalize_split_data(processed_split_data,
                                                                      method=options['normalization'])

            # Save the processed data
            saved_paths = processor.save_processed_split_data(processed_split_data)

            # Prepare results to display
            results = {
                "Status": "Success",
                "Target variable": options['target_type'],
                "Prediction horizon": options['prediction_horizon'],
                "Normalization": options['normalization']
            }

            # Add information about each split
            for split_name, timeframe_data in processed_split_data.items():
                for tf, df in timeframe_data.items():
                    results[f"{split_name.capitalize()} {tf} shape"] = f"{df.shape[0]} rows, {df.shape[1]} columns"

            self.cli.show_results("Data Processing Complete", results)
        except Exception as e:
            logger.exception("Error processing data: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _handle_train_model(self):
        """Handle model training action."""
        options = self.cli.train_model_menu()
        logger.info("Train model options: %s", options)
        if not self.cli.confirm_action("train a new model"):
            logger.info("User cancelled model training.")
            return

        try:
            # Find latest processed split data
            storage = DataStorage()
            split_paths = storage.find_latest_split_data()

            if not split_paths or "train" not in split_paths or not split_paths["train"]:
                self.cli.show_results("Error",
                                      {"Status": "No processed training data found. Run data processing first."})
                return

            # Update configuration based on user options
            if not hasattr(self, 'config') or not self.config:
                project_root = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(project_root, "config", "config.yaml")
                try:
                    with open(config_path, "r") as file:
                        self.config = yaml.safe_load(file)
                    logger.info("Loaded configuration from %s", config_path)
                except Exception as e:
                    logger.warning(f"Could not load config file: {str(e)}")
                    self.config = {}

            # Set model configuration options
            if 'model' not in self.config:
                self.config['model'] = {}
            self.config['model']['type'] = options['model_type']
            self.config['model']['hyperparameter_tuning'] = options['hyperparameter_tuning']
            self.config['model']['feature_selection'] = options['feature_selection']
            self.config['model']['cross_validation'] = options['cross_validation']
            self.config['model']['use_bayes_optimizer'] = True

            # Train the model using the updated training pipeline
            self.cli.show_progress("Training model", 100)
            model, metrics = train_model_pipeline(self.config, options['timeframe'])

            # Display results
            results = {
                "Status": "Success",
                "Model Type": options['model_type'],
                "Timeframe": options['timeframe']
            }

            if 'test' in metrics:
                test_metrics = metrics['test']
                if 'test_accuracy' in test_metrics:
                    results["Test Accuracy"] = f"{test_metrics['test_accuracy']:.4f}"
                    results["Test F1 Score"] = f"{test_metrics['test_f1']:.4f}"
                elif 'test_rmse' in test_metrics:
                    results["Test RMSE"] = f"{test_metrics['test_rmse']:.4f}"
                    results["Test R²"] = f"{test_metrics['test_r2']:.4f}"

            if 'final_test' in metrics:
                final_metrics = metrics['final_test']
                results["Final Test Accuracy"] = f"{final_metrics['test_accuracy']:.4f}"
                results["Final Test F1 Score"] = f"{final_metrics['test_f1']:.4f}"

            results["Feature Count"] = metrics.get('n_features', 0)
            results["Training Time"] = f"{metrics.get('training_time', 0):.2f} seconds"

            self.cli.show_results("Model Training Complete", results)

        except Exception as e:
            logger.exception("Error during model training: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _handle_backtest(self):
        """Handle backtesting action."""
        options = self.cli.backtest_menu()
        logger.info("Backtest options: %s", options)

        if options.get('model_file') is None:
            self.cli.show_results("Error", {"Status": "No models available for backtesting"})
            return

        if not self.cli.confirm_action("run backtest"):
            logger.info("User cancelled backtest.")
            return

        try:
            # Update config for backtest
            self.config = self.config if hasattr(self, 'config') else {}
            self.config['backtest'] = self.config.get('backtest', {})
            self.config['backtest']['initial_balance'] = options['initial_balance']
            self.config['risk'] = self.config.get('risk', {})
            self.config['risk']['risk_per_trade'] = options['risk_per_trade']
            self.config['backtest']['include_spread'] = options['include_spread']
            self.config['backtest']['include_slippage'] = options['include_slippage']

            # Set up strategy configuration
            self.config['strategy'] = self.config.get('strategy', {})
            self.config['strategy']['take_profit_pct'] = 1.0  # Default values
            self.config['strategy']['stop_loss_pct'] = 0.5
            self.config['strategy']['max_hold_hours'] = 12
            self.config['strategy']['use_trailing_stop'] = True
            self.config['strategy']['min_confidence'] = 0.7

            # Verify test data exists
            storage = DataStorage()
            split_paths = storage.find_latest_split_data()

            if "test" not in split_paths or options['timeframe'] not in split_paths["test"]:
                self.cli.show_results("Error", {
                    "Status": "No test data found. Please fetch and process data first.",
                    "Details": f"Missing test data for timeframe {options['timeframe']}"
                })
                return

            # Create strategy instance
            from strategy.strategies import StrategyFactory
            strategy = StrategyFactory.create_strategy('goldtrend', self.config)

            # Call the backtest function with our strategy
            from models.evaluator import backtest_model_with_strategy

            self.cli.show_progress("Running backtest", 100)
            metrics, results_df, trades = backtest_model_with_strategy(
                self.config,
                options['model_file'],
                options['timeframe'],
                strategy,
                options['initial_balance'],
                options['risk_per_trade'],
                options['include_spread'],
                options['include_slippage']
            )

            # Display results
            results = {
                "Status": "Success",
                "Model": os.path.basename(options['model_file']),
                "Strategy": strategy.name,
                "Timeframe": options['timeframe'],
                "Initial Balance": f"${options['initial_balance']:.2f}",
                "Final Balance": f"${metrics['final_balance']:.2f}",
                "Total Return": f"{metrics['return_pct']:.2f}%",
                "Number of Trades": metrics['n_trades']
            }

            # Add additional metrics if available
            if 'win_rate' in metrics:
                results["Win Rate"] = f"{metrics['win_rate']:.2f}"
            if 'profit_factor' in metrics:
                results["Profit Factor"] = f"{metrics['profit_factor']:.2f}"
            if 'max_drawdown_pct' in metrics:
                results["Max Drawdown"] = f"{metrics['max_drawdown_pct']:.2f}%"
            if 'sharpe_ratio' in metrics:
                results["Sharpe Ratio"] = f"{metrics['sharpe_ratio']:.2f}"

            # Get exit reasons if available
            strategy_stats = strategy.get_stats()
            if 'tracking' in strategy_stats and 'exit_reasons' in strategy_stats['tracking']:
                exit_reasons = strategy_stats['tracking']['exit_reasons']
                if exit_reasons:
                    reasons_str = ', '.join([f"{k}: {v}" for k, v in exit_reasons.items() if v > 0])
                    if reasons_str:
                        results["Exit Reasons"] = reasons_str

            self.cli.show_results("Backtest Complete", results)

            # Show visualizations if requested
            if self.cli.confirm_action("view detailed backtest visualizations"):
                from models.visualization import visualize_backtest_results
                storage = DataStorage()
                models_dir = os.path.join(storage.project_root, "data_output", "trained_models")
                backtest_files = [f for f in os.listdir(models_dir)
                                  if "_backtest_" in f and f.endswith('.pkl')]
                if backtest_files:
                    latest_backtest = max(backtest_files, key=lambda f: os.path.getctime(os.path.join(models_dir, f)))
                    backtest_path = os.path.join(models_dir, latest_backtest)
                    visualize_backtest_results(backtest_path)

        except Exception as e:
            logger.exception("Error during backtesting: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _handle_live_trade(self):
        """Handle live trading action."""
        self.cli.show_results("Not Implemented", {"Status": "Live trading will be implemented in Phase 3"})

    def _handle_optimize(self):
        """Handle hyperparameter optimization."""
        options = self.cli.train_model_menu()
        logger.info("Hyperparameter optimization options: %s", options)
        if not self.cli.confirm_action("optimize hyperparameters"):
            logger.info("User cancelled hyperparameter optimization.")
            return

        try:
            self.config = self.config if hasattr(self, 'config') else {}
            self.config['model'] = self.config.get('model', {})
            self.config['model']['type'] = options['model_type']
            from models.optimizer import optimize_hyperparameters
            self.cli.show_progress("Optimizing hyperparameters", 100)
            updated_config = optimize_hyperparameters(self.config, options['timeframe'])
            self.config = updated_config
            best_params = {}
            if 'hyperparameters' in updated_config.get('model', {}):
                model_type = options['model_type']
                if model_type in updated_config['model']['hyperparameters']:
                    best_params = updated_config['model']['hyperparameters'][model_type]
            results = {
                "Status": "Success",
                "Model Type": options['model_type'],
                "Timeframe": options['timeframe'],
                "Best Parameters": str(best_params)
            }
            self.cli.show_results("Hyperparameter Optimization Complete", results)
        except Exception as e:
            logger.exception("Error during hyperparameter optimization: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _handle_visualize(self):
        """Handle visualization of results."""
        from ui.cli import Choice
        vis_choice = self.cli.select(
            "Select visualization type:",
            choices=[
                Choice("Backtest Results", "backtest"),
                Choice("Feature Importance", "features"),
                Choice("Model Performance", "performance"),
                Choice("Optimization Results", "optimization")
            ]
        )
        storage = DataStorage()
        models_dir = os.path.join(storage.base_path, "../data_output/trained_models")
        if not os.path.exists(models_dir):
            self.cli.show_results("Error", {"Status": "No models directory found"})
            return

        if vis_choice == "backtest":
            backtest_files = [f for f in os.listdir(models_dir)
                              if "_backtest_" in f and f.endswith('.pkl')]
            if not backtest_files:
                self.cli.show_results("Error", {"Status": "No backtest results found"})
                return
            backtest_file = self.cli.select(
                "Select backtest file:",
                choices=[Choice(f, f) for f in backtest_files]
            )
            backtest_path = os.path.join(models_dir, backtest_file)
            from models.visualization import visualize_backtest_results
            visualize_backtest_results(backtest_path)
        elif vis_choice == "features":
            model_files = [f for f in os.listdir(models_dir)
                           if f.endswith('.joblib') and "_backtest_" not in f]
            if not model_files:
                self.cli.show_results("Error", {"Status": "No model files found"})
                return
            model_file = self.cli.select(
                "Select model file:",
                choices=[Choice(f, f) for f in model_files]
            )
            model_path = os.path.join(models_dir, model_file)
            from models.factory import ModelFactory
            from models.visualization import ModelVisualizer
            model = ModelFactory.load_model(model_path)
            visualizer = ModelVisualizer(self.config)
            visualizer.plot_feature_importance(
                model.get_feature_importance(),
                title=f"Feature Importance - {os.path.basename(model_path)}"
            )
        elif vis_choice == "performance":
            metrics_files = [f for f in os.listdir(models_dir)
                             if f.endswith('_metrics.pkl')]
            if not metrics_files:
                self.cli.show_results("Error", {"Status": "No model metrics found"})
                return
            metrics_file = self.cli.select(
                "Select metrics file:",
                choices=[Choice(f, f) for f in metrics_files]
            )
            metrics_path = os.path.join(models_dir, metrics_file)
            metrics = storage.load_results(metrics_path)
            results = {"Model": os.path.basename(metrics_path).replace('_metrics.pkl', '')}
            for dataset in ['train', 'val', 'test']:
                if dataset in metrics:
                    dataset_metrics = metrics[dataset]
                    for k, v in dataset_metrics.items():
                        if isinstance(v, (int, float)):
                            results[k] = f"{v:.4f}" if isinstance(v, float) else v
            self.cli.show_results("Model Performance Metrics", results)
        elif vis_choice == "optimization":
            opt_files = [f for f in os.listdir(models_dir)
                         if f.endswith('_optimization.pkl')]
            if not opt_files:
                self.cli.show_results("Error", {"Status": "No optimization results found"})
                return
            opt_file = self.cli.select(
                "Select optimization file:",
                choices=[Choice(f, f) for f in opt_files]
            )
            opt_path = os.path.join(models_dir, opt_file)
            opt_results = storage.load_results(opt_path)
            from models.visualization import ModelVisualizer
            visualizer = ModelVisualizer(self.config)
            visualizer.plot_optimization_results(
                opt_results,
                title=f"Optimization Results - {os.path.basename(opt_path)}"
            )


if __name__ == "__main__":
    app = TradingBotApp()
    app.run()
