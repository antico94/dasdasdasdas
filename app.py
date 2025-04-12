import os
import sys
from typing import Dict, List, Optional

from config.constants import AppMode
from data.fetcher import MT5DataFetcher
from data.processor import DataProcessor
from data.storage import DataStorage
from ui.cli import TradingBotCLI
from utils.logger import setup_logger

# Initialize a unified logger for the application
logger = setup_logger("TradingBotAppLogger")


class TradingBotApp:
    def __init__(self):
        self.cli = TradingBotCLI()
        self.data_storage = DataStorage()
        # You might also want to load a global configuration here
        self.config = {}  # or load your config from a file if needed
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
                    self.cli.show_progress("Fetching price data", options['lookback_days'])
                    data_dict = fetcher.fetch_all_timeframes(lookback_days=options['lookback_days'])
                    if not data_dict:
                        self.cli.show_results("Error", {"Status": "No data retrieved. Check MT5 connection and symbol."})
                        return
                    fetcher.save_data(data_dict)

                    if options['fetch_external']:
                        self.cli.show_progress("Fetching external data", 100)
                        external_data = fetcher.fetch_external_data()
                        fetcher.save_external_data(external_data)

                    latest_mt5_data = fetcher.get_latest_data_paths()
                    results = {"Status": "Success", "Data saved to": fetcher.config["data"]["save_path"]}
                    for tf, path in latest_mt5_data.items():
                        if tf in options['timeframes']:
                            df = fetcher.load_data({tf: path})[tf]
                            results[f"{tf} rows"] = len(df)
                    self.cli.show_results("Data Fetching Complete", results)
                finally:
                    fetcher.shutdown()
            else:
                self.cli.show_results("Error", {"Status": "Failed to connect to MT5. Check your settings in config.yaml"})
        except Exception as e:
            logger.exception("Error during data fetching: %s", str(e))
            self.cli.show_results("Error", {"Status": "Failed", "Error": str(e)})

    def _update_mt5_config(self, server: str, login: int, password: str):
        """Update MT5 configuration in config.yaml file."""
        import yaml
        config_path = "config/config.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        config["mt5"]["server"] = server
        config["mt5"]["login"] = login
        config["mt5"]["password"] = password
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        logger.info("Updated MT5 configuration in %s", config_path)

    def _handle_process_data(self):
        """Handle data processing action."""
        options = self.cli.process_data_menu()
        logger.info("Process data options: %s", options)

        if not self.cli.confirm_action("process the historical data"):
            logger.info("User cancelled data processing.")
            return

        try:
            latest_mt5_data_paths = self.data_storage.find_latest_data()
            if not latest_mt5_data_paths:
                self.cli.show_results("Error", {"Status": "No data found. Please fetch data first."})
                return

            latest_external_data_paths = {}
            for data_source in ["usd_index", "us_10y_yield"]:
                path = self.data_storage.find_latest_file(f"{data_source}_*.csv")
                if path:
                    latest_external_data_paths[data_source] = path

            processor = DataProcessor()
            processor.config["model"]["prediction_target"] = options['target_type']
            processor.config["model"]["prediction_horizon"] = options['prediction_horizon']
            processor.config["data"]["split_ratio"] = options['train_test_split']

            self.cli.show_progress("Loading data", 50)
            data_dict = processor.load_data(latest_mt5_data_paths)
            external_data = processor.load_external_data(latest_external_data_paths)
            self.cli.show_progress("Processing data", 100)
            processed_data = processor.process_data(data_dict, external_data)

            if options['normalization'] != 'none':
                self.cli.show_progress("Normalizing data", 50)
                processed_data = processor.normalize_data(processed_data, method=options['normalization'])

            processor.save_processed_data(processed_data)
            results = {
                "Status": "Success",
                "Target variable": options['target_type'],
                "Prediction horizon": options['prediction_horizon'],
                "Normalization": options['normalization']
            }
            for tf, df in processed_data.items():
                results[f"{tf} shape"] = f"{df.shape[0]} rows, {df.shape[1]} columns"
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
            self.config = self.config if hasattr(self, 'config') else {}
            self.config['model'] = self.config.get('model', {})
            self.config['model']['type'] = options['model_type']
            self.config['model']['hyperparameter_tuning'] = options['hyperparameter_tuning']
            self.config['model']['feature_selection'] = options['feature_selection']
            self.config['model']['cross_validation'] = options['cross_validation']

            from models.trainer import train_model_pipeline
            self.cli.show_progress("Training model", 100)
            model, metrics = train_model_pipeline(self.config, options['timeframe'])
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
            # Update backtest-related config options
            self.config = self.config if hasattr(self, 'config') else {}
            self.config['backtest'] = self.config.get('backtest', {})
            self.config['backtest']['initial_balance'] = options['initial_balance']
            self.config['backtest']['position_size'] = options['risk_per_trade']
            self.config['backtest']['include_spread'] = options['include_spread']
            self.config['backtest']['include_slippage'] = options['include_slippage']

            from models.evaluator import backtest_model
            self.cli.show_progress("Running backtest", 100)
            # Load model, processed data, and perform backtest.
            metrics, results_df, trades = backtest_model(
                self.config,
                options['model_file'],
                options['timeframe'],
                options['initial_balance'],
                options['risk_per_trade'],
                options['include_spread'],
                options['include_slippage']
            )
            # Additional logging: check test data features from backtest metrics file
            logger.debug("Backtest metrics: %s", metrics)

            results = {
                "Status": "Success",
                "Model": os.path.basename(options['model_file']),
                "Timeframe": options['timeframe'],
                "Initial Balance": f"${options['initial_balance']:.2f}",
                "Final Balance": f"${metrics['final_balance']:.2f}",
                "Total Return": f"{metrics['return_pct']:.2f}%",
                "Number of Trades": metrics['n_trades'],
                "Win Rate": f"{metrics['win_rate']:.2f}",
                "Profit Factor": f"{metrics['profit_factor']:.2f}",
                "Max Drawdown": f"{metrics['max_drawdown_pct']:.2f}%",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}"
            }
            self.cli.show_results("Backtest Complete", results)

            if self.cli.confirm_action("view detailed backtest visualizations"):
                from models.visualization import visualize_backtest_results
                storage = DataStorage()
                models_dir = os.path.join(storage.base_path, "../models")
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
        models_dir = os.path.join(storage.base_path, "../models")
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
