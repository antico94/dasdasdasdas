import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import MetaTrader5 as mt5
import pandas as pd
import yaml
from tqdm import tqdm

from config.constants import TimeFrame


class MT5DataFetcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.initialized = False

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def initialize(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal."""
        if not mt5.initialize(
                login=self.config["mt5"]["login"],
                server=self.config["mt5"]["server"],
                password=self.config["mt5"]["password"],
                timeout=self.config["mt5"]["timeout"]
        ):
            error_code = mt5.last_error()
            error_details = ""

            if error_code[0] == -2:
                error_details = "Check your MT5 server/login/password configuration in config.yaml"
            elif error_code[0] == -3:
                error_details = "MetaTrader 5 terminal not running. Please start it first."

            print(f"MT5 initialization failed. Error code: {error_code}. {error_details}")
            return False

        # Verify account connection
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to connect to trading account. Check your credentials.")
            mt5.shutdown()
            return False

        self.initialized = True
        print(f"Connected to MT5 server: {self.config['mt5']['server']}")
        print(f"Account: {account_info.login}, Balance: {account_info.balance}")
        return True

    def shutdown(self) -> None:
        """Shutdown connection to MetaTrader 5 terminal."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            print("MT5 connection closed.")

    def get_timeframe(self, tf_str: str) -> int:
        """Convert string timeframe to MT5 timeframe constant."""
        return TimeFrame[tf_str].value

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in MT5."""
        if not self.initialized and not self.initialize():
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found in MT5. Check symbol name.")
            return False

        if not symbol_info.visible:
            print(f"Symbol {symbol} is not visible in Market Watch. Adding it...")
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to add {symbol} to Market Watch")
                return False

        return True

    def fetch_data(
            self,
            symbol: str,
            timeframe: Union[str, int],
            start_date: Optional[dt.datetime] = None,
            end_date: Optional[dt.datetime] = None,
            num_bars: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical price data from MT5."""
        if not self.initialized and not self.initialize():
            raise ConnectionError("Failed to connect to MT5 terminal")

        # Validate symbol
        if not self.validate_symbol(symbol):
            raise ValueError(f"Symbol {symbol} is not available")

        # Convert string timeframe to MT5 timeframe constant if needed
        if isinstance(timeframe, str):
            timeframe = self.get_timeframe(timeframe)

        # Set default dates if not provided
        if end_date is None:
            end_date = dt.datetime.now()

        if start_date is None and num_bars is None:
            # Default to config lookback period
            lookback_days = self.config["data"]["lookback_days"]
            start_date = end_date - dt.timedelta(days=lookback_days)

        # Fetch the historical data
        try:
            if num_bars:
                rates = mt5.copy_rates_from(symbol, timeframe, end_date, num_bars)
            else:
                rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        except Exception as e:
            error_code = mt5.last_error()
            raise ValueError(f"Failed to fetch data for {symbol}. Error: {str(e)}. MT5 error: {error_code}")

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            if error_code[0] == 0:
                raise ValueError(f"No data returned for {symbol}. Check date range or timeframe.")
            else:
                raise ValueError(f"Failed to fetch data for {symbol}. Error code: {error_code}")

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        return df

    def check_connection_params(self) -> Tuple[bool, str]:
        """Check if connection parameters are properly configured."""
        required_params = {
            "server": self.config["mt5"]["server"],
            "login": self.config["mt5"]["login"],
            "password": self.config["mt5"]["password"]
        }

        missing = [k for k, v in required_params.items() if not v]

        if missing:
            return False, f"Missing MT5 connection parameters: {', '.join(missing)}. Update config.yaml."

        return True, "Connection parameters valid"

    def fetch_all_timeframes(
            self,
            symbol: Optional[str] = None,
            lookback_days: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured timeframes."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        if lookback_days is None:
            lookback_days = self.config["data"]["lookback_days"]

        # Check connection parameters before attempting connection
        valid, message = self.check_connection_params()
        if not valid:
            print(message)
            return {}

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)

        data_dict = {}
        timeframes = self.config["data"]["timeframes"]

        # Validate symbol first
        if not self.initialized:
            if not self.initialize():
                print("Failed to initialize MT5 connection. Check your settings.")
                return {}

        if not self.validate_symbol(symbol):
            print(f"Symbol {symbol} is not available in your MT5 terminal.")
            return {}

        for tf in tqdm(timeframes, desc="Fetching timeframes"):
            try:
                print(f"Fetching {symbol} data for {tf} timeframe...")
                data_dict[tf] = self.fetch_data(symbol, tf, start_date, end_date)
                print(f"  → {len(data_dict[tf])} bars retrieved")
            except Exception as e:
                print(f"Error fetching {tf} data: {str(e)}")

        # Close connection if no data retrieved
        if not data_dict and self.initialized:
            self.shutdown()

        return data_dict

    def fetch_external_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch external data sources defined in config."""
        import yfinance as yf

        external_data = {}
        for data_source in self.config["features"]["external_data"]:
            if data_source["source"] == "yfinance":
                print(f"Fetching external data: {data_source['name']}...")
                lookback_days = self.config["data"]["lookback_days"]
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=lookback_days * 1.1)  # Add buffer

                try:
                    data = yf.download(
                        data_source["symbol"],
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    external_data[data_source["name"]] = data
                    print(f"  → {len(data)} rows retrieved for {data_source['name']}")
                except Exception as e:
                    print(f"Error fetching {data_source['name']}: {str(e)}")

        return external_data

    def save_data(self, data_dict: Dict[str, pd.DataFrame], symbol: Optional[str] = None) -> None:
        """Save fetched data to disk."""
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        save_path = self.config["data"]["save_path"]
        os.makedirs(save_path, exist_ok=True)

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save MT5 data
        for timeframe, df in data_dict.items():
            filename = f"{save_path}/{symbol}_{timeframe}_{timestamp}.csv"
            df.to_csv(filename)
            print(f"Saved {len(df)} rows to {filename}")

        print(f"All data saved to {save_path}")

    def save_external_data(self, external_data: Dict[str, pd.DataFrame]) -> None:
        """Save external data sources to disk."""
        save_path = self.config["data"]["save_path"]
        os.makedirs(save_path, exist_ok=True)

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, df in external_data.items():
            filename = f"{save_path}/{name}_{timestamp}.csv"
            df.to_csv(filename)
            print(f"Saved external data {name}: {len(df)} rows to {filename}")

        print(f"All external data saved to {save_path}")

    def get_latest_data_paths(self) -> Dict[str, str]:
        """Get paths to the most recent data files for each timeframe."""
        save_path = self.config["data"]["save_path"]
        symbol = self.config["data"]["symbol"]

        if not os.path.exists(save_path):
            return {}

        latest_files = {}
        timeframes = self.config["data"]["timeframes"]

        for tf in timeframes:
            pattern = f"{symbol}_{tf}_*.csv"
            matching_files = list(Path(save_path).glob(pattern))
            if matching_files:
                latest_file = max(matching_files, key=os.path.getctime)
                latest_files[tf] = str(latest_file)

        return latest_files

    def get_latest_external_data_paths(self) -> Dict[str, str]:
        """Get paths to the most recent external data files."""
        save_path = self.config["data"]["save_path"]

        if not os.path.exists(save_path):
            return {}

        latest_files = {}
        for data_source in self.config["features"]["external_data"]:
            name = data_source["name"]
            pattern = f"{name}_*.csv"
            matching_files = list(Path(save_path).glob(pattern))
            if matching_files:
                latest_file = max(matching_files, key=os.path.getctime)
                latest_files[name] = str(latest_file)

        return latest_files

    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        data_dict = {}
        for timeframe, path in data_paths.items():
            if os.path.exists(path):
                try:
                    data_dict[timeframe] = pd.read_csv(path, index_col=0, parse_dates=True)
                    print(f"Loaded {timeframe} data: {len(data_dict[timeframe])} rows")
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
            else:
                print(f"Warning: File not found - {path}")

        return data_dict


def main():
    """Test function to fetch and save data."""
    fetcher = MT5DataFetcher()
    if fetcher.initialize():
        try:
            # Fetch MT5 data
            data_dict = fetcher.fetch_all_timeframes()
            fetcher.save_data(data_dict)

            # Fetch external data
            external_data = fetcher.fetch_external_data()
            fetcher.save_external_data(external_data)

            # Get latest data paths
            latest_mt5_data = fetcher.get_latest_data_paths()
            latest_external_data = fetcher.get_latest_external_data_paths()

            print("\nLatest MT5 data files:")
            for tf, path in latest_mt5_data.items():
                print(f"  {tf}: {path}")

            print("\nLatest external data files:")
            for name, path in latest_external_data.items():
                print(f"  {name}: {path}")

        finally:
            fetcher.shutdown()


if __name__ == "__main__":
    main()