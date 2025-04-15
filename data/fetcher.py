import datetime as dt
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import MetaTrader5 as mt5
import pandas as pd
import yaml
from tqdm import tqdm
import yfinance as yf

from config.constants import TimeFrame
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger


class MT5DataFetcher:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        self.config = self._load_config(str(config_path))
        self.initialized = False
        self.scalers = {}
        self.indicators = TechnicalIndicators()
        self.logger = setup_logger(name="MT5DataFetcher")

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def initialize(self) -> bool:
        if not mt5.initialize(
                login=self.config["mt5"]["login"],
                server=self.config["mt5"]["server"],
                password=self.config["mt5"]["password"],
                timeout=self.config["mt5"]["timeout"]
        ):
            error_code = mt5.last_error()
            if error_code[0] == -2:
                msg = "Check your MT5 server/login/password configuration in config.yaml"
            elif error_code[0] == -3:
                msg = "MetaTrader 5 terminal not running. Please start it first."
            else:
                msg = ""
            self.logger.error(f"MT5 initialization failed. Error code: {error_code}. {msg}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to connect to trading account. Check your credentials.")
            mt5.shutdown()
            return False

        self.initialized = True
        self.logger.info(f"Connected to MT5 server: {self.config['mt5']['server']}")
        self.logger.info(f"Account: {account_info.login}, Balance: {account_info.balance}")
        return True

    def shutdown(self) -> None:
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            self.logger.info("MT5 connection closed.")

    def get_timeframe(self, tf_str: str) -> int:
        return TimeFrame[tf_str].value

    def validate_symbol(self, symbol: str) -> bool:
        if not self.initialized and not self.initialize():
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found in MT5. Check symbol name.")
            return False

        if not symbol_info.visible:
            self.logger.info(f"Symbol {symbol} not visible in Market Watch. Adding it...")
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to add {symbol} to Market Watch")
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
        if not self.initialized and not self.initialize():
            raise ConnectionError("Failed to connect to MT5 terminal")
        if not self.validate_symbol(symbol):
            raise ValueError(f"Symbol {symbol} is not available")

        if isinstance(timeframe, str):
            timeframe = self.get_timeframe(timeframe)

        if end_date is None:
            end_date = dt.datetime.now()
        if start_date is None and num_bars is None:
            lookback_days = self.config["data"]["lookback_days"]
            start_date = end_date - dt.timedelta(days=lookback_days)

        try:
            if num_bars:
                # Use copy_rates_from_pos to fetch the latest num_bars
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
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

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        self.logger.info(f"Fetched {len(df)} bars for {symbol} on timeframe {timeframe}")
        return df

    def check_connection_params(self) -> Tuple[bool, str]:
        params = {
            "server": self.config["mt5"]["server"],
            "login": self.config["mt5"]["login"],
            "password": self.config["mt5"]["password"]
        }
        missing = [k for k, v in params.items() if not v]
        if missing:
            return False, f"Missing MT5 connection parameters: {', '.join(missing)}. Update config.yaml."
        return True, "Connection parameters valid"

    def fetch_all_timeframes(
            self,
            symbol: Optional[str] = None,
            lookback_days: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        if symbol is None:
            symbol = self.config["data"]["symbol"]
        if lookback_days is None:
            lookback_days = self.config["data"]["lookback_days"]

        valid, msg = self.check_connection_params()
        if not valid:
            self.logger.error(msg)
            return {}

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)

        data_dict = {}
        timeframes = self.config["data"]["timeframes"]

        if not self.initialized and not self.initialize():
            self.logger.error("Failed to initialize MT5 connection. Check your settings.")
            return {}
        if not self.validate_symbol(symbol):
            self.logger.error(f"Symbol {symbol} is not available in your MT5 terminal.")
            return {}

        for tf in tqdm(timeframes, desc="Fetching timeframes"):
            try:
                self.logger.info(f"Fetching {symbol} data for {tf} timeframe...")
                data_dict[tf] = self.fetch_data(symbol, tf, start_date, end_date)
                self.logger.info(f" → {len(data_dict[tf])} bars retrieved for {tf}")
            except Exception as e:
                self.logger.error(f"Error fetching {tf} data: {str(e)}")
        if not data_dict and self.initialized:
            self.shutdown()
        return data_dict

    def fetch_external_data(self) -> Dict[str, pd.DataFrame]:
        external_data = {}
        ext_sources = self.config.get("features", {}).get("external_data", [])
        if not ext_sources:
            self.logger.warning("No external data sources configured.")
            return external_data

        for source in ext_sources:
            self.logger.info(f"Fetching external data: {source['name']} (symbol: {source['symbol']})")
            lookback_days = self.config["data"].get("lookback_days", 365)
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=int(lookback_days * 1.1))
            try:
                data = yf.download(
                    source["symbol"],
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if data.empty:
                    self.logger.error(
                        f"No external data returned for {source['name']} (symbol: {source['symbol']}). Check symbol and date range.")
                else:
                    external_data[source["name"]] = data
                    self.logger.info(f" → Retrieved {len(data)} rows for {source['name']}")
            except Exception as e:
                self.logger.error(f"Error fetching external data for {source['name']}: {str(e)}")
        return external_data

    def save_data(self, data_dict: Dict[str, pd.DataFrame], symbol: Optional[str] = None) -> None:
        if symbol is None:
            symbol = self.config["data"]["symbol"]

        save_path = Path(self.config["data"]["save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for tf, df in data_dict.items():
            filename = save_path / f"{symbol}_{tf}_{timestamp}.csv"
            df.to_csv(filename)
            self.logger.info(f"Saved {len(df)} rows to {filename}")
        self.logger.info(f"All MT5 data saved to {save_path}")

    def save_external_data(self, external_data: Dict[str, pd.DataFrame]) -> None:
        save_path = Path(self.config["data"]["save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, df in external_data.items():
            filename = save_path / f"{name}_{timestamp}.csv"
            df.to_csv(filename)
            self.logger.info(f"Saved external data {name}: {len(df)} rows to {filename}")
        self.logger.info(f"All external data saved to {save_path}")

    def get_latest_data_paths(self) -> Dict[str, str]:
        save_path = Path(self.config["data"]["save_path"])
        if not save_path.exists():
            return {}
        latest_files = {}
        timeframes = self.config["data"]["timeframes"]
        for tf in timeframes:
            pattern = f"{self.config['data']['symbol']}_{tf}_*.csv"
            matching = list(save_path.glob(pattern))
            if matching:
                latest = max(matching, key=lambda p: p.stat().st_ctime)
                latest_files[tf] = str(latest)
        return latest_files

    def get_latest_external_data_paths(self) -> Dict[str, str]:
        save_path = Path(self.config["data"]["save_path"])
        if not save_path.exists():
            return {}
        latest_files = {}
        for source in self.config.get("features", {}).get("external_data", []):
            name = source["name"]
            pattern = f"{name}_*.csv"
            matching = list(save_path.glob(pattern))
            if matching:
                latest = max(matching, key=lambda p: p.stat().st_ctime)
                latest_files[name] = str(latest)
        return latest_files

    def load_data_from_paths(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for tf, path in data_paths.items():
            p = Path(path)
            if p.exists():
                try:
                    data_dict[tf] = pd.read_csv(str(p), index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded {tf} data: {len(data_dict[tf])} rows")
                except Exception as e:
                    self.logger.error(f"Error loading {path}: {str(e)}")
            else:
                self.logger.warning(f"File not found: {path}")
        return data_dict

    # Alias load_data so external modules expecting this name work correctly.
    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        return self.load_data_from_paths(data_paths)


def main():
    fetcher = MT5DataFetcher()
    if fetcher.initialize():
        try:
            mt5_data = fetcher.fetch_all_timeframes()
            fetcher.save_data(mt5_data)
            ext_data = fetcher.fetch_external_data()
            fetcher.save_external_data(ext_data)
            latest_mt5 = fetcher.get_latest_data_paths()
            latest_ext = fetcher.get_latest_external_data_paths()
            fetcher.logger.info("Latest MT5 data files:")
            for tf, path in latest_mt5.items():
                fetcher.logger.info(f"  {tf}: {path}")
            fetcher.logger.info("Latest external data files:")
            for name, path in latest_ext.items():
                fetcher.logger.info(f"  {name}: {path}")
        finally:
            fetcher.shutdown()


if __name__ == "__main__":
    main()
