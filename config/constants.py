from enum import Enum
import MetaTrader5 as mt5


class TimeFrame(Enum):
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1


class MarketSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"


class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    CUSTOM = "custom"


class ModelType(Enum):
    RANDOM_FOREST = "rf"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    GRU = "gru"
    ENSEMBLE = "ensemble"


class PredictionTarget(Enum):
    DIRECTION = "direction"
    PRICE = "price"
    RETURN = "return"
    VOLATILITY = "volatility"


class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class AppMode(Enum):
    FETCH_DATA = "fetch_data"
    PROCESS_DATA = "process_data"
    TRAIN_MODEL = "train_model"
    BACKTEST = "backtest"
    LIVE_TRADE = "live_trade"
    OPTIMIZE = "optimize"
    VISUALIZE = "visualize"


# XAUUSD-specific constants
XAUUSD_POINT_VALUE = 0.01
DEFAULT_VOLUME = 0.01
SPREAD_TYPICAL = 30  # Typical spread in points for XAUUSD
SLIPPAGE_TYPICAL = 5  # Typical slippage in points for XAUUSD

# Trading session times (UTC)
ASIAN_SESSION = {"start": "22:00", "end": "08:00"}
LONDON_SESSION = {"start": "08:00", "end": "16:00"}
NEW_YORK_SESSION = {"start": "13:00", "end": "22:00"}

# Economic indicators that impact gold
GOLD_IMPACT_INDICATORS = [
    "Fed Interest Rate Decision",
    "US Non-Farm Payrolls",
    "US CPI",
    "US GDP",
    "ECB Interest Rate Decision"
]

# Feature importance thresholds
MIN_FEATURE_IMPORTANCE = 0.02
