"""
Constants and configuration values for the AI trading bot.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- MetaTrader 5 Configuration ---
ACCOUNT = int(os.getenv("MT5_ACCOUNT", "79745631"))
PASSWORD = os.getenv("MT5_PASSWORD", "Qaz123,./")
SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial8")
BOT_MAGIC_NUMBER = 345678

# --- API Keys ---
XAI_API_KEY = os.getenv("XAI_API_KEY", "xai-1OAfUX6PBW0fK9fx1s8sGk1uXhIzxUwluqZz1JlVHBjj55W8hK8T23aAMflaCsjtJPw8ps3NMvpQWR8u")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7602903955:AAH4fJlI2OoK7FsaCt6UxB3KA5NwgFg4KTs")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-4677734700")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Model Identifiers ---
XAI_MODEL = "grok-3-fast-beta"
MODEL_GROK = "grok/grok-3-fast-beta"
MODEL_GPT4O_G4F = "gpt-4o"
MODEL_GEMINI = "gemini-2.0-flash"
MODEL_DEEPSEEK = "deepseek/deepseek-chat-v3-0324:free"
MODEL_META = "meta-llama/llama-4-maverick:free"
MODEL_NEMOTRON = "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
MODEL_REKA = "google/gemma-3-12b-it:free"
MODEL_NEMOTRON_SUPER = "nvidia/llama-3.3-nemotron-super-49b-v1:free"
MODEL_QWEN = "qwen/qwen-2.5-7b-instruct:free"

# --- AI Models Configuration ---
AI_MODELS_TO_CALL = {
    "GROK": {"type": "xai", "model_name": XAI_MODEL},
    "GPT4O": {"type": "g4f", "model_name": MODEL_GPT4O_G4F},
    "DEEPSEEK": {"type": "openrouter", "model_name": MODEL_DEEPSEEK},
    "GEMINI": {"type": "gemini", "model_name": MODEL_GEMINI},
    "NEMOTRON": {"type": "openrouter", "model_name": MODEL_NEMOTRON},
    "REKA": {"type": "openrouter", "model_name": MODEL_REKA},
    "META": {"type": "openrouter", "model_name": MODEL_META},
    "NEMOTRON_SUPER": {"type": "openrouter", "model_name": MODEL_NEMOTRON_SUPER},
    "QWEN_INSTRUCT": {"type": "openrouter", "model_name": MODEL_QWEN},
}

# --- Trading Symbols Configuration ---
SYMBOL_SETTINGS = {
    "XAUUSD": {
        "volume": 0.01,
        "max_orders": 0,
        "sl_pips": 30.0,
        "breakeven_pips": 10.0,
        "be_target_profit_pips": 5.0,
        "trailing_trigger_pips": 15.0,
        "trailing_distance_pips": 15,
        "pip_definition_in_points": 10
    },
    "BTCUSD": {
        "volume": 0.01,
        "max_orders": 1,
        "sl_pips": 300.0,
        "breakeven_pips": 120.0,
        "be_target_profit_pips": 20.0,
        "trailing_trigger_pips": 170.0,
        "trailing_distance_pips": 50.0,
        "pip_definition_in_points": 100
    }
}

SCALPING_SYMBOLS = list(SYMBOL_SETTINGS.keys())

# --- Trading Parameters ---
SCALPING_ENTRY_INTERVAL = 90
MANAGE_POSITIONS_INTERVAL = 4

# --- Technical Indicators Parameters ---
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 20
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
BBANDS_PERIOD = 20
BBANDS_STDDEV = 2.0
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# --- Data Parameters ---
MIN_DATA_ROWS_FOR_ANALYSIS = 20
MIN_DATA_ROWS_AFTER_INITIAL_DROP = 50
DEFAULT_CANDLES_TO_REQUEST = 300
ADDITIONAL_LOOKBACK_BUFFER = 200
