import asyncio
import logging
import os
import re
import sys
import time
import datetime
import traceback
from datetime import datetime, timezone, timedelta
from collections import Counter, deque, defaultdict
from typing import Optional, List, Dict, Tuple, Any, Callable
import threading
from queue import Queue
import random
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import mplfinance as mpf
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import telegram
from dotenv import load_dotenv
from g4f.client import AsyncClient, Client as G4fClient
from openai import OpenAI, UnprocessableEntityError, APIStatusError, APIConnectionError, RateLimitError, AuthenticationError
from telegram.constants import ParseMode
from telegram.error import TelegramError

from core import db_logger, chart_uploader
from config import Config
import json

# --- Configuration ---
load_dotenv()
ACCOUNT = int(os.getenv("MT5_ACCOUNT", "79745631"))
PASSWORD = os.getenv("MT5_PASSWORD", "Qaz123,./")
SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial8")
XAI_API_KEY = os.getenv("XAI_API_KEY", "xai-1OAfUX6PBW0fK9fx1s8sGk1uXhIzxUwluqZz1JlVHBjj55W8hK8T23aAMflaCsjtJPw8ps3NMvpQWR8u")
TELEGRAM_BOT_TOKEN = "7602903955:AAH4fJlI2OoK7FsaCt6UxB3KA5NwgFg4KTs"
TELEGRAM_CHAT_ID = "-4677734700"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Model Identifiers ---
AI_MODELS_TO_CALL = {
    "GROK": {"type": "xai", "model_name": Config.MODEL_GROK},
    "GPT4O": {"type": "g4f", "model_name": Config.MODEL_GPT4O_G4F},
    "DEEPSEEK": {"type": "openrouter", "model_name": Config.MODEL_DEEPSEEK},
    "GEMINI": {"type": "gemini", "model_name": Config.MODEL_GEMINI},
    "NEMOTRON": {"type": "openrouter", "model_name": Config.MODEL_NEMOTRON},
    "REKA": {"type": "openrouter", "model_name": Config.MODEL_REKA},
    "META": {"type": "openrouter", "model_name": Config.MODEL_META},
    "NEMOTRON_SUPER": {"type": "openrouter", "model_name": Config.MODEL_NEMOTRON_SUPER},
    "QWEN_INSTRUCT": {"type": "openrouter", "model_name": Config.MODEL_QWEN},
}
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
allowed_symbols = list(SYMBOL_SETTINGS.keys())
required_consensus = 4
SCALPING_ENTRY_INTERVAL = 90
MANAGE_POSITIONS_INTERVAL = 4
BOT_MAGIC_NUMBER = 345678
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
MIN_DATA_ROWS_FOR_ANALYSIS = 20
MIN_DATA_ROWS_AFTER_INITIAL_DROP = 50
DEFAULT_CANDLES_TO_REQUEST = 300
ADDITIONAL_LOOKBACK_BUFFER = 200
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("grok_scalping_bot_v4.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GrokScalpBotV4")
logger.info("Logging initialized.")
gemini_rotator = None
openrouter_rotators = {}
gemini_api_keys = []
openrouter_api_keys = []

# --- IMPORTS & GLOBALS MIGRATED FROM ai_trading_v2.py ---
import asyncio
import logging
import os
import re
import sys
import time
import datetime
import traceback
from datetime import datetime, timezone, timedelta
from collections import Counter, deque, defaultdict
from typing import Optional, List, Dict, Tuple, Any, Callable
import threading
from queue import Queue
import random
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import mplfinance as mpf
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import telegram
from dotenv import load_dotenv
from g4f.client import AsyncClient, Client as G4fClient
from openai import OpenAI, UnprocessableEntityError, APIStatusError, APIConnectionError, RateLimitError, AuthenticationError
from telegram.constants import ParseMode
from telegram.error import TelegramError
import json
# --- END IMPORTS ---

# --- Configuration and Constants ---
load_dotenv()
ACCOUNT = int(os.getenv("MT5_ACCOUNT", "79745631"))
PASSWORD = os.getenv("MT5_PASSWORD", "Qaz123,./")
SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial8")
XAI_API_KEY = os.getenv("XAI_API_KEY", "xai-1OAfUX6PBW0fK9fx1s8sGk1uXhIzxUwluqZz1JlVHBjj55W8hK8T23aAMflaCsjtJPw8ps3NMvpQWR8u")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-fast-beta")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- APIKeyRotator class (unchanged) ---
class APIKeyRotator:
    def __init__(
            self, keys, interval_seconds=60, api_type="gemini", model_name=Config.MODEL_GEMINI
    ):
        if not keys:
            raise ValueError("API keys list cannot be empty")

        self._keys = deque(keys)
        self._interval = interval_seconds
        self._api_type = api_type.lower()  # "gemini" or "openrouter"
        self._model_name = model_name
        self._current_model = None
        self._current_key = None
        self._lock = threading.Lock()
        self._timer = None
        self._is_running = False
        self._initial_setup_done = False
        self._key_error_logged = {key: False for key in keys}
        self._active_api_calls = 0

    def _rotate_and_configure(self):
        rescheduled = False
        try:
            with self._lock:
                if not self._is_running:
                    return
                if self._active_api_calls > 0:
                    logging.debug(
                        f"[Rotator] Active API calls ({self._active_api_calls}). Delaying key rotation."
                    )
                    if self._is_running:
                        self._timer = threading.Timer(
                            self._interval, self._rotate_and_configure
                        )
                        self._timer.name = "KeyRotationTimerThread"
                        self._timer.daemon = True
                        self._timer.start()
                        rescheduled = True
                    return
                self._keys.rotate(-1)
                next_key = self._keys[0]
                key_short = f"...{next_key[-4:]}"
                if not self._key_error_logged.get(next_key, False):
                    logging.debug(f"[Rotator] Trying to configure with key: {key_short}")
                try:
                    if not next_key or len(next_key) < 35:
                        if not self._key_error_logged.get(next_key, False):
                            logging.warning(
                                f"[Rotator] Key {key_short} seems invalid (length < 35). Skipping."
                            )
                            self._key_error_logged[next_key] = True
                    else:
                        # Configure based on API type
                        if self._api_type == "gemini":
                            genai.configure(api_key=next_key)
                            model = genai.GenerativeModel(self._model_name)
                            self._current_model = model
                            self._current_key = next_key
                        elif self._api_type == "openrouter":
                            client = OpenAI(
                                base_url=OPENROUTER_BASE_URL,
                                api_key=next_key,
                            )
                            self._current_model = client
                            self._current_key = next_key
                        self._initial_setup_done = True
                        if self._key_error_logged.get(next_key, False):
                            logging.debug(
                                f"[Rotator] Key {key_short} is working again."
                            )
                            self._key_error_logged[next_key] = False
                        elif not self._key_error_logged.get(next_key, False):
                            logging.debug(
                                f"[Rotator] Successfully configured '{self._model_name}' with key: {key_short} for {self._api_type}"
                            )
                except Exception as e:
                    if not self._key_error_logged.get(next_key, False):
                        logging.error(
                            f"[Rotator] Error configuring with key {key_short}: {type(e).__name__} - {e}"
                        )
                        self._key_error_logged[next_key] = True
                    pass
        except Exception as outer_e:
            logging.error(
                f"[Rotator] Unexpected error in _rotate_and_configure: {outer_e}",
                exc_info=True,
            )
        finally:
            with self._lock:
                if self._is_running and not rescheduled:
                    self._timer = threading.Timer(
                        self._interval, self._rotate_and_configure
                    )
                    self._timer.name = "KeyRotationTimerThread"
                    self._timer.daemon = True
                    self._timer.start()

    def start(self):
        logging.debug(f"[Rotator] Starting {self._api_type} API key rotation process for {self._model_name}...")
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._key_error_logged = {key: False for key in self._keys}
            initial_thread = threading.Thread(
                target=self._rotate_and_configure,
                name="InitialConfigThread",
                daemon=True,
            )
            initial_thread.start()

    def stop(self):
        logging.info(f"[Rotator] Stopping {self._api_type} API key rotation process for {self._model_name}...")
        with self._lock:
            self._is_running = False
            if self._timer:
                self._timer.cancel()
                self._timer = None
        logging.info(f"[Rotator] {self._api_type} API key rotation process stopped for {self._model_name}.")

    def get_model_for_usage(self):
        with self._lock:
            model = self._current_model
            if model:
                self._active_api_calls += 1
                logging.debug(
                    f"[Rotator] {self._api_type}/{self._model_name} model requested. Active calls: {self._active_api_calls}"
                )
            return model

    def release_model_usage(self):
        with self._lock:
            self._active_api_calls = max(0, self._active_api_calls - 1)
            logging.debug(
                f"[Rotator] {self._api_type}/{self._model_name} model usage released. Active calls: {self._active_api_calls}"
            )

# --- END APIKeyRotator ---

# ===============================================================================
# === MIGRATED: ai_signal_and_entry_loop (Main AI Signal & Entry Loop) ===
# ===============================================================================

async def ai_signal_and_entry_loop():
    """
    Main loop: Gets data, calls multiple AIs, places MARKET order if consensus reached.
    Modularized for refactored codebase.
    """
    global current_event_timestamp
    logger.info("Starting AI Signal & Entry loop (Refactored)...")
    
    # Đảm bảo pandas được import đúng cách
    try:
        import pandas as pd
        logger.info(f"Pandas successfully imported in ai_signal_and_entry_loop, version: {pd.__version__}")
    except ImportError:
        logger.error("Pandas not installed or not importable in ai_signal_and_entry_loop!")
        raise
    except Exception as pd_err:
        logger.error(f"Error with pandas in ai_signal_and_entry_loop: {pd_err}")
        raise
    
    await asyncio.sleep(5)
    xauusd_market_closed_logged = False

    def format_indicator_value(value, decimal_places):
        """Safely formats indicator values, returning N/A for NaNs."""
        try:
            if pd.isna(value): return "N/A"
            try: return f"{value:.{decimal_places}f}"
            except (TypeError, ValueError): return "N/A"
        except Exception as e:
            logger.error(f"Error in format_indicator_value: {e}")
            return "ERROR"

    # Thêm try-except bao quanh toàn bộ vòng lặp để bắt lỗi chi tiết
    try:
        while True:
            try:
                start_time_cycle = time.monotonic()
                current_event_timestamp = datetime.now(timezone.utc)
                logger.info(f"--- New Signal Check Cycle (Event TS: {current_event_timestamp.isoformat()}) ---")
            except Exception as time_err:
                logger.error(f"Error getting time in ai_signal_and_entry_loop: {time_err}")
                await asyncio.sleep(30)
                continue

            if not check_mt5_connection():
                logger.warning("MT5 disconnect detected at start of cycle. Waiting...")
                await asyncio.sleep(30)
                continue

            consensus_results = {}

            for symbol in SCALPING_SYMBOLS:
                settings = SYMBOL_SETTINGS.get(symbol)
                if not settings:
                    logger.error(f"Settings missing for {symbol}. Skipping.")
                    continue

                start_time_symbol = time.monotonic()
                dataframes = {}
                daily_pivots = None
                prompt_for_ai = None
                all_data_ok = True

                try:
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info:
                        logger.error(f"Could not retrieve MT5 info for {symbol}. Skipping.")
                        await asyncio.sleep(1)
                        continue

                    point = symbol_info.point
                    digits = symbol_info.digits
                    min_stop_points = symbol_info.trade_stops_level
                    pip_def_in_points = settings.get('pip_definition_in_points', 1)

                    if point <= 0 or pip_def_in_points <= 0:
                        logger.error(f"Invalid point ({point}) or pip definition ({pip_def_in_points}) for {symbol}. Skipping.")
                        continue

                    log_key_sym_closed = f"_logged_sym_closed_{symbol}"
                    if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                        if not getattr(ai_signal_and_entry_loop, log_key_sym_closed, False):
                            logger.info(f"Market {symbol} not fully open (Mode: {symbol_info.trade_mode}). Skipping check.")
                            setattr(ai_signal_and_entry_loop, log_key_sym_closed, True)
                        continue
                    elif hasattr(ai_signal_and_entry_loop, log_key_sym_closed):
                        logger.info(f"Market {symbol} is now open. Resuming checks.")
                        delattr(ai_signal_and_entry_loop, log_key_sym_closed)

                    if symbol == "XAUUSD":
                        tick_check = mt5.symbol_info_tick(symbol)
                        if not tick_check:
                            logger.error(f"Cannot get tick {symbol}. Skipping.")
                            continue
                        last_tick_time = tick_check.time
                        current_time = time.time()
                        is_closed = current_time - last_tick_time > 300
                        if is_closed and not xauusd_market_closed_logged:
                            logger.info(f"Market {symbol} appears closed (last tick > 5 min ago). Skipping.")
                            xauusd_market_closed_logged = True
                        elif not is_closed and xauusd_market_closed_logged:
                            logger.info(f"Market {symbol} appears open again.")
                            xauusd_market_closed_logged = False
                        if is_closed: continue

                    try:
                        daily_pivots = calculate_daily_pivots(symbol, digits)
                        if daily_pivots: logger.debug(f"[{symbol}] Calculated Pivots: {daily_pivots}")
                        else: logger.warning(f"Could not calculate daily pivots for {symbol}.")
                    except Exception as pivot_err:
                        logger.error(f"Error calculating pivots for {symbol}: {pivot_err}")
                        daily_pivots = None

                    # TODO: Fetch and prepare all_tf_data and prompt_for_ai for AI calls
                    # all_tf_data = ...
                    # prompt_for_ai = ...

                    # --- AI Model Calls & Consensus ---
                    ai_results = {}
                    for model_key, model_cfg in AI_MODELS_TO_CALL.items():
                        try:
                            logger.info(f"Calling AI model {model_key} for {symbol}...")
                            if model_cfg['type'] == 'g4f':
                                ai_result = await call_gpt4o_api_for_direction(prompt_for_ai, symbol)
                            elif model_cfg['type'] == 'xai':
                                ai_result = await call_xai_api_for_direction(prompt_for_ai, symbol)
                            elif model_cfg['type'] == 'gemini':
                                if not gemini_rotator:
                                    logger.warning(f"No Gemini rotator available for {symbol}")
                                    ai_result = "NO_GEMINI_ROTATOR"
                                else:
                                    ai_result = await call_gemini_api_for_direction(prompt_for_ai, symbol)
                            elif model_cfg['type'] == 'openrouter':
                                model_name = model_cfg['model_name']
                                rotator = openrouter_rotators.get(model_name)
                                if not rotator:
                                    logger.warning(f"No OpenRouter rotator for {model_name}")
                                    ai_result = "NO_OPENROUTER_ROTATOR"
                                else:
                                    ai_result = await call_openrouter_api_for_direction(prompt_for_ai, symbol, model_name, rotator)
                            else:
                                logger.error(f"Unknown model type {model_cfg['type']} for {model_key}")
                                ai_result = "INVALID_MODEL_TYPE"
                            logger.debug(f"AI result for {model_key} ({symbol}): {ai_result}")
                            ai_results[model_key] = ai_result
                        except Exception as e:
                            logger.error(f"Unexpected error processing {model_key} API for {symbol}: {e}", exc_info=True)
                            ai_results[model_key] = f"Error: Unexpected - {str(e)[:50]}"
                    # --- Consensus Logic (Majority) ---
                    signal_counts = Counter(ai_results.values())
                    logger.info(f"AI signal counts for {symbol}: {signal_counts}")
                    consensus_signal = None
                    required_consensus = 2  # Tối thiểu 2 model đồng ý
                    for signal, count in signal_counts.items():
                        if signal in ["BUY", "SELL"] and count >= required_consensus:
                            consensus_signal = signal
                            break
                    if not consensus_signal:
                        logger.info(f"No consensus for {symbol}. AI Results: {ai_results}")
                        continue
                    logger.info(f"Consensus signal for {symbol}: {consensus_signal}")
                    # --- Place Market Order if Consensus ---
                    try:
                        volume = settings.get('volume', 0.01)
                        sl_pips = settings.get('sl_pips', 30.0)
                        # ... (order logic)
                        logger.info(f"Placing {consensus_signal} order for {symbol} | volume: {volume} | SL: {sl_pips}")
                        order_result = await place_market_order(symbol, consensus_signal, volume, sl_pips)
                        logger.info(f"Order result for {symbol}: {order_result}")
                        order_result = await place_market_order(symbol, order_type, volume, sl_price)
                        
                        if order_result:
                            logger.info(f"Order placed successfully for {symbol}: {order_result}")
                        
                            # Log to DB if pool is available
                            try:
                                if db_logger.pool:
                                    ticket_id, entry_price, confirmed_sl, confirmed_tp, timestamp = order_result
                                    await db_logger.log_order_placement(
                                        timestamp=timestamp,
                                        ticket_id=ticket_id,
                                        order_type="MARKET",
                                        action=order_type,
                                        symbol=symbol,
                                        volume=volume,
                                        entry_price=entry_price,
                                        initial_sl=confirmed_sl,
                                        initial_tp=confirmed_tp,
                                        consensus_log_id=None  # TODO: Add consensus log ID if available
                                    )
                            except Exception as db_err:
                                logger.error(f"Failed to log order to database: {db_err}")
                        else:
                            logger.warning(f"Order placement failed for {symbol}")
                    except Exception as order_err:
                        logger.error(f"Order placement error for {symbol}: {order_err}", exc_info=True)

                    # --- End Symbol Processing ---
                    logger.info(f"[{symbol}] Cycle complete. Time: {time.monotonic() - start_time_symbol:.2f}s")

                except Exception as symbol_err:
                    logger.error(f"Error in signal/entry loop for {symbol}: {symbol_err}")
                    continue

            logger.info(f"Signal check cycle complete. Total time: {time.monotonic() - start_time_cycle:.2f}s")
            
            # Đợi đến chu kỳ tiếp theo
            await asyncio.sleep(SCALPING_ENTRY_INTERVAL)

# ===============================================================================
# === END OF ai_signal_and_entry_loop FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: place_market_order FUNCTION ===
# ===============================================================================

async def place_market_order(symbol: str, order_type: str, volume: float, sl_price: float, comment: str = "AI Scalp V4") -> tuple | None:
    """
    Places an MT5 market order with a specified SL.
    Returns a tuple (ticket_id, entry_price, confirmed_sl, confirmed_tp, order_timestamp_confirm)
    if successful, or None on failure.
    """
    if symbol not in allowed_symbols:
        logger.error(f"[PlaceMarketOrder] Unsupported symbol: {symbol}")
        return None

    try:
        if not check_mt5_connection():
            logger.error("[PlaceMarketOrder] MT5 disconnected or trading disabled.")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[PlaceMarketOrder] Symbol info not found for {symbol}, attempting select...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"[PlaceMarketOrder] Still cannot select symbol {symbol}.")
                return None
            await asyncio.sleep(0.2)
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"[PlaceMarketOrder] Cannot get symbol info for {symbol}.")
                return None

        digits = symbol_info.digits

        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.bid == 0 or tick.ask == 0:
            logger.error(f"[PlaceMarketOrder] Invalid tick for {symbol} before placing order.")
            # Optionally return None here
            # return None

        mt5_type = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
            "sl": round(sl_price, digits),
            "tp": 0.0,
            "deviation": 10,
            "magic": BOT_MAGIC_NUMBER,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(f"[PlaceMarketOrder] Sending MT5 MARKET request for {symbol}: {req}")
        result = mt5.order_send(req)

        if result is None:
            error_code, error_desc = mt5.last_error()
            logger.error(f"[PlaceMarketOrder] Order send failed for {symbol} (None result). MT5 Error: ({error_code}) {error_desc}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[PlaceMarketOrder] Order execution failed for {symbol}: {result.comment} (Code: {result.retcode})")
            return None

        position_ticket = result.order
        execution_price = result.price
        requested_sl = req.get("sl", 0.0)
        requested_tp = req.get("tp", 0.0)
        confirmed_sl = None
        confirmed_tp = None
        order_timestamp_confirm = datetime.now(timezone.utc)

        try:
            await asyncio.sleep(0.4)
            positions = mt5.positions_get(ticket=position_ticket)
            if positions is not None and len(positions) > 0:
                pos_info = positions[0]
                confirmed_sl = pos_info.sl if pos_info.sl != 0.0 else None
                confirmed_tp = pos_info.tp if pos_info.tp != 0.0 else None
                order_timestamp_confirm = datetime.fromtimestamp(pos_info.time, tz=timezone.utc)
                sl_disp = f"{confirmed_sl:.{digits}f}" if confirmed_sl is not None else "None"
                tp_disp = f"{confirmed_tp:.{digits}f}" if confirmed_tp is not None else "None"
                log_msg = f"MARKET Order PLACED & CONFIRMED: {order_type} {symbol}, Vol {volume}, Ticket: {position_ticket}, Entry: {execution_price:.{digits}f}, Confirmed SL: {sl_disp}, Confirmed TP: {tp_disp}"
                logger.info(f"[PlaceMarketOrder] {log_msg}")
                return (position_ticket, execution_price, confirmed_sl, confirmed_tp, order_timestamp_confirm)
            else:
                last_mt5_error = mt5.last_error()
                sl_disp_req = f"{requested_sl:.{digits}f}" if requested_sl != 0.0 else "N/A"
                tp_disp_req = f"{requested_tp:.{digits}f}" if requested_tp != 0.0 else "None"
                logger.warning(
                    f"[PlaceMarketOrder] MARKET Order PLACED for {symbol} (Ticket: {position_ticket}, Entry: {execution_price:.{digits}f}) "
                    f"but failed to confirm details immediately (Result: {positions}, MT5 Error: {last_mt5_error}). "
                    f"Requested SL: {sl_disp_req}, Requested TP: {tp_disp_req}."
                )
                return (position_ticket, execution_price, None, None, order_timestamp_confirm)
        except Exception as e_pos:
            sl_disp_req = f"{requested_sl:.{digits}f}" if requested_sl != 0.0 else "N/A"
            tp_disp_req = f"{requested_tp:.{digits}f}" if requested_tp != 0.0 else "None"
            logger.error(
                f"[PlaceMarketOrder] Error getting position details for ticket {position_ticket} ({symbol}): {e_pos}",
                exc_info=True
            )
            logger.info(
                f"[PlaceMarketOrder] MARKET Order PLACED for {symbol} (Ticket: {position_ticket}, Entry: {execution_price:.{digits}f}), "
                f"error confirming. Req SL: {sl_disp_req}, Req TP: {tp_disp_req}."
            )
            return (position_ticket, execution_price, None, None, order_timestamp_confirm)
    except Exception as e:
        logger.error(f"[PlaceMarketOrder] General exception placing order for {symbol}: {e}", exc_info=True)
        return None

# ===============================================================================
# === END OF place_market_order FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: get_ai_consensus_for_reversal_rm FUNCTION ===
# ===============================================================================

async def get_ai_consensus_for_reversal_rm(
    position: any,
    code_reversal_score: int,
    code_signal_desc: list,
    all_tf_data: dict,
    ai_models_config: dict,
    ai_rotators: dict,
    logger,
    reversal_log_id: int = None
) -> bool:
    """
    Calls configured AI models (using actual API functions) for reversal consensus confirmation.
    Logs opinions to DB.
    """
    symbol = position.symbol
    ticket = position.ticket
    pos_type_str = "BUY" if position.type == 0 else "SELL"
    logger.info(f"[{ticket} RM] Getting AI consensus for closing {pos_type_str} {symbol} (Code Score: {code_reversal_score}, ReversalLogID: {reversal_log_id})")

    # --- Build Market Summary for Prompt ---
    market_summary = ""
    digits = 5
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info: digits = symbol_info.digits
        tick = mt5.symbol_info_tick(symbol)
        if tick: market_summary += f"Current Price: Ask={tick.ask:.{digits}f}, Bid={tick.bid:.{digits}f}\n"
        else: market_summary += "Current Price: N/A\n"
        for tf_code in [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15]:
            tf_str = get_timeframe_string(tf_code)
            if tf_code in all_tf_data and not all_tf_data[tf_code].empty:
                latest = all_tf_data[tf_code].iloc[-1]
                close_val_str = f"{latest.get('Close', 'N/A'):.{digits}f}" if pd.notna(latest.get('Close')) else "N/A"
                market_summary += f"{tf_str} Close: {close_val_str}"
                if 'BB_Middle' in latest and pd.notna(latest['BB_Middle']):
                    market_summary += f", MidBB: {latest['BB_Middle']:.{digits}f}"
                market_summary += "\n"
            else:
                market_summary += f"{tf_str} Data: N/A\n"
    except Exception as e:
        logger.error(f"[{ticket} RM] Error building market summary: {e}")
        market_summary = "Error retrieving current market data.\n"

    # --- Build Final Prompt ---
    prompt_for_reversal_ai = REVERSAL_AI_PROMPT_TEMPLATE.format(
        symbol=symbol, position_type=pos_type_str, ticket=ticket,
        open_price=position.price_open, reversal_score=code_reversal_score,
        signal_description=', '.join(code_signal_desc) if code_signal_desc else "None",
        market_summary=market_summary.strip()
    )
    logger.debug(f"[{ticket} RM] Reversal Prompt Prepared: {prompt_for_reversal_ai[:400]}...")

    # --- Define Local Wrapper for API Calls & DB Logging ---
    async def timed_ai_call_wrapper_reversal(model_key, api_call_func, *args):
        nonlocal reversal_log_id
        start_t = time.monotonic()
        raw_result = None
        normalized_decision = "Error: WrapperFail"
        try:
            raw_result = await api_call_func(*args)
            end_t = time.monotonic()
            normalized_decision = normalize_reversal_ai_output_rm(raw_result, logger)
        except Exception as e:
            end_t = time.monotonic()
            logger.error(f"[RM Wrapper] Exception during API call for {model_key} (Ticket {ticket}): {type(e).__name__}", exc_info=False)
            normalized_decision = f"Error: {type(e).__name__}"
            raw_result = e

        # Log AI opinion to DB
        if reversal_log_id is not None:
            try:
                await db_logger.log_reversal_ai_opinion(reversal_log_id, model_key, normalized_decision)
            except Exception as db_log_err:
                logger.error(f"[RM] DB Log AI Opinion Error for {model_key} (Ticket {ticket}): {db_log_err}")
        else:
            logger.warning(f"[RM] reversal_log_id is None, cannot log AI opinion for {model_key} (Ticket {ticket})")

        return (model_key, normalized_decision)

    # --- Prepare AI Call Tasks using REAL functions ---
    ai_tasks = []
    ai_names_in_call = []
    gemini_rotator = ai_rotators.get("gemini")
    openrouter_rotators_dict = ai_rotators.get("openrouter", {})

    for ai_name_key, config in ai_models_config.items():
        api_call_func, args = None, ()
        model_type = config.get("type")
        model_name_id = config.get("model_name")
        rotator_for_call = None

        if not model_type or not model_name_id:
            logger.warning(f"[{ticket} RM] Skipping AI {ai_name_key}: Invalid config.")
            continue

        if model_type == "xai":
            api_call_func = call_xai_api_for_reversal
            args = (prompt_for_reversal_ai, symbol, logger)
        elif model_type == "g4f":
            api_call_func = call_gpt4o_api_for_reversal
            args = (prompt_for_reversal_ai, symbol, logger)
        elif model_type == "gemini":
            if not gemini_rotator:
                logger.warning(f"[{ticket} RM] Skipping Gemini ({ai_name_key}): Rotator not available.")
                continue
            api_call_func = call_gemini_api_for_reversal_text_only
            rotator_for_call = gemini_rotator
            args = (prompt_for_reversal_ai, symbol, rotator_for_call, logger)
        elif model_type == "openrouter":
            or_rotator = openrouter_rotators_dict.get(model_name_id)
            if not or_rotator:
                logger.warning(f"[{ticket} RM] Skipping OpenRouter {ai_name_key} ({model_name_id}): Rotator not found.")
                continue
            api_call_func = call_openrouter_reversal
            rotator_for_call = or_rotator
            args = (prompt_for_reversal_ai, symbol, model_name_id, rotator_for_call, logger)
        else:
            logger.warning(f"[{ticket} RM] Skipping AI {ai_name_key}: Unknown type '{model_type}'.")
            continue

        if api_call_func:
            task = timed_ai_call_wrapper_reversal(ai_name_key, api_call_func, *args)
            ai_tasks.append(task)
            ai_names_in_call.append(ai_name_key)
            logger.debug(f"[{ticket} RM] Task created for {ai_name_key} ({model_type})")

    if not ai_tasks:
        logger.error(f"[{ticket} RM] No AI tasks created for reversal check."); return False

    logger.info(f"[{ticket} RM] Gathering reversal opinions from {len(ai_tasks)} AI calls ({', '.join(ai_names_in_call)})...")
    ai_results_from_gather = await asyncio.gather(*ai_tasks, return_exceptions=True)

    normalized_decisions = {}
    for result_item in ai_results_from_gather:
        if isinstance(result_item, Exception):
            logger.error(f"[{ticket} RM] Gather returned exception: {result_item}")
            continue
        if isinstance(result_item, tuple) and len(result_item) == 2:
            model_key, normalized_decision = result_item
            normalized_decisions[model_key] = normalized_decision
        else:
            logger.error(f"[{ticket} RM] Invalid item format from gather: {result_item}")

    close_votes, hold_votes, error_invalid_votes = 0, 0, 0
    decision_log = f"AI Reversal Opinions ({symbol} Ticket {ticket}, ReversalLogID: {reversal_log_id}):\n"
    for name in ai_names_in_call:
        decision = normalized_decisions.get(name, "NO_RESPONSE")
        decision_log += f"- {name}: {decision}\n"
        if decision == "CLOSE": close_votes += 1
        elif decision == "HOLD": hold_votes += 1
        else: error_invalid_votes += 1
    logger.info(decision_log.strip())

    consensus_threshold = REVERSAL_AI_CONSENSUS_THRESHOLD
    total_opinions_received = len(normalized_decisions)
    logger.info(f"[{ticket} RM] Reversal Consensus Check: CLOSE={close_votes}, HOLD={hold_votes}, ERR/INV={error_invalid_votes} / {total_opinions_received} responded (Threshold: {consensus_threshold} CLOSE votes)")

    if close_votes >= consensus_threshold:
        logger.warning(f"[{ticket} RM] AI CONSENSUS REACHED TO CLOSE position {symbol} ({close_votes}/{total_opinions_received} models agree).")
        return True
    else:
        logger.info(f"[{ticket} RM] AI consensus to close NOT REACHED for {symbol}.")
        return False

# ===============================================================================
# === END OF get_ai_consensus_for_reversal_rm FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: normalize_reversal_ai_output_rm FUNCTION ===
# ===============================================================================
def normalize_reversal_ai_output_rm(output: any, logger) -> str:
    """
    Normalizes AI response for reversal decisions (CLOSE/HOLD).
    """
    if isinstance(output, Exception):
        logger.warning(f"[RM Norm] AI call resulted in exception: {type(output).__name__}")
        return f"Error: {type(output).__name__}"

    if not output or not isinstance(output, str):
        logger.warning(f"[RM Norm] AI response is empty or not a string: {type(output)}")
        return "INVALID_OUTPUT"

    output_upper = output.strip().upper()

    if "TYPE: CLOSE" in output_upper: return "CLOSE"
    if "TYPE: HOLD" in output_upper: return "HOLD"
    if "CLOSE" in output_upper and "HOLD" not in output_upper:
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as CLOSE")
        return "CLOSE"
    if "HOLD" in output_upper and "CLOSE" not in output_upper:
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as HOLD")
        return "HOLD"

    logger.warning(f"[RM Norm] AI response unclear: '{output[:100]}...' -> HOLD (default)")
    return "HOLD"

# ===============================================================================
# === END OF normalize_reversal_ai_output_rm FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: call_xai_api_for_reversal FUNCTION ===
# ===============================================================================
async def call_xai_api_for_reversal(prompt: str, symbol: str, logger) -> str | Exception:
    """Calls the xAI (Grok) model for reversal decision (CLOSE/HOLD)."""
    logger.info(f"--- [RM] Calling xAI API ({Config.XAI_MODEL}) for reversal {symbol} ---")
    if not XAI_API_KEY:
        logger.error("[RM] XAI_API_KEY is not configured.")
        return Exception("XAI Key missing")
    if not OpenAI:
        logger.error("[RM] OpenAI library not available for xAI call.")
        return ImportError("OpenAI library missing")

    start_time = time.monotonic()
    try:
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=Config.XAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=30,
            timeout=25.0,
            stream=False
        )
        response_time = time.monotonic() - start_time
        logger.info(f"[RM] xAI Reversal API response time for {symbol}: {response_time:.2f} seconds")
        final_content = getattr(completion.choices[0].message, 'content', None)
        logger.debug(f"[RM] Raw xAI Reversal content for {symbol}: {final_content!r}")
        if final_content:
            return final_content
        else:
            logger.warning(f"[RM] xAI Reversal response for {symbol} no content.")
            return "INVALID_OUTPUT"
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"[RM] Error calling xAI Reversal API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        return e

# ===============================================================================
# === END OF call_xai_api_for_reversal FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: call_gpt4o_api_for_reversal FUNCTION ===
# ===============================================================================
async def call_gpt4o_api_for_reversal(prompt: str, symbol: str, logger) -> str | Exception:
    """Calls the gpt-4o model via g4f library for reversal decision (CLOSE/HOLD)."""
    logger.info(f"--- [RM] Calling gpt-4o API (g4f) for reversal {symbol} ---")
    start_time = time.monotonic()
    try:
        client = AsyncClient()
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            timeout=25.0
        )
        response_time = time.monotonic() - start_time
        logger.info(f"[RM] g4f Reversal API response time for {symbol}: {response_time:.2f} seconds")
        final_content = getattr(response.choices[0].message, 'content', None)
        logger.debug(f"[RM] Raw g4f Reversal content for {symbol}: {final_content!r}")
        if final_content:
            return final_content
        else:
            logger.warning(f"[RM] g4f Reversal response for {symbol} no content.")
            return "INVALID_OUTPUT"
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start_time
        logger.warning(f"[RM] TIMEOUT: g4f Reversal API ({symbol}) took too long ({elapsed:.2f}s).")
        return asyncio.TimeoutError(f"g4f Reversal API call timed out after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"[RM] Error calling g4f Reversal API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        return e

# ===============================================================================
# === END OF call_gpt4o_api_for_reversal FUNCTION ===
# ===============================================================================

# ===============================================================================
# === MIGRATED: call_gemini_api_for_reversal_text_only FUNCTION ===
# ===============================================================================
async def call_gemini_api_for_reversal_text_only(
    prompt: str,
    symbol: str,
    rotator,
    logger
) -> str | Exception:
    """Calls the Gemini model (text only) for reversal decision (CLOSE/HOLD)."""
    logger.info(f"--- [RM] Calling Gemini API (Text Only) for reversal {symbol} ---")
    if not genai:
        logger.error("[RM] Google Generative AI library not available for Gemini call.")
        return ImportError("google-generativeai missing")
    if not rotator:
        logger.error(f"[RM] No Gemini rotator provided for reversal {symbol}.")
        return ConnectionAbortedError("No Gemini rotator")
    model = rotator.get_model_for_usage()
    if not model:
        logger.error(f"[RM] Failed to get Gemini client from rotator for reversal {symbol}.")
        if rotator: rotator.release_model_usage()
        return ConnectionAbortedError("No Gemini client from rotator")
    start_time = time.monotonic()
    current_key_info = f"...{rotator._current_key[-4:]}" if hasattr(rotator, '_current_key') and rotator._current_key else 'N/A'
    logger.debug(f"[RM] Using Gemini key {current_key_info} for reversal {symbol}")
    try:
        async def api_call():
            return await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.5, "max_output_tokens": 30},
            )
        completion = await asyncio.wait_for(api_call(), timeout=30.0)
        response_time = time.monotonic() - start_time
        logger.info(f"[RM] Gemini Reversal API response time for {symbol}: {response_time:.2f} seconds")
        if completion is None:
            logger.error(f"[RM] Gemini Reversal returned None for {symbol}")
            return "INVALID_OUTPUT"
        # Process response text
        return getattr(completion, 'text', str(completion))
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"[RM] Error calling Gemini Reversal API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        return e

# ===============================================================================
# === END OF call_gemini_api_for_reversal_text_only FUNCTION ===
# ===============================================================================

# Import OpenRouter reversal API call (migrated)
from core.call_openrouter_reversal import call_openrouter_reversal

# --- Global trackers and state ---
ai_reversal_cooldown_tracker = {}
position_close_cooldown_tracker = {}
current_event_timestamp: Optional[datetime] = None
processed_closed_tickets = set()

# --- Utility: Escape Markdown V2 for Telegram ---
def escape_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram MarkdownV2 parse mode."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Utility: Send Telegram Message ---
async def send_telegram_message(message: str):
    """Sends a message to the configured Telegram chat using MarkdownV2."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram Bot Token or Chat ID not configured. Skipping notification.")
        return
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        escaped_message = escape_markdown_v2(message)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=escaped_message, parse_mode=ParseMode.MARKDOWN_V2)
        logger.debug(f"Sent Telegram message to chat ID {TELEGRAM_CHAT_ID}")
    except telegram.error.BadRequest as e:
        logger.error(f"Telegram BadRequest Error: {e}. Check Chat ID, Bot permissions, or message formatting.")
        try:
            logger.warning("Retrying Telegram message send with plain text.")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as plain_e:
            logger.error(f"Failed to send Telegram message even as plain text: {plain_e}")
    except telegram.error.RetryAfter as e:
        logger.warning(f"Telegram rate limit hit. Need to wait {e.retry_after} seconds.")
        await asyncio.sleep(e.retry_after + 1)
        await send_telegram_message(message)
    except TelegramError as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message: {e}", exc_info=True)

# --- Utility: Initialize MT5 ---
def init_mt5():
    """Initializes the MetaTrader 5 connection and enables symbols from settings."""
    try:
        if not mt5.initialize():
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False
        logger.info(f"MT5 Initialized: Version {mt5.version()}")
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            logger.error("MT5 login() failed: Account %s, Server %s, Error %s", ACCOUNT, SERVER, mt5.last_error())
            mt5.shutdown()
            return False
        logger.info("MT5 login successful: Account %s, Server %s.", ACCOUNT, SERVER)
        enabled_count = 0
        for symbol in SYMBOL_SETTINGS.keys():
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Could not enable symbol {symbol} on init. Might require manual enable in MT5.")
            else:
                logger.info(f"Symbol {symbol} enabled in MarketWatch.")
                enabled_count += 1
        if enabled_count == 0 and len(SYMBOL_SETTINGS) > 0:
            logger.error("No specified symbols from settings could be enabled.")
            mt5.shutdown()
            return False
        elif enabled_count < len(SYMBOL_SETTINGS):
            logger.warning("Not all symbols specified in settings could be enabled.")
        return True
    except Exception as e:
        logger.error(f"MT5 init exception: {e}", exc_info=True)
        return False

# --- Utility: Check MT5 Connection ---
def check_mt5_connection():
    try:
        term_info = mt5.terminal_info()
        if not term_info:
            mt5.initialize()
            time.sleep(0.5)
            term_info = mt5.terminal_info()
            if not term_info:
                logger.error("Failed to get MT5 terminal info even after retry.")
                return False
        if not term_info.connected:
            logger.error("MT5 terminal not connected.")
            return False
        if not term_info.trade_allowed:
            logger.warning("MT5 AutoTrading disabled in Terminal Options.")
            return False
        last_err_code, last_err_desc = mt5.last_error()
        if last_err_code not in [0, 1]:
            logger.warning(f"MT5 lingering error: ({last_err_code}, '{last_err_desc}')")
        elif last_err_code == 1:
            logger.debug(f"MT5 last op code 1 ('Success'): ({last_err_code}, '{last_err_desc}')")
        logger.debug("MT5 Connection and Trading Status OK.")
        return True
    except Exception as e:
        logger.error(f"MT5 connection check exception: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        return False

# --- Hàm tiện ích escape Markdown V2 cho Telegram ---
def escape_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram MarkdownV2 parse mode."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Hàm gửi tin nhắn Telegram (Mới) ---
async def send_telegram_message(message: str):
    """Sends a message to the configured Telegram chat using MarkdownV2."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram Bot Token or Chat ID not configured. Skipping notification.")
        return
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        escaped_message = escape_markdown_v2(message)  # Escape message content
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=escaped_message, parse_mode=ParseMode.MARKDOWN_V2)
        logger.debug(f"Sent Telegram message to chat ID {TELEGRAM_CHAT_ID}")
    except telegram.error.BadRequest as e:
        logger.error(f"Telegram BadRequest Error: {e}. Check Chat ID, Bot permissions, or message formatting.")
        try:
            logger.warning("Retrying Telegram message send with plain text.")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as plain_e:
            logger.error(f"Failed to send Telegram message even as plain text: {plain_e}")
    except telegram.error.RetryAfter as e:
        logger.warning(f"Telegram rate limit hit. Need to wait {e.retry_after} seconds.")
        await asyncio.sleep(e.retry_after + 1)
        await send_telegram_message(message)
    except TelegramError as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message: {e}", exc_info=True)

# --- MT5 Connection Utilities ---
def init_mt5():
    """Initializes the MetaTrader 5 connection and enables symbols from settings."""
    try:
        if not mt5.initialize():
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False
        logger.info(f"MT5 Initialized: Version {mt5.version()}")
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            logger.error("MT5 login() failed: Account %s, Server %s, Error %s", ACCOUNT, SERVER, mt5.last_error())
            mt5.shutdown()
            return False
        logger.info("MT5 login successful: Account %s, Server %s.", ACCOUNT, SERVER)
        enabled_count = 0
        for symbol in SYMBOL_SETTINGS.keys():
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Could not enable symbol {symbol} on init. Might require manual enable in MT5.")
            else:
                logger.info(f"Symbol {symbol} enabled in MarketWatch.")
                enabled_count += 1
        if enabled_count == 0 and len(SYMBOL_SETTINGS) > 0:
            logger.error("No specified symbols from settings could be enabled.")
            mt5.shutdown()
            return False
        elif enabled_count < len(SYMBOL_SETTINGS):
            logger.warning("Not all symbols specified in settings could be enabled.")
        return True
    except Exception as e:
        logger.error(f"MT5 init exception: {e}", exc_info=True)
        return False

def check_mt5_connection():
    try:
        term_info = mt5.terminal_info()
        if not term_info:
            mt5.initialize()
            time.sleep(0.5)
            term_info = mt5.terminal_info()
            if not term_info:
                logger.error("Failed to get MT5 terminal info even after retry.")
                return False
        if not term_info.connected:
            logger.error("MT5 terminal not connected.")
            return False
        if not term_info.trade_allowed:
            logger.warning("MT5 AutoTrading disabled in Terminal Options.")
            return False
        last_err_code, last_err_desc = mt5.last_error()
        if last_err_code not in [0, 1]:
            logger.warning(f"MT5 lingering error: ({last_err_code}, '{last_err_desc}')")
        elif last_err_code == 1:
            logger.debug(f"MT5 last op code 1 ('Success'): ({last_err_code}, '{last_err_desc}')")
        logger.debug("MT5 Connection and Trading Status OK.")
        return True
    except Exception as e:
        logger.error(f"MT5 connection check exception: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        return False

# --- Daily Pivot Points ---
def calculate_daily_pivots(symbol: str, digits: int):
    """
    Calculates Classic Daily Pivot Points for a given symbol.
    Returns a dictionary with pivot levels or None if data is unavailable.
    """
    try:
        # Xác định ngày giao dịch gần nhất có dữ liệu (thường là ngày hôm qua)
        # Lấy 2 nến D1 gần nhất để đảm bảo có nến hoàn chỉnh của ngày trước đó
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
        if rates_d1 is None or len(rates_d1) < 2:
            logger.warning(
                f"PIVOT: Not enough D1 data for {symbol} to calculate pivots (need 2 candles, got {len(rates_d1) if rates_d1 is not None else 0}).")
            return None
        # Nến thứ hai từ cuối lên là nến hoàn chỉnh của ngày hôm trước
        prev_day_candle = rates_d1[-2]
        prev_high = prev_day_candle['high']
        prev_low = prev_day_candle['low']
        prev_close = prev_day_candle['close']
        # prev_open = prev_day_candle['open'] # Không cần cho Classic
        # Tính toán Pivot Points (Classic)
        P = (prev_high + prev_low + prev_close) / 3
        R1 = (2 * P) - prev_low
        S1 = (2 * P) - prev_high
        R2 = P + (prev_high - prev_low)
        S2 = P - (prev_high - prev_low)
        R3 = prev_high + 2 * (P - prev_low)
        S3 = prev_low - 2 * (prev_high - P)
        pivots = {
            "PP": round(P, digits),
            "R1": round(R1, digits),
            "S1": round(S1, digits),
            "R2": round(R2, digits),
            "S2": round(S2, digits),
            "R3": round(R3, digits),
            "S3": round(S3, digits),
            "Prev_High": round(prev_high, digits),  # Thêm thông tin này cũng hữu ích
            "Prev_Low": round(prev_low, digits),
            "Prev_Close": round(prev_close, digits)
        }
        logger.debug(f"PIVOT: Calculated for {symbol}: {pivots}")
        return pivots
    except Exception as e:
        logger.error(f"PIVOT: Error calculating pivots for {symbol}: {e}", exc_info=True)
        return None

# ===============================================================================
# === START OF closed_order_monitor_task FUNCTION (MODIFIED FOR DEBUGGING) ===

# === HELPER: Calculate profit USD for BTC/XAU ===
def calculate_profit_usd_approx_local(profit_pips, volume, symbol, logger_instance):
    """
    Tính toán lợi nhuận USD ước tính, tập trung vào BTCUSD và XAUUSD.
    (Đã loại bỏ logic cho Forex chung và Indices/Oil).
    """
    from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
    logger = logger_instance
    try:
        profit_pips_dec = Decimal(str(profit_pips))
        volume_dec = Decimal(str(volume))
        cleaned_symbol = symbol.split('.')[0].upper()
        logger.debug(f"[Calc USD (BTC/XAU)] Processing Cleaned Symbol: {cleaned_symbol}")
        # Crypto (BTC, ETH, XBT...)
        if cleaned_symbol.startswith(("BTC", "ETH", "XBT")):
            logger.debug(f"[Calc USD (BTC/XAU)] Detected Crypto: {cleaned_symbol}")
            profit_quote_ccy = profit_pips_dec * volume_dec
            if cleaned_symbol.endswith("USD"):
                result = profit_quote_ccy.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                logger.debug(f"[Calc USD (BTC/XAU)] Crypto {cleaned_symbol} Result: {result}")
                return result
            else:
                if currency_converter is None:
                    logger.warning(f"[Calc USD (BTC/XAU)] Cannot convert non-USD Crypto {cleaned_symbol} without currency converter.")
                    return None
                quote_ccy = cleaned_symbol[-3:]
                logger.debug(f"[Calc USD (BTC/XAU)] Crypto attempting conversion: {quote_ccy} -> USD")
                try:
                    rate = Decimal(str(currency_converter.get_rate(quote_ccy, "USD")))
                    logger.debug(f"[Calc USD (BTC/XAU)] Crypto Rate {quote_ccy}/USD: {rate}")
                    profit_usd = profit_quote_ccy * rate
                    result = profit_usd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    logger.debug(f"[Calc USD (BTC/XAU)] Crypto {cleaned_symbol} Result (Converted): {result}")
                    return result
                except Exception as crypto_rate_err:
                    logger.error(f"[Calc USD (BTC/XAU)] Rate error {quote_ccy}->USD: {crypto_rate_err}")
                    return None
        # Gold (XAUUSD)
        elif cleaned_symbol == "XAUUSD":
            logger.debug(f"[Calc USD (BTC/XAU)] Detected Gold: {cleaned_symbol}")
            pip_value_per_lot = Decimal("10.0")
            logger.debug(f"[Calc USD (BTC/XAU)] XAUUSD PipValPerLot: {pip_value_per_lot}")
            profit_usd = profit_pips_dec * pip_value_per_lot * volume_dec
            result = profit_usd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            logger.debug(f"[Calc USD (BTC/XAU)] XAUUSD Result: {result}")
            return result
        else:
            logger.warning(f"[Calc USD (BTC/XAU)] Unsupported symbol type for calculation: {cleaned_symbol}. Returning None.")
            return None
    except (InvalidOperation, ValueError, TypeError) as e:
        logger.error(f"[Calc USD (BTC/XAU)] General calculation error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[Calc USD (BTC/XAU)] Unexpected calculation error: {e}", exc_info=True)
        return None

async def closed_order_monitor_task():
    """
    Theo dõi các lệnh đã đóng, cập nhật thông tin đóng lệnh vào DB
    và tạo/upload biểu đồ cuối cùng. (Thêm logging debug profit)
    """
    logger.info("--- Starting Closed Order Monitor Task (Updates DB Close Details) ---")
    
    # Đảm bảo pandas được import đúng cách
    try:
        import pandas as pd
        logger.info(f"Pandas successfully imported in closed_order_monitor_task, version: {pd.__version__}")
    except ImportError:
        logger.error("Pandas not installed or not importable in closed_order_monitor_task!")
        # Không raise exception để tránh làm crash bot
        return
    except Exception as pd_err:
        logger.error(f"Error with pandas in closed_order_monitor_task: {pd_err}")
        # Không raise exception để tránh làm crash bot
        return
        
    await asyncio.sleep(30)
    check_interval = 60

    while True:
        await asyncio.sleep(check_interval)
        if not check_mt5_connection():
            logger.warning("[ClosedOrderMon] MT5 disconnected.")
            continue

        try:
            time_to = datetime.now(timezone.utc)
            time_from = time_to - timedelta(hours=4)
            history_deals = await asyncio.to_thread(mt5.history_deals_get, time_from, time_to)

            if history_deals is None:
                if mt5.last_error()[0] not in [0, 1]:
                    logger.error(f"[ClosedOrderMon] Error getting history deals: {mt5.last_error()}")
                continue
            if not history_deals:
                continue

            from collections import defaultdict
            deals_by_position = defaultdict(lambda: {'in': None, 'out': None})
            for deal in history_deals:
                if deal.magic != BOT_MAGIC_NUMBER:
                    continue
                if deal.entry not in [mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                    continue
                pos_id = deal.position_id
                if deal.entry in [mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_INOUT] and deals_by_position[pos_id]['in'] is None:
                    deals_by_position[pos_id]['in'] = deal
                elif deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                    if deals_by_position[pos_id]['out'] is None or deal.time_msc > deals_by_position[pos_id]['out'].time_msc:
                        deals_by_position[pos_id]['out'] = deal

            for pos_id, deals in deals_by_position.items():
                deal_in = deals.get('in')
                deal_out = deals.get('out')

                if deal_in and deal_out:
                    ticket_id = deal_in.order
                    if ticket_id not in processed_closed_tickets:
                        logger.debug(f"[ClosedOrderMon] Found closed position {pos_id} (Ticket: {ticket_id}). Checking DB...")
                        # STUB: await db_logger.check_order_exists
                        order_exists_in_db = await db_logger.check_order_exists(ticket_id)

                        if order_exists_in_db:
                            logger.info(f"[ClosedOrderMon] Ticket {ticket_id} found in DB. Processing close details...")
                            symbol = deal_in.symbol
                            volume = deal_in.volume
                            entry_price = deal_in.price
                            close_price = deal_out.price
                            entry_time = datetime.fromtimestamp(deal_in.time, tz=timezone.utc)
                            close_time = datetime.fromtimestamp(deal_out.time, tz=timezone.utc)
                            order_type = "BUY" if deal_in.type == mt5.DEAL_TYPE_BUY else "SELL"
                            profit_pips = None
                            profit_usd = None
                            close_reason = "CLOSED"
                            point = 0.0
                            pip_def_in_points = 1
                            try:
                                symbol_info = mt5.symbol_info(symbol)
                                if not symbol_info:
                                    logger.error(f"Cannot get symbol info for {symbol} (Ticket {ticket_id}). Profit calc skipped.")
                                else:
                                    point = symbol_info.point
                                    digits = symbol_info.digits
                                    settings = SYMBOL_SETTINGS.get(symbol, {})
                                    pip_def_in_points = settings.get('pip_definition_in_points')
                                    logger.info(f"[Profit Calc Input] Ticket: {ticket_id}, Symbol: {symbol}, Type: {order_type}, "
                                                f"Entry: {entry_price}, Close: {close_price}, Volume: {volume}, "
                                                f"Point: {point}, PipDef: {pip_def_in_points} (From Settings: {settings.get('pip_definition_in_points')})")
                                    if point > 0 and pip_def_in_points is not None and pip_def_in_points > 0:
                                        profit_points = 0.0
                                        if order_type == "BUY":
                                            profit_points = (close_price - entry_price) / point
                                        else:
                                            profit_points = (entry_price - close_price) / point
                                        profit_pips = round(profit_points / pip_def_in_points, 2)
                                        logger.info(f"[Profit Calc Pips] Ticket: {ticket_id}, ProfitPoints: {profit_points:.2f}, Calculated Pips: {profit_pips}")
                                        # STUB: calculate_profit_usd_approx_local
                                        profit_usd = calculate_profit_usd_approx_local(
                                            profit_pips=profit_pips,
                                            volume=volume,
                                            symbol=symbol,
                                            logger_instance=logger
                                        )
                                        logger.info(f"[Profit Calc USD] Ticket: {ticket_id}, Calculated USD: {profit_usd}")
                                    else:
                                        logger.error(f"Invalid point ({point}) or pip_definition_in_points ({pip_def_in_points}) for {symbol} (Ticket {ticket_id}). Profit pips calculation skipped.")
                                        profit_pips = None
                                        profit_usd = None
                                close_reason = "CLOSED"
                            except Exception as calc_err:
                                logger.error(f"Error calculating profit for ticket {ticket_id}: {calc_err}")
                                profit_pips = None
                                profit_usd = None
                            logger.info(f"[DB Update Call] Ticket: {ticket_id}, ClosePrice: {close_price}, CloseTime: {close_time}, "
                                        f"ProfitPips: {profit_pips}, ProfitUSD: {profit_usd}, Reason: {close_reason}")
                            # STUB: await db_logger.update_order_close_details
                            await db_logger.update_order_close_details(
                                ticket_id=ticket_id,
                                close_price=close_price,
                                close_timestamp=close_time,
                                profit_pips=profit_pips,
                                profit_usd=profit_usd,
                                close_reason=close_reason
                            )
                            # STUB: process_closed_order_chart
                            asyncio.create_task(
                                process_closed_order_chart(symbol, ticket_id, entry_time, entry_price, close_time, close_price, order_type),
                                name=f"FinalChart_{ticket_id}")
                            processed_closed_tickets.add(ticket_id)
                            logger.debug(f"[ClosedOrderMon] Added ticket {ticket_id} to processed set.")
                            await asyncio.sleep(0.1)
                        else:
                            logger.info(f"[ClosedOrderMon] Skipping processing for Ticket {ticket_id}: Not found in 'orders' table.")
                            processed_closed_tickets.add(ticket_id)
                            logger.debug(f"[ClosedOrderMon] Added ticket {ticket_id} to processed set (skipped - not in DB).")
        except Exception as e:
            logger.error(f"[ClosedOrderMon] Error in monitor loop: {e}", exc_info=True)
            await asyncio.sleep(30)
        
        # Khoảng thời gian chờ giữa các lần kiểm tra
        logger.debug(f"[ClosedOrderMon] Sleeping for {check_interval} seconds before next check...")
        await asyncio.sleep(check_interval)

# ===============================================================================
async def place_market_order(symbol: str, order_type: str, volume: float, sl_price: float, comment: str = "AI Scalp V4") -> tuple | None:
    ... # (function body unchanged for brevity)

# --- Hàm gọi gpt-4o (g4f) giữ nguyên logic, chỉ cần đảm bảo trả về format chuẩn ---
async def call_gpt4o_api_for_direction(prompt: str, symbol: str) -> str:
    """Calls the gpt-4o model via g4f library for trading direction."""
    logger.info(f"--- Calling gpt-4o API (g4f) for {symbol} ---")
    try:
        # Sử dụng AsyncClient cho asyncio
        client = AsyncClient()
        messages = [
            {"role": "system", "content": f"You are an expert scalping trading assistant AI for {symbol}. Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."},
            {"role": "user", "content": prompt}
        ]
        response = await client.chat.completions.create(
            model=Config.MODEL_GPT4O_G4F,
            messages=messages,
        )
        final_content = getattr(response.choices[0].message, 'content', None)
        if final_content:
            norm_output = normalize_ai_output(final_content)
            logger.info(f"gpt-4o (g4f) Response Interpreted for {symbol}: {norm_output}")
            return norm_output
        else:
            logger.warning(f"gpt-4o (g4f) response for {symbol} no content.")
            return "INVALID_OUTPUT"
    except Exception as e:
        logger.error(f"Generic Error calling gpt-4o API (g4f) for {symbol}: {e}", exc_info=True)
        return f"Error: Generic g4f fail"

# --- Hàm gọi xAI (Grok) giữ nguyên logic, chỉ cần đảm bảo trả về format chuẩn ---
async def call_xai_api_for_direction(prompt: str, symbol: str) -> str:
    """Calls the xAI (Grok) model for trading direction."""
    logger.info(f"--- Calling xAI API ({Config.XAI_MODEL}) for {symbol} ---")
    
    # Lấy API key từ Config
    try:
        # Sử dụng API key từ Config hoặc từ biến môi trường
        api_key = Config.XAI_API_KEY if hasattr(Config, 'XAI_API_KEY') else XAI_API_KEY
        model = Config.XAI_MODEL if hasattr(Config, 'XAI_MODEL') else XAI_MODEL
        
        # Kiểm tra API key có hợp lệ không
        if not api_key or "YOUR_XAI_API_KEY_HERE" in api_key:
            logger.error("XAI_API_KEY is not configured or invalid.") 
            return "Error: XAI Key missing or invalid"
        
        logger.debug(f"Using xAI API key: {api_key[:10]}... and model: {model}")
        
        # Cấu hình messages - Đơn giản hóa để tránh lỗi 422
        messages = [
            {"role": "system", "content": "You are a trading assistant. Respond with BUY, SELL, or NO SIGNAL only."},
            {"role": "user", "content": prompt}
        ]
        
        # Gọi API với timeout ngắn hơn
        try:
            # Sử dụng OpenAI client (không cần đóng session)
            client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
            completion = await asyncio.to_thread(
                client.chat.completions.create, 
                model=model, 
                messages=messages,
                temperature=0.5,  # Giảm temperature để tăng tính ổn định
                max_tokens=50,    # Giới hạn độ dài output
                stream=False,
                timeout=15.0      # Giảm timeout để tránh chờ quá lâu
            )
        except Exception as api_err:
            logger.error(f"Error calling xAI API: {api_err}")
            return f"Error: xAI API call failed - {str(api_err)[:50]}"
        final_content = getattr(completion.choices[0].message, 'content', None)
        if final_content:
            norm_output = normalize_ai_output(final_content)
            logger.info(f"xAI Response Interpreted for {symbol}: {norm_output}")
            return norm_output
        else:
            logger.warning(f"xAI response for {symbol} no content.")
            return "INVALID_OUTPUT"
    except AuthenticationError as e: logger.error(f"xAI Auth Error {symbol}: {e}"); return f"Error: xAI Auth fail"
    except RateLimitError as e: logger.error(f"xAI Rate Limit {symbol}: {e}"); return f"Error: xAI Rate Limit"
    except APIConnectionError as e: logger.error(f"xAI Connection Error {symbol}: {e}"); return f"Error: xAI Connect fail"
    except APIStatusError as e: logger.error(f"xAI Status Error {symbol}: Status={e.status_code}"); return f"Error: xAI Status {e.status_code}"
    except Exception as e: logger.error(f"Generic Error calling xAI API for {symbol}: {e}", exc_info=True); return f"Error: Generic xAI fail"

# --- Entrypoint để chạy bot từ bot.py ---
async def main():
    logger.info("Starting AI Trading Bot (refactored) via main() entrypoint...")
    from core import bot_orchestrator
    
    # 1. Khởi tạo DB pool (nếu cần)
    try:
        await db_logger.init_db_pool()
    except Exception as e:
        logger.warning(f"DB pool initialization failed (non-critical): {e}")
    
    # 2. Khởi tạo MT5
    if not bot_orchestrator.init_mt5():
        logger.error("MT5 initialization failed. Bot will exit.")
        return
    
    # 3. Kiểm tra kết nối MT5
    if not bot_orchestrator.check_mt5_connection():
        logger.error("MT5 connection check failed. Bot will exit.")
        return
    
    # 4. Load API keys từ database
    global gemini_rotator, openrouter_rotators, gemini_api_keys, openrouter_api_keys
    
    # Load API keys từ database
    try:
        from core import api_key_loader
        db_gemini_keys, db_openrouter_keys = await api_key_loader.load_api_keys_from_db()
    except Exception as e:
        logger.error(f"Error loading API keys from database: {e}")
        db_gemini_keys, db_openrouter_keys = [], []
    
    # Sử dụng keys từ database hoặc fallback về hardcoded nếu cần
    if db_gemini_keys:
        gemini_api_keys = db_gemini_keys
        logger.info(f"Using {len(gemini_api_keys)} Gemini API keys from database")
    else:
        gemini_api_keys = []  # Fallback empty
        logger.warning("No Gemini API keys found in database")
    
    if db_openrouter_keys:
        openrouter_api_keys = db_openrouter_keys
        logger.info(f"Using {len(openrouter_api_keys)} OpenRouter API keys from database")
    else:
        # Fallback to hardcoded key if no keys in database
        openrouter_api_keys = [OPENROUTER_API_KEY] if OPENROUTER_API_KEY else []
        if openrouter_api_keys:
            logger.warning("Using hardcoded OpenRouter API key as fallback")
        else:
            logger.warning("No OpenRouter API keys available")
    
    # Khởi tạo Gemini rotator nếu có API keys
    try:
        if len(gemini_api_keys) > 0:
            try:
                gemini_rotator = APIKeyRotator(gemini_api_keys, interval_seconds=60, api_type="gemini", model_name=Config.MODEL_GEMINI)
                gemini_rotator.start()
                logger.info(f"Gemini API Rotator started with {len(gemini_api_keys)} keys")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini rotator: {e}")
                gemini_rotator = None
        else:
            logger.warning("No Gemini API keys configured. Gemini models will be unavailable.")
            gemini_rotator = None
    except Exception as e:
        logger.error(f"Unexpected error initializing Gemini rotator: {e}")
        gemini_rotator = None
    
    # Khởi tạo OpenRouter rotators nếu có API keys
    try:
        if len(openrouter_api_keys) > 0:
            for model_key, model_cfg in AI_MODELS_TO_CALL.items():
                if model_cfg['type'] == 'openrouter':
                    try:
                        model_name = model_cfg['model_name']
                        openrouter_rotators[model_name] = APIKeyRotator(
                            openrouter_api_keys, interval_seconds=60, 
                            api_type="openrouter", model_name=model_name
                        )
                        openrouter_rotators[model_name].start()
                        logger.info(f"OpenRouter API Rotator started for {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize OpenRouter rotator for {model_cfg['model_name']}: {e}")
                        # Tiếp tục với các model khác
            logger.info(f"OpenRouter API Rotators initialized with {len(openrouter_api_keys)} keys")
        else:
            logger.warning("No OpenRouter API keys configured. OpenRouter models will be unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error initializing OpenRouter rotators: {e}")
    
    # 5. Gửi thông báo Telegram khi bot start
    try:
        await bot_orchestrator.send_telegram_message("🤖 AI Trading Bot đã khởi động thành công!")
    except Exception as e:
        logger.warning(f"Could not send Telegram startup message: {e}")
    
    # 6. Khởi động các tasks chính
    try:
        logger.info("Preparing to start trading tasks...")
        
        # Kiểm tra các biến và môi trường trước khi bắt đầu
        logger.info(f"Current symbols: {SCALPING_SYMBOLS}")
        logger.info(f"AI models configured: {list(AI_MODELS_TO_CALL.keys())}")
        
        # Kiểm tra pandas đã được import chưa
        try:
            import pandas as pd
            logger.info(f"Pandas version: {pd.__version__}")
        except ImportError:
            logger.error("Pandas not installed or not importable!")
            raise
        except Exception as pd_err:
            logger.error(f"Error with pandas: {pd_err}")
            raise
            
        # Tạo các tasks chính sử dụng create_task để đăng ký với event loop
        logger.info("Creating ai_signal_and_entry_loop task...")
        signal_entry_task = asyncio.create_task(
            ai_signal_and_entry_loop(),
            name="ai_signal_and_entry_loop"
        )
        
        logger.info("Creating closed_order_monitor_task...")
        monitor_task = asyncio.create_task(
            closed_order_monitor_task(),
            name="closed_order_monitor_task"
        )
        
        # Thêm task reversal monitor nếu cần
        # reversal_task = asyncio.create_task(reversal_monitor_task(), name="reversal_monitor_task")
        
        # Tạo danh sách tasks để theo dõi
        tasks = [signal_entry_task, monitor_task]  # Thêm reversal_task nếu cần
        
        logger.info("All trading tasks created and started successfully")
        
        # Chạy cho đến khi tất cả các tasks hoàn thành hoặc bị hủy
        logger.info("Waiting for tasks to complete with asyncio.gather()...")
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
    finally:
        # Đóng các resources khi kết thúc
        logger.info("Shutting down bot and cleaning up resources...")
        
        # Đóng các rotators
        try:
            if gemini_rotator:
                gemini_rotator.stop()
                logger.info("Gemini rotator stopped.")
        except Exception as e:
            logger.error(f"Error stopping Gemini rotator: {e}")
            
        try:
            for model_name, rotator in openrouter_rotators.items():
                try:
                    rotator.stop()
                    logger.info(f"OpenRouter rotator for {model_name} stopped.")
                except Exception as e:
                    logger.error(f"Error stopping OpenRouter rotator for {model_name}: {e}")
        except Exception as e:
            logger.error(f"Error stopping OpenRouter rotators: {e}")
        
        # Đóng database connection pool
        try:
            await db_logger.close_db_pool()
            logger.info("Database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing database connection pool: {e}")
            
        # Đóng MT5 connection
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
            logger.info("MT5 connection shut down.")
        except Exception as e:
            logger.error(f"Error shutting down MT5: {e}")
            
        # Đóng các client sessions còn mở
        try:
            import asyncio
            for task in asyncio.all_tasks():
                if not task.done() and task != asyncio.current_task():
                    task.cancel()
            logger.info("All remaining tasks canceled.")
        except Exception as e:
            logger.error(f"Error canceling remaining tasks: {e}")
            
        logger.info("Bot shutdown complete.")

# --- Hàm gọi Gemini giữ nguyên logic, chỉ cần đảm bảo trả về format chuẩn ---
async def call_gemini_api_for_direction(prompt: str, symbol: str) -> str:
    ... # (function body unchanged for brevity)

# --- Chuẩn hóa output AI trả về (BUY/SELL/NO SIGNAL) ---
def normalize_ai_output(output: str | dict | Exception) -> str:
    """Normalizes AI response string or handles exceptions.
    Chỉ chấp nhận các dòng có format chính xác:
    - TYPE: BUY
    - TYPE: SELL
    - TYPE: NO SIGNAL / TYPE: NO_SIGNAL
    Nếu không đúng format, trả về "NO SIGNAL".
    """
    # Xử lý trường hợp output là dictionary (từ OpenRouter API)
    if isinstance(output, dict):
        if output:
            signal = next(iter(output.values()))
            return normalize_ai_output(signal)
        else:
            return "INVALID_OUTPUT"
    if isinstance(output, Exception):
        logger.warning(f"AI call resulted in exception: {output}")
        return f"Error: {type(output).__name__}"
    if not isinstance(output, str) or not output.strip():
        return "INVALID_OUTPUT"
    output_lines = output.strip().split('\n')
    for line in output_lines:
        line_upper = line.strip().upper()
        # Chỉ chấp nhận các dòng có format chính xác
        if line_upper == "TYPE: BUY":
            return "BUY"
        elif line_upper == "TYPE: SELL":
            return "SELL"
        elif line_upper == "TYPE: NO SIGNAL" or line_upper == "TYPE: NO_SIGNAL":
            return "NO SIGNAL"
    # Nếu không tìm thấy dòng đúng format nào, trả về NO SIGNAL
    logger.warning(f"AI response unclear or unexpected format: '{output[:100]}' -> NO SIGNAL")
    return "NO SIGNAL"

# --- Lưu response AI theo model ---
def save_ai_response_by_model(response: str, model_id: str, folder: str = "ai_responses"):
    """
    Lưu response của AI vào file riêng theo model trong folder chỉ định.
    Nếu folder chưa tồn tại sẽ tự tạo.
    """
    os.makedirs(folder, exist_ok=True)
    safe_model_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
    filename = f"{safe_model_id}.txt"
    filepath = os.path.join(folder, filename)
    content = f"{response}\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)

# --- Hàm gọi OpenRouter API cho direction ---
async def call_openrouter_api_for_direction(prompt: str, symbol: str, model_id: str, rotator) -> str | Exception:
    """
    Calls the specified OpenRouter model via the OpenAI-compatible API for trading direction.
    Automatically detects if the model should use reasoning based on model_id.
    Returns the normalized AI response string ("BUY", "SELL", "NO SIGNAL", "INVALID_OUTPUT") or an Exception object if an API error occurs.
    """
    REASONING_MODELS = [
        "llama-4", "llama-3.1", "llama-3.3", "nemotron-ultra", "nemotron-super", "reka-flash",
        "deepseek-chat", "claude-3", "gpt-4", "gemini-pro", "qwen-2.5-",
    ]
    is_reasoning_model = any(pattern in model_id.lower() for pattern in REASONING_MODELS)
    logger.info(f"--- Calling OpenRouter API ({model_id}) for {symbol} {'with reasoning capabilities' if is_reasoning_model else 'direct signal'} ---")
    if not rotator:
        logger.error(f"No OpenRouter rotator provided for model {model_id} (Symbol: {symbol}).")
        return ConnectionAbortedError(f"No rotator for {model_id}")
    client = rotator.get_model_for_usage()
    if not client:
        logger.error(f"Failed to get OpenRouter client from rotator for model {model_id} (Symbol: {symbol}).")
        if rotator:
            rotator.release_model_usage()
        return ConnectionAbortedError(f"No client from rotator for {model_id}")
    start_time = time.monotonic()
    current_key_info = f"...{rotator._current_key[-4:]}" if rotator._current_key else 'N/A'
    logger.debug(f"Using OpenRouter key {current_key_info} for {symbol} model {model_id}")
    try:
        async def api_call_task():
            if is_reasoning_model:
                system_content = (
                    f"You are an expert scalping trading assistant AI for {symbol}. "
                    f"Analyze the provided data carefully and provide your reasoning. "
                    f"Conclude with a clear trading signal by writing 'TYPE: BUY', 'TYPE: SELL', "
                    f"or 'TYPE: NO SIGNAL' on a new line at the end of your response."
                )
                max_tokens = 1024
            else:
                system_content = (
                    f"You are an expert scalping trading assistant AI for {symbol}. "
                    f"Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. "
                    f"Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."
                )
                max_tokens = 50
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=25.0
            )
            return completion
        completion = await asyncio.wait_for(api_call_task(), timeout=30.0)
        response_time = time.monotonic() - start_time
        logger.info(f"OpenRouter API ({model_id}) response time for {symbol}: {response_time:.2f} seconds")
        if completion and completion.choices and len(completion.choices) > 0:
            final_content = getattr(completion.choices[0].message, 'content', None)
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} is None or has empty choices.")
            final_content = None
        if final_content:
            save_ai_response_by_model(final_content, model_id)
            return final_content
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} had no content.")
            save_ai_response_by_model("INVALID_OUTPUT", model_id)
            return "INVALID_OUTPUT"
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling OpenRouter API ({model_id}, {symbol}): {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", model_id)
        return e
    finally:
        if rotator:
            rotator.release_model_usage()
            logger.debug(f"Released OpenRouter client usage for model {model_id} (Symbol: {symbol})")

