"""
Reversal Monitor Task (migrated): Monitors open positions for reversal signals and closes positions based on code/AI consensus.
"""
import asyncio
import logging
import time
import traceback
import datetime
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Callable, Any
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

REVERSAL_MONITOR_INTERVAL = 10
REVERSAL_TIMEFRAMES = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15]
REVERSAL_BAR_COUNT = 50
REVERSAL_CODE_SIGNAL_THRESHOLD = 3
REVERSAL_AI_CONFIRMATION_ENABLED = True
REVERSAL_AI_CONSENSUS_THRESHOLD = 3
REVERSAL_AI_CALL_COOLDOWN_MINUTES = 15
REVERSAL_POSITION_CLOSE_COOLDOWN_MINUTES = 5

TIMEFRAME_TO_STRING_MAP = {
    mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
    mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1",
}
def get_timeframe_string(tf_code):
    return TIMEFRAME_TO_STRING_MAP.get(tf_code, str(tf_code))

def check_pin_bar_rm(candle: pd.Series) -> tuple[bool, Optional[str]]:
    logger = logging.getLogger("ReversalMonitor")
    logger.debug(f"Checking pin bar: {candle}")
    if candle is None or not isinstance(candle, pd.Series):
        logger.debug("Invalid candle input for pin bar check.")
        return False, None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in candle.index for col in required_cols) or candle[required_cols].isnull().any():
        logger.debug("Candle missing required columns or contains NaN.")
        return False, None
    open_price, high_price, low_price, close_price = candle['Open'], candle['High'], candle['Low'], candle['Close']
    body = abs(close_price - open_price)
    range_total = high_price - low_price
    if range_total == 0:
        logger.debug("Candle range is zero.")
        return False, None
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    is_bullish_pin, is_bearish_pin = False, False
    min_wick_ratio = 2.0
    max_other_wick_ratio = 0.5
    max_body_ratio = 0.33
    if body > 0:
        if lower_wick >= body * min_wick_ratio and upper_wick <= body * max_other_wick_ratio and body / range_total <= max_body_ratio:
            is_bullish_pin = True
        if upper_wick >= body * min_wick_ratio and lower_wick <= body * max_other_wick_ratio and body / range_total <= max_body_ratio:
            is_bearish_pin = True
    if is_bullish_pin:
        logger.info("Bullish pin bar detected.")
        return True, 'bullish'
    if is_bearish_pin:
        logger.info("Bearish pin bar detected.")
        return True, 'bearish'
    logger.debug("No pin bar detected.")
    return False, None

def calculate_bbands_rm(df: pd.DataFrame, period: int = 14, std_dev: float = 2.0, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger("ReversalMonitor")
    logger.debug(f"Calculating BBands | period: {period}, std_dev: {std_dev}, df_len: {len(df) if df is not None else 'None'}")
    if df is None or len(df) < period:
        logger.warning("Insufficient data for BBands calculation.")
        return df
    try:
        import pandas_ta as ta
        bbands_df = df.ta.bbands(length=period, std=std_dev, append=True)
        df.rename(columns={
            f'BBL_{period}_{std_dev}': 'BB_Lower',
            f'BBM_{period}_{std_dev}': 'BB_Middle',
            f'BBU_{period}_{std_dev}': 'BB_Upper',
        }, inplace=True, errors='ignore')
        logger.debug("BBands calculated using pandas_ta.")
    except ImportError:
        logger.debug("[RM BBands] pandas_ta not found. Calculating manually.")
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (std_dev * df['BB_StdDev'])
        df['BB_Lower'] = df['BB_Middle'] - (std_dev * df['BB_StdDev'])
        logger.debug("BBands calculated manually.")
    except Exception as e:
        logger.error(f"[RM BBands] Error calculating BBands: {e}")
        df['BB_Upper'] = np.nan
        df['BB_Middle'] = np.nan
        df['BB_Lower'] = np.nan
    return df

async def get_rates_rm(symbol: str, timeframe: int, count: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    logger.debug(f"Fetching rates | symbol: {symbol}, timeframe: {timeframe}, count: {count}")
    try:
        rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates data returned for {symbol} | timeframe: {timeframe}")
            return None
        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        rates_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True, errors='ignore')
        logger.debug(f"Fetched {len(rates_df)} rates for {symbol} | timeframe: {timeframe}")
        return rates_df
    except Exception as e:
        logger.error(f"[RM] Error fetching rates for {symbol} ({timeframe}): {e}", exc_info=True)
        return None

def check_reversal_signals_code_only_rm(position, all_tf_data: Dict[int, pd.DataFrame], logger: logging.Logger) -> tuple[int, List[str]]:
    logger.debug(f"Checking code-based reversal signals | position: {getattr(position, 'ticket', None)}, symbol: {getattr(position, 'symbol', None)}")
    score = 0
    code_signals = []
    tf_keys = list(all_tf_data.keys())
    for tf_code in tf_keys:
        df = all_tf_data[tf_code]
        tf_str = get_timeframe_string(tf_code)
        try:
            if df is None or len(df) < 20:
                logger.debug(f"[{getattr(position, 'ticket', None)} RM] Not enough data for {tf_str}")
                continue
            df = calculate_bbands_rm(df, logger=logger)
            last_candle = df.iloc[-1]
            mid_bb = last_candle.get('BB_Middle', None)
            close = last_candle['Close']
            if mid_bb is None or pd.isna(mid_bb):
                logger.debug(f"[{getattr(position, 'ticket', None)} RM] No BB_Middle for {tf_str}")
                continue
            is_pin, pin_type = check_pin_bar_rm(last_candle)
            if position.type == 0: # BUY
                if is_pin and pin_type == 'bearish':
                    code_signals.append(f"{tf_str}:BearPin")
                    score += 2 if tf_code >= mt5.TIMEFRAME_M5 else 1
                if close < mid_bb:
                    code_signals.append(f"{tf_str}:Close<MidBB")
                    score += 2 if tf_code >= mt5.TIMEFRAME_M5 else 1
            elif position.type == 1: # SELL
                if is_pin and pin_type == 'bullish':
                    code_signals.append(f"{tf_str}:BullPin")
                    score += 2 if tf_code >= mt5.TIMEFRAME_M5 else 1
                if close > mid_bb:
                    code_signals.append(f"{tf_str}:Close>MidBB")
                    score += 2 if tf_code >= mt5.TIMEFRAME_M5 else 1
        except Exception as e:
            logger.error(f"[{getattr(position, 'ticket', None)} RM] Error checking signals for {tf_str}: {e}", exc_info=True)
    unique_signals = sorted(list(set(code_signals)))
    logger.info(f"[{getattr(position, 'ticket', None)} {getattr(position, 'symbol', None)} RM] Code Reversal Score: {score}. Signals: {', '.join(unique_signals) if unique_signals else 'None'}")
    return score, unique_signals

def normalize_reversal_ai_output_rm(output: Any, logger: logging.Logger) -> str:
    logger.debug(f"Normalizing AI output | output: {output}")
    if isinstance(output, Exception):
        logger.warning(f"[RM Norm] AI call resulted in exception: {type(output).__name__}")
        return f"Error: {type(output).__name__}"
    if not output or not isinstance(output, str):
        logger.warning(f"[RM Norm] AI response is empty or not a string: {type(output)}")
        return "INVALID_OUTPUT"
    output_upper = output.strip().upper()
    if "TYPE: CLOSE" in output_upper: 
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as CLOSE")
        return "CLOSE"
    if "TYPE: HOLD" in output_upper: 
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as HOLD")
        return "HOLD"
    if "CLOSE" in output_upper and "HOLD" not in output_upper:
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as CLOSE")
        return "CLOSE"
    if "HOLD" in output_upper and "CLOSE" not in output_upper:
        logger.debug(f"[RM Norm] Interpreted '{output[:50]}...' as HOLD")
        return "HOLD"
    logger.warning(f"[RM Norm] AI response unclear: '{output[:100]}...' -> HOLD (default)")
    return "HOLD"

async def close_position_by_ticket_rm(ticket: int, magic_number: int, logger_instance: logging.Logger) -> bool:
    logger_instance.info(f"[RM] Attempting to close position | ticket: {ticket}, magic: {magic_number}")
    try:
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger_instance.info(f"[{ticket} RM Close] Position already closed or not found.")
            return True
        position_to_close = positions[0]
        symbol = position_to_close.symbol
        volume = position_to_close.volume
        order_type = position_to_close.type
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.POSITION_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": magic_number,
            "comment": "AI Reversal Close",
        }
        result = await asyncio.to_thread(mt5.order_send, request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger_instance.info(f"[{ticket} RM Close] Close order successful.")
            return True
        else:
            logger_instance.error(f"[{ticket} RM Close] Close order failed: {result.comment if result else 'No result'}, Code: {result.retcode if result else 'N/A'}")
            return False
    except Exception as e:
        logger_instance.error(f"[{ticket} RM Close] Exception during close attempt: {e}", exc_info=True)
        return False

async def get_ai_consensus_for_reversal_rm(
    position: Any,
    code_reversal_score: int,
    code_signal_desc: List[str],
    all_tf_data: Dict[int, pd.DataFrame],
    ai_models_config: Dict[str, Dict[str, Any]],
    ai_rotators: Dict[str, Any],
    logger: logging.Logger,
    reversal_log_id: Optional[int]
) -> bool:
    """
    Calls configured AI models for reversal consensus confirmation.
    Logs opinions to DB.
    """
    logger.info(f"[RM] Starting AI consensus for reversal | position: {getattr(position, 'ticket', None)}, code_score: {code_reversal_score}, signals: {code_signal_desc}")
    ai_votes = {}
    consensus_threshold = REVERSAL_AI_CONSENSUS_THRESHOLD
    for model_key, model_cfg in ai_models_config.items():
        if model_key not in ai_rotators:
            logger.warning(f"[RM] No rotator for model {model_key}")
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

    # --- Prompt Template for Reversal AI ---
    REVERSAL_AI_PROMPT_TEMPLATE = """
You are an expert trading assistant AI focused on risk management for {symbol}.
A {position_type} position (Ticket: {ticket}) opened at {open_price} is currently open.
Internal code analysis detected potential reversal signals against this position with a score of {reversal_score}.
Key signals observed: {signal_description}.

**Current Market Data Summary:**
{market_summary}

**Instruction:** Based ONLY on the provided market data and reversal signals, assess the immediate risk of the market reversing significantly against the open position.
Should this {position_type} position be closed NOW to protect profit or minimize loss due to a high probability of imminent reversal?

**Required Output Format:** ONLY one line: TYPE: CLOSE / TYPE: HOLD
"""
    prompt_for_reversal_ai = REVERSAL_AI_PROMPT_TEMPLATE.format(
        symbol=symbol, position_type=pos_type_str, ticket=ticket,
        open_price=position.price_open, reversal_score=code_reversal_score,
        signal_description=', '.join(code_signal_desc) if code_signal_desc else "None",
        market_summary=market_summary.strip()
    )
    logger.debug(f"[{ticket} RM] Reversal Prompt Prepared: {prompt_for_reversal_ai[:400]}...")

    # --- Local Wrapper for API Calls & DB Logging ---
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
                import db_logger
                await db_logger.log_reversal_ai_opinion(reversal_log_id, model_key, normalized_decision)
            except Exception as db_log_err:
                logger.error(f"[RM] DB Log AI Opinion Error for {model_key} (Ticket {ticket}): {db_log_err}")
        else:
            logger.warning(f"[RM] reversal_log_id is None, cannot log AI opinion for {model_key} (Ticket {ticket})")
        return (model_key, normalized_decision)

    # --- Prepare AI Call Tasks ---
    ai_tasks = []
    ai_names_in_call = []
    gemini_rotator = ai_rotators.get("gemini")
    openrouter_rotators_dict = ai_rotators.get("openrouter", {})
    for ai_name_key, config in ai_models_config.items():
        api_call_func, args = None, ()
        model_type = config.get("type")
        model_name_id = config.get("model_name")
        rotator_for_call = None
        if model_type == "gemini":
            api_call_func = config.get("api_call_func")
            rotator_for_call = gemini_rotator
            args = (prompt_for_reversal_ai, symbol, rotator_for_call, logger)
        elif model_type == "openrouter":
            api_call_func = config.get("api_call_func")
            rotator_for_call = openrouter_rotators_dict.get(model_name_id)
            args = (prompt_for_reversal_ai, symbol, model_name_id, rotator_for_call, logger)
        elif model_type == "gpt4o":
            api_call_func = config.get("api_call_func")
            args = (prompt_for_reversal_ai, symbol, logger)
        elif model_type == "xai":
            api_call_func = config.get("api_call_func")
            args = (prompt_for_reversal_ai, symbol, logger)
        else:
            logger.warning(f"[RM] Unknown model type {model_type} for {ai_name_key}, skipping.")
            continue
        if api_call_func:
            ai_tasks.append(timed_ai_call_wrapper_reversal(ai_name_key, api_call_func, *args))
            ai_names_in_call.append(ai_name_key)
    results = await asyncio.gather(*ai_tasks, return_exceptions=True)
    consensus_votes = {k: v for k, v in results if isinstance(k, str)}
    close_votes = sum(1 for v in consensus_votes.values() if v == "CLOSE")
    logger.info(f"[RM] AI Consensus votes: {consensus_votes}. CLOSE votes: {close_votes}/{len(consensus_votes)}")
    return close_votes >= REVERSAL_AI_CONSENSUS_THRESHOLD


async def reversal_monitor_task(
    magic_number: int,
    ai_models_config: Dict[str, Dict[str, Any]],
    ai_rotators: Dict[str, Any],
    ai_cooldown_tracker: Dict[str, datetime],
    close_cooldown_tracker: Dict[int, datetime],
    check_mt5_connection_func: Callable[[], bool],
    logger_instance: logging.Logger
):
    logger = logger_instance
    logger.info("--- Starting Reversal Monitor Task ---")
    await asyncio.sleep(10)
    while True:
        await asyncio.sleep(REVERSAL_MONITOR_INTERVAL)
        if not check_mt5_connection_func():
            logger.warning("[RM] MT5 disconnected.")
            continue
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                logger.info("[RM] No open positions to monitor.")
                continue
            for pos in positions:
                if pos.magic != magic_number:
                    continue
                symbol = pos.symbol
                ticket = pos.ticket
                now_utc = datetime.now(timezone.utc)
                ai_cooldown_minutes = REVERSAL_AI_CALL_COOLDOWN_MINUTES
                close_cooldown_minutes = REVERSAL_POSITION_CLOSE_COOLDOWN_MINUTES
                if symbol in ai_cooldown_tracker and ai_cooldown_tracker[symbol] > now_utc:
                    logger.info(f"[{ticket} RM] AI cooldown active for {symbol}. Skipping.")
                    continue
                if ticket in close_cooldown_tracker and close_cooldown_tracker[ticket] > now_utc:
                    logger.info(f"[{ticket} RM] Close cooldown active. Skipping.")
                    continue
                cached_tf_data = {}
                for tf in REVERSAL_TIMEFRAMES:
                    df = await get_rates_rm(symbol, tf, REVERSAL_BAR_COUNT, logger)
                    if df is not None:
                        cached_tf_data[tf] = df
                if not cached_tf_data:
                    logger.warning(f"[{ticket} RM] No data for symbol {symbol}.")
                    continue
                code_score, code_signals = check_reversal_signals_code_only_rm(pos, cached_tf_data, logger)
                decision_to_close = False
                final_decision_str = "HOLD"
                if code_score >= REVERSAL_CODE_SIGNAL_THRESHOLD:
                    logger.info(f"[{ticket} RM] Code reversal score {code_score} >= threshold. AI confirmation enabled: {REVERSAL_AI_CONFIRMATION_ENABLED}")
                    if REVERSAL_AI_CONFIRMATION_ENABLED:
                        ai_agrees_to_close = False  # --- AI Consensus Logic ---
                        ai_agrees_to_close = await get_ai_consensus_for_reversal_rm(
                            position=pos,
                            code_reversal_score=code_score,
                            code_signal_desc=code_signals,
                            all_tf_data=cached_tf_data,
                            ai_models_config=ai_models_config,
                            ai_rotators=ai_rotators,
                            logger=logger,
                            reversal_log_id=None  # TODO: integrate DB log ID if available
                        )
                        if ai_agrees_to_close:
                            logger.warning(f"[{ticket} RM] AI CONSENSUS CONFIRMS closing position {symbol}.")
                            decision_to_close = True
                            final_decision_str = "CLOSE_BY_AI_CONSENSUS"
                        else:
                            logger.info(f"[{ticket} RM] AI consensus does NOT support closing position {symbol}.")
                            final_decision_str = "HOLD_BY_AI_CONSENSUS"
                    else:
                        decision_to_close = True
                        final_decision_str = "CLOSE_BY_CODE"
                action_taken_str = "HOLD"
                if decision_to_close:
                    logger.warning(f"[{ticket} RM] FINAL DECISION: Attempting to close position {symbol} ({final_decision_str}).")
                    action_taken_str = "CLOSE_ATTEMPTED"
                    close_success = await close_position_by_ticket_rm(ticket, magic_number, logger)
                    if close_success:
                        action_taken_str = "CLOSE_SUCCESS"
                        close_cooldown_tracker[ticket] = datetime.now(timezone.utc) + timedelta(minutes=close_cooldown_minutes)
                        logger.info(f"[{ticket} RM] Close successful. Cooldown set.")
                    else:
                        action_taken_str = "CLOSE_FAILED"
                        logger.error(f"[{ticket} RM] Failed to close position {symbol} after reversal signal.")
                # TODO: Log to DB if needed (e.g., update_reversal_event_end)
                await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"[RM] Unexpected error in main monitor loop: {e}", exc_info=True)
            logger.error(traceback.format_exc())
            await asyncio.sleep(30)
