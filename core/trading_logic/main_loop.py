"""
Main trading loop for the AI trading bot.
Contains the core trading logic and orchestration of the bot.
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import MetaTrader5 as mt5

from core.utils.logging_setup import logger
from core.utils.constants import (
    SCALPING_SYMBOLS, SYMBOL_SETTINGS, SCALPING_ENTRY_INTERVAL, 
    MANAGE_POSITIONS_INTERVAL, AI_MODELS_TO_CALL
)
from core.trading_logic.market_data import (
    check_mt5_connection, fetch_multi_timeframe_data, 
    calculate_daily_pivots, prepare_ai_prompt
)
from core.trading_logic.signal_processor import (
    get_ai_signals, get_consensus_signal, log_ai_results
)
from core.trading_logic.order_manager import (
    place_market_order, manage_positions
)
from core.db_logger import log_order_placement

async def ai_signal_and_entry_loop(
    gemini_rotator=None, 
    openrouter_rotators=None
):
    """
    Main loop for AI signal generation and trade entry.
    
    Args:
        gemini_rotator: Rotator for Gemini API keys
        openrouter_rotators: Dictionary of rotators for OpenRouter API keys
    """
    logger.info("===== STARTING AI SIGNAL AND ENTRY LOOP =====")
    logger.info(f"Gemini rotator available: {'Yes' if gemini_rotator else 'No'}")
    logger.info(f"OpenRouter rotators available: {len(openrouter_rotators) if openrouter_rotators else 0} models")
    
    # Initialize variables for market status tracking
    xauusd_market_closed_logged = False
    
    # Ensure pandas is imported correctly
    logger.info("Importing pandas library...")
    try:
        import pandas as pd
        logger.info("Pandas imported successfully in ai_signal_and_entry_loop")
    except ImportError as pd_err:
        logger.error(f"Failed to import pandas in ai_signal_and_entry_loop: {pd_err}")
        logger.critical("Cannot continue without pandas library")
        raise
    except Exception as pd_err:
        logger.error(f"Unexpected error with pandas in ai_signal_and_entry_loop: {pd_err}")
        logger.critical("Cannot continue due to pandas error")
        raise
    
    logger.info("Waiting 5 seconds before starting main loop...")
    await asyncio.sleep(5)
    logger.info("Starting main trading loop")
    
    # Main trading loop
    cycle_count = 0
    while True:
        try:
            cycle_count += 1
            start_time_cycle = time.monotonic()
            current_event_timestamp = datetime.now(timezone.utc)
            logger.info(f"===== STARTING SIGNAL CHECK CYCLE #{cycle_count} at {current_event_timestamp} =====")

            # Check MT5 connection
            logger.info("Checking MT5 connection...")
            if not check_mt5_connection():
                logger.warning("MT5 disconnect detected at start of cycle. Waiting 30 seconds before retry...")
                await asyncio.sleep(30)
                continue
            logger.info("MT5 connection verified successfully")

            # Process each symbol
            logger.info(f"Processing {len(SCALPING_SYMBOLS)} trading symbols: {', '.join(SCALPING_SYMBOLS)}")
            for symbol in SCALPING_SYMBOLS:
                logger.info(f"\n----- Processing symbol: {symbol} -----")
                settings = SYMBOL_SETTINGS.get(symbol)
                if not settings:
                    logger.error(f"Settings missing for {symbol}. Skipping.")
                    continue
                logger.info(f"Symbol settings loaded: volume={settings.get('volume')}, sl_pips={settings.get('sl_pips')}")

                start_time_symbol = time.monotonic()
                logger.info(f"Initializing data containers for {symbol}")
                dataframes = {}
                daily_pivots = None
                prompt_for_ai = None
                all_data_ok = True

                try:
                    # Get symbol information
                    logger.info(f"Retrieving MT5 symbol info for {symbol}...")
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info:
                        logger.error(f"Could not retrieve MT5 info for {symbol}. Skipping.")
                        await asyncio.sleep(1)
                        continue
                    logger.info(f"MT5 symbol info retrieved successfully for {symbol}")

                    logger.info(f"Extracting symbol parameters for {symbol}...")
                    point = symbol_info.point
                    digits = symbol_info.digits
                    min_stop_points = symbol_info.trade_stops_level
                    pip_def_in_points = settings.get('pip_definition_in_points', 1)
                    logger.info(f"Symbol parameters: point={point}, digits={digits}, min_stop_points={min_stop_points}, pip_def={pip_def_in_points}")

                    if point <= 0 or pip_def_in_points <= 0:
                        logger.error(f"Invalid point ({point}) or pip definition ({pip_def_in_points}) for {symbol}. Skipping.")
                        continue

                    # Check if market is closed
                    logger.info(f"Checking market status for {symbol}...")
                    log_key_sym_closed = f"_logged_sym_closed_{symbol}"
                    if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                        logger.warning(f"Market {symbol} not fully open (Mode: {symbol_info.trade_mode})")
                        if not getattr(ai_signal_and_entry_loop, log_key_sym_closed, False):
                            logger.info(f"Market {symbol} not fully open (Mode: {symbol_info.trade_mode}). Skipping check.")
                            setattr(ai_signal_and_entry_loop, log_key_sym_closed, True)
                        continue
                    elif hasattr(ai_signal_and_entry_loop, log_key_sym_closed):
                        logger.info(f"Market {symbol} is now open. Resuming checks.")
                        delattr(ai_signal_and_entry_loop, log_key_sym_closed)
                    logger.info(f"Market {symbol} is open and available for trading")

                    # Special check for XAUUSD
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

                    # Calculate daily pivot points
                    try:
                        daily_pivots = calculate_daily_pivots(symbol, digits)
                        if daily_pivots: 
                            logger.debug(f"[{symbol}] Calculated Pivots: {daily_pivots}")
                        else: 
                            logger.warning(f"Could not calculate daily pivots for {symbol}.")
                    except Exception as pivot_err:
                        logger.error(f"Error calculating pivots for {symbol}: {pivot_err}")
                        daily_pivots = None

                    # Fetch market data for all timeframes
                    logger.info(f"Fetching multi-timeframe data for {symbol}...")
                    dataframes = fetch_multi_timeframe_data(symbol, digits)
                    if not dataframes:
                        logger.error(f"Failed to fetch data for {symbol}. Skipping.")
                        continue
                    logger.info(f"Successfully fetched data for {symbol} across {len(dataframes)} timeframes")

                    # Prepare AI prompt with market data
                    logger.info(f"Preparing AI prompt for {symbol} with market data and indicators...")
                    prompt_for_ai = prepare_ai_prompt(symbol, dataframes, daily_pivots)
                    if not prompt_for_ai or "Error preparing market data" in prompt_for_ai:
                        logger.error(f"Failed to prepare AI prompt for {symbol}. Skipping.")
                        continue
                    logger.info(f"Successfully prepared AI prompt for {symbol} ({len(prompt_for_ai)} characters)")

                    # Get signals from AI models
                    logger.info(f"Requesting trading signals from AI models for {symbol}...")
                    ai_results = await get_ai_signals(
                        prompt_for_ai, 
                        symbol, 
                        gemini_rotator, 
                        openrouter_rotators
                    )
                    logger.info(f"Received responses from {len(ai_results)} AI models for {symbol}")

                    # Get consensus signal
                    logger.info(f"Calculating consensus signal for {symbol} (required consensus: 2)...")
                    consensus_signal = get_consensus_signal(ai_results, required_consensus=2)
                    
                    # Log results
                    log_ai_results(symbol, ai_results, consensus_signal)
                    
                    # Skip if no consensus
                    if not consensus_signal:
                        logger.info(f"No consensus signal reached for {symbol}. Skipping trade entry.")
                        continue
                    logger.info(f"Consensus signal for {symbol}: {consensus_signal}")

                    # Place market order if consensus
                    try:
                        logger.info(f"Preparing to place {consensus_signal} order for {symbol}...")
                        
                        # Get current price
                        logger.info(f"Getting current price for {symbol}...")
                        tick = mt5.symbol_info_tick(symbol)
                        if not tick:
                            logger.error(f"Cannot get tick for {symbol}. Skipping.")
                            continue
                        current_price = tick.ask if consensus_signal == "BUY" else tick.bid
                        logger.info(f"Current price for {symbol}: {current_price}")
                            
                        # Get settings for this symbol
                        volume = settings.get('volume', 0.01)
                        sl_pips = settings.get('sl_pips', 10)
                        tp_pips = settings.get('tp_pips', 20)
                        logger.info(f"Order parameters for {symbol}: volume={volume}, SL={sl_pips} pips, TP={tp_pips} pips")
                        
                        # Calculate SL price based on order direction
                        point = symbol_info.point
                        sl_points = sl_pips * pip_def_in_points
                        if consensus_signal == "BUY":
                            entry_price = tick.ask
                            sl_price = entry_price - (sl_points * point)
                        else:  # SELL
                            entry_price = tick.bid
                            sl_price = entry_price + (sl_points * point)
                        sl_price = round(sl_price, digits)
                        
                        # Place order
                        logger.info(f"Placing {consensus_signal} market order for {symbol}...")
                        order_result = await place_market_order(symbol, consensus_signal, volume, sl_price)
                        
                        if order_result:
                            logger.info(f"Order placed successfully for {symbol}: {order_result}")
                            
                            # Send notification
                            logger.info(f"Sending Telegram notification for new {consensus_signal} order on {symbol}...")
                            try:
                                await send_telegram_message(
                                    f"🤖 *AI Trading Bot - New Order* 🤖\n\n"
                                    f"Symbol: {symbol}\n"
                                    f"Direction: {consensus_signal}\n"
                                    f"Entry: {entry_price}\n"
                                    f"Stop Loss: {sl_price}\n"
                                    f"Volume: {volume}\n"
                                    f"AI Consensus: {len([r for r in ai_results if r.get('signal') == consensus_signal])}/{len(ai_results)}"
                                )
                                logger.info(f"Telegram notification sent successfully for {symbol} {consensus_signal} order")
                            except Exception as e:
                                logger.warning(f"Failed to send Telegram notification for {symbol} order: {e}")
                                
                            # Log to DB if available
                            try:
                                ticket_id, entry_price, confirmed_sl, confirmed_tp, timestamp = order_result
                                logger.info(f"Logging order to database: ticket={ticket_id}, entry={entry_price}")
                                await log_order_placement(
                                    timestamp=timestamp,
                                    ticket_id=ticket_id,
                                    order_type="MARKET",
                                    action=consensus_signal,
                                    symbol=symbol,
                                    volume=volume,
                                    entry_price=entry_price,
                                    initial_sl=confirmed_sl,
                                    initial_tp=confirmed_tp,
                                    consensus_log_id=None
                                )
                                logger.info(f"Order successfully logged to database for {symbol}")
                            except Exception as db_err:
                                logger.error(f"Failed to log order to database: {db_err}")
                        else:
                            logger.warning(f"Order placement failed for {symbol}")
                    except Exception as order_err:
                        logger.error(f"Order placement error for {symbol}: {order_err}", exc_info=True)

                    # End symbol processing
                    symbol_processing_time = time.monotonic() - start_time_symbol
                    logger.info(f"===== [{symbol}] Processing complete. Time: {symbol_processing_time:.2f}s =====")

                except Exception as symbol_err:
                    logger.error(f"Error in signal/entry loop for {symbol}: {symbol_err}", exc_info=True)
                    logger.warning(f"Continuing to next symbol after error in {symbol}")
                    continue

            # End of cycle for all symbols
            cycle_total_time = time.monotonic() - start_time_cycle
            logger.info(f"===== SIGNAL CHECK CYCLE #{cycle_count} COMPLETE. Total time: {cycle_total_time:.2f}s =====")
            
            # Wait until next cycle
            next_interval = SCALPING_ENTRY_INTERVAL
            logger.info(f"Waiting {next_interval} seconds until next cycle...")
            await asyncio.sleep(next_interval)
            
        except Exception as e:
            logger.error(f"Unexpected error in ai_signal_and_entry_loop: {e}", exc_info=True)
            await asyncio.sleep(30)  # Wait before retrying

async def position_management_loop():
    """
    Loop for managing existing positions (trailing stop, breakeven, etc.).
    """
    logger.info("===== STARTING POSITION MANAGEMENT LOOP =====")
    
    # Initialize counter for logging purposes
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            start_time = time.monotonic()
            current_timestamp = datetime.now(timezone.utc)
            logger.info(f"===== POSITION MANAGEMENT CYCLE #{cycle_count} at {current_timestamp} =====")
            
            # Check MT5 connection
            logger.info("Checking MT5 connection for position management...")
            if not check_mt5_connection():
                logger.warning("MT5 disconnect detected in position management. Waiting 30 seconds before retry...")
                await asyncio.sleep(30)
                continue
            logger.info("MT5 connection verified for position management")
                
            # Manage positions
            logger.info("Managing open positions...")
            positions_updated = await manage_positions()
            if positions_updated:
                logger.info(f"Successfully updated {positions_updated} positions")
            else:
                logger.info("No positions required updates")
            
            # End of cycle
            cycle_time = time.monotonic() - start_time
            logger.info(f"===== POSITION MANAGEMENT CYCLE #{cycle_count} COMPLETE. Time: {cycle_time:.2f}s =====")
            
            # Wait until next check
            logger.info(f"Waiting {MANAGE_POSITIONS_INTERVAL} seconds until next position check...")
            await asyncio.sleep(MANAGE_POSITIONS_INTERVAL)
            
        except Exception as e:
            logger.error(f"Unexpected error in position_management_loop: {e}", exc_info=True)
            logger.warning("Position management cycle failed. Waiting 10 seconds before retry...")
            await asyncio.sleep(10)  # Wait before retrying
