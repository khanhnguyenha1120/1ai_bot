"""
Main orchestration module for the AI trading bot.
Handles initialization, API key loading, and task orchestration.
"""
import asyncio
import os
import re
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import MetaTrader5 as mt5
import telegram
from telegram.constants import ParseMode

from core.utils.logging_setup import logger
from core.db_logger import log_bot_status
from core.utils.constants import (
    ACCOUNT, PASSWORD, SERVER, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    XAI_API_KEY, XAI_MODEL, OPENROUTER_API_KEY, 
    MODEL_GEMINI, MODEL_DEEPSEEK, MODEL_META, MODEL_NEMOTRON,
    MODEL_REKA, MODEL_NEMOTRON_SUPER, MODEL_QWEN,
    SCALPING_SYMBOLS, SYMBOL_SETTINGS
)
from core.models.api_key_rotator import APIKeyRotator
from core.trading_logic.market_data import check_mt5_connection
from core.trading_logic.main_loop import ai_signal_and_entry_loop, position_management_loop
from core.api_key_loader import load_api_keys_from_db

# --- Global trackers ---
ai_reversal_cooldown_tracker = {}
position_close_cooldown_tracker = {}
current_event_timestamp: Optional[datetime] = None
processed_closed_tickets = set()

# --- Utility: Escape Markdown for Telegram ---
def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\1', text)

# --- Utility: Send Telegram message ---
async def send_telegram_message(message: str):
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        escaped_message = escape_markdown_v2(message)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=escaped_message, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"[Telegram] Failed to send message: {e}")
        asyncio.create_task(log_bot_status(status="ERROR", stage="send_telegram_message", details={"error": str(e)}))

# --- Utility: MT5 initialization/check helpers ---
def init_mt5():
    try:
        if not mt5.initialize(): 
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            asyncio.create_task(log_bot_status(status="ERROR", stage="init_mt5", details={"error": str(mt5.last_error())}))
            return False
            
        logger.info(f"MT5 Initialized: Version {mt5.version()}")
        asyncio.create_task(log_bot_status(status="MT5_INIT", stage="init_mt5", details={"version": mt5.version()}))
        
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            logger.error("MT5 login() failed: Account %s, Server %s, Error %s", ACCOUNT, SERVER, mt5.last_error())
            mt5.shutdown()
            asyncio.create_task(log_bot_status(status="ERROR", stage="mt5_login", details={"account": ACCOUNT, "server": SERVER, "error": str(mt5.last_error())}))
            return False
        logger.info("MT5 login successful: Account %s, Server %s.", ACCOUNT, SERVER)
        asyncio.create_task(log_bot_status(status="MT5_LOGIN", stage="mt5_login", details={"account": ACCOUNT, "server": SERVER}))
        
        enabled_count = 0
        for symbol in SCALPING_SYMBOLS:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Could not enable symbol {symbol} on init. Might require manual enable in MT5.")
                asyncio.create_task(log_bot_status(status="SYMBOL_ENABLE_FAILED", stage="mt5_symbol_enable", details={"symbol": symbol}))
            else:
                logger.info(f"Symbol {symbol} enabled in MarketWatch.")
                asyncio.create_task(log_bot_status(status="SYMBOL_ENABLED", stage="mt5_symbol_enable", details={"symbol": symbol}))
                enabled_count += 1
                
        if enabled_count == 0 and len(SCALPING_SYMBOLS) > 0:
            logger.error("No specified symbols from settings could be enabled.")
            mt5.shutdown()
            asyncio.create_task(log_bot_status(status="ERROR", stage="mt5_symbol_enable", details={"error": "No symbols enabled"}))
            return False
        elif enabled_count < len(SCALPING_SYMBOLS):
            logger.warning("Not all symbols specified in settings could be enabled.")
            asyncio.create_task(log_bot_status(status="WARNING", stage="mt5_symbol_enable", details={"enabled": enabled_count, "total": len(SCALPING_SYMBOLS)}))
            
        return True
        
    except Exception as e:
        logger.error(f"MT5 init exception: {e}", exc_info=True)
        asyncio.create_task(log_bot_status(status="ERROR", stage="init_mt5_exception", details={"error": str(e)}))
        return False

# --- API Key Management ---
async def initialize_api_key_rotators():
    """
    Initialize API key rotators for different AI models.
    """
    try:
        logger.info("Starting API key rotator initialization...")
        await log_bot_status(status="ROTATOR_INIT", stage="api_key_rotator", details={"event": "start"})
        # Try to load API keys from database first
        logger.info("Attempting to load API keys from database...")
        db_keys = await load_api_keys_from_db()
        logger.info(f"Database API keys loaded: {len(db_keys)} key types found")
        await log_bot_status(status="ROTATOR_DB_KEYS", stage="api_key_rotator", details={"db_keys": list(db_keys.keys())})
        # Initialize with default keys if DB keys not available
        gemini_keys = db_keys.get('gemini', [])
        if not gemini_keys:
            gemini_keys = [os.getenv("GEMINI_API_KEY", "")]
            logger.warning("Using default Gemini API key from environment")
            await log_bot_status(status="ROTATOR_GEMINI_ENV", stage="api_key_rotator", details={})
        else:
            logger.info(f"Using {len(gemini_keys)} Gemini API keys from database")
            await log_bot_status(status="ROTATOR_GEMINI_DB", stage="api_key_rotator", details={"count": len(gemini_keys)})
        openrouter_keys = db_keys.get('openrouter', [])
        if not openrouter_keys:
            openrouter_keys = [OPENROUTER_API_KEY]
            logger.warning("Using default OpenRouter API key from environment")
            await log_bot_status(status="ROTATOR_OPENROUTER_ENV", stage="api_key_rotator", details={})
        else:
            logger.info(f"Using {len(openrouter_keys)} OpenRouter API keys from database")
            await log_bot_status(status="ROTATOR_OPENROUTER_DB", stage="api_key_rotator", details={"count": len(openrouter_keys)})
        # Create rotators
        logger.info("Creating API key rotators...")
        gemini_rotator = None
        if any(gemini_keys):
            logger.info(f"Initializing Gemini API key rotator with {len(gemini_keys)} keys...")
            gemini_rotator = APIKeyRotator(gemini_keys, interval_seconds=60, api_type="gemini", model_name=MODEL_GEMINI)
            logger.info("Starting Gemini API key rotator...")
            await gemini_rotator.start()
            logger.info("Gemini API key rotator initialized and started successfully")
            await log_bot_status(status="ROTATOR_GEMINI_STARTED", stage="api_key_rotator", details={})
        else:
            logger.warning("No Gemini API keys available. Gemini models will be disabled.")
            await log_bot_status(status="ROTATOR_GEMINI_DISABLED", stage="api_key_rotator", details={})
        # Create OpenRouter rotators for different models
        logger.info("Creating OpenRouter API key rotators...")
        openrouter_rotators = {}
        if any(openrouter_keys):
            logger.info(f"Will create rotators for {len([MODEL_DEEPSEEK, MODEL_META, MODEL_NEMOTRON, MODEL_REKA, MODEL_NEMOTRON_SUPER, MODEL_QWEN])} OpenRouter models")
            for model_name in [MODEL_DEEPSEEK, MODEL_META, MODEL_NEMOTRON, MODEL_REKA, MODEL_NEMOTRON_SUPER, MODEL_QWEN]:
                logger.info(f"Initializing OpenRouter API key rotator for {model_name}...")
                rotator = APIKeyRotator(openrouter_keys, interval_seconds=60, api_type="openrouter", model_name=model_name)
                logger.info(f"Starting OpenRouter API key rotator for {model_name}...")
                await rotator.start()
                openrouter_rotators[model_name] = rotator
                logger.info(f"OpenRouter API key rotator for {model_name} initialized and started successfully")
                await log_bot_status(status="ROTATOR_OPENROUTER_STARTED", stage="api_key_rotator", details={"model": model_name})
            logger.info(f"Created {len(openrouter_rotators)} OpenRouter API key rotators successfully")
        else:
            logger.warning("No OpenRouter API keys available. OpenRouter models will be disabled.")
            await log_bot_status(status="ROTATOR_OPENROUTER_DISABLED", stage="api_key_rotator", details={})
        await log_bot_status(status="ROTATOR_INIT_DONE", stage="api_key_rotator", details={})
        return gemini_rotator, openrouter_rotators
        
    except Exception as e:
        logger.error(f"Error initializing API key rotators: {e}", exc_info=True)
        await log_bot_status(status="ERROR", stage="api_key_rotator_exception", details={"error": str(e)})
        return None, {}

# --- Main Bot Orchestration ---
async def start_bot():
    """
    Start the AI trading bot with all necessary components.
    """
    try:
        logger.info("===== STARTING AI TRADING BOT =====")
        await log_bot_status(status="BOT_START", stage="start_bot", details={"symbols": SCALPING_SYMBOLS})
        logger.info(f"Trading symbols: {', '.join(SCALPING_SYMBOLS)}")
        
        # Initialize MetaTrader 5
        logger.info("Initializing MetaTrader 5 connection...")
        if not init_mt5():
            logger.error("Failed to initialize MetaTrader 5. Bot cannot start.")
            await log_bot_status(status="ERROR", stage="start_bot", details={"error": "MT5 init failed"})
            return False
        logger.info("MetaTrader 5 initialized successfully")
        await log_bot_status(status="MT5_READY", stage="start_bot", details={})
            
        # Initialize API key rotators
        logger.info("Initializing API key rotators...")
        gemini_rotator, openrouter_rotators = await initialize_api_key_rotators()
        logger.info(f"API key rotators initialized: Gemini: {'Yes' if gemini_rotator else 'No'}, OpenRouter: {len(openrouter_rotators)} models")
        await log_bot_status(status="ROTATORS_READY", stage="start_bot", details={"gemini": bool(gemini_rotator), "openrouter": len(openrouter_rotators)})
        
        # Start trading tasks
        logger.info("Starting AI signal and entry loop task...")
        signal_task = asyncio.create_task(
            ai_signal_and_entry_loop(gemini_rotator, openrouter_rotators)
        )
        logger.info("AI signal and entry loop task started")
        
        logger.info("Starting position management loop task...")
        position_task = asyncio.create_task(
            position_management_loop()
        )
        logger.info("Position management loop task started")
        
        # Send startup notification
        logger.info("Sending Telegram startup notification...")
        try:
            await send_telegram_message("🤖 *AI Trading Bot Started* 🤖\n\nMonitoring symbols: " + ", ".join(SCALPING_SYMBOLS))
            logger.info("Telegram notification sent successfully")
            await log_bot_status(status="TELEGRAM_SENT", stage="start_bot", details={})
        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")
            await log_bot_status(status="TELEGRAM_FAIL", stage="start_bot", details={"error": str(e)})
        
        logger.info("All bot components started successfully")
        await log_bot_status(status="BOT_STARTED", stage="start_bot", details={})
        
        # Wait for tasks to complete (they should run indefinitely)
        logger.info("Waiting for trading tasks to run...")
        await asyncio.gather(signal_task, position_task)
        
        return True
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=True)
        await log_bot_status(status="ERROR", stage="start_bot_exception", details={"error": str(e)})
        return False

# --- Entry Point ---
def main():
    """
    Main entry point for the bot.
    """
    try:
        # Run the bot
        asyncio.run(start_bot())
        asyncio.run(log_bot_status(status="BOT_EXIT", stage="main", details={"exit": "normal"}))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        asyncio.run(log_bot_status(status="BOT_EXIT", stage="main", details={"exit": "user_interrupt"}))
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        asyncio.run(log_bot_status(status="ERROR", stage="main_exception", details={"error": str(e)}))

if __name__ == "__main__":
    main()
