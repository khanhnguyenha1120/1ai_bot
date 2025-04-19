"""
Main entry point for the AI trading bot.
"""
import asyncio
import sys
import os

from core.utils.logging_setup import logger
from core.bot_orchestrator import start_bot

def main():
    """
    Main entry point for the bot.
    """
    try:
        logger.info("=== AI TRADING BOT STARTING ===")
        logger.info("Python version: %s", sys.version)
        logger.info("Current directory: %s", os.getcwd())
        logger.info("Starting main bot process...")
        
        # Run the bot
        asyncio.run(start_bot())
        
        logger.info("Bot execution completed normally")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        logger.critical("Bot terminated due to unhandled exception")
        sys.exit(1)

if __name__ == "__main__":
    main()
