"""
Logging configuration for the AI trading bot.
"""
import logging
import sys

def setup_logging():
    """
    Configure and set up logging for the AI trading bot.
    Returns the configured logger.
    """
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
    return logger

# Create and export the logger
logger = setup_logging()
