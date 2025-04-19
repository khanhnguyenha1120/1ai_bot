"""
Telegram Utilities (migrated): Sending messages, Markdown escaping, and helpers for notifications.
"""
import logging
import re
import os
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("[Telegram] Bot token or chat ID not set. Skipping message.")
        return
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        escaped_message = escape_markdown_v2(message)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=escaped_message, parse_mode=ParseMode.MARKDOWN_V2)
        logging.info("[Telegram] Message sent.")
    except Exception as e:
        logging.error(f"[Telegram] Failed to send message: {e}")
