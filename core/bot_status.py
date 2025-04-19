import os
import json
import threading

STATUS_FILE = os.path.join(os.path.dirname(__file__), 'bot_status.json')
_status_lock = threading.Lock()

def set_bot_status(status: str, details: dict = None):
    """Update the bot status for web display."""
    with _status_lock:
        data = {'status': status}
        if details:
            data.update(details)
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f)

def get_bot_status():
    with _status_lock:
        if not os.path.isfile(STATUS_FILE):
            return {'status': 'unknown'}
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
