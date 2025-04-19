import asyncio
import logging
import sys
from core import trading

def main():
    try:
        # Nếu trading.py có hàm run_bot hoặc tương tự thì gọi ở đây
        # Nếu không, gọi hàm chính orchestrator hoặc trading loop
        if hasattr(trading, 'run_bot'):
            asyncio.run(trading.run_bot())
        elif hasattr(trading, 'main'):
            asyncio.run(trading.main())
        else:
            print("Không tìm thấy hàm run_bot hoặc main trong core.trading. Hãy kiểm tra lại module.")
            sys.exit(1)
    except Exception as e:
        logging.exception(f"Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
