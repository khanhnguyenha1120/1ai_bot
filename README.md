# AI Trading Bot

An advanced algorithmic trading bot that uses multiple AI models to generate trading signals for MetaTrader 5.

## Features

- Multi-timeframe technical analysis
- AI-powered trading signals from multiple models (GPT-4o, Gemini, xAI/Grok, OpenRouter models)
- Consensus-based decision making
- Automated position management (breakeven, trailing stop)
- Detailed logging system
- Database integration for trade history
- Telegram notifications

## Architecture

The bot is structured into several core modules:

- **API Clients**: Interfaces with various AI providers (OpenRouter, Gemini, xAI, G4F)
- **Trading Logic**: Main trading loops, market data processing, order management
- **Signal Processing**: AI consensus evaluation and signal normalization
- **Position Management**: Trailing stops, breakeven points, take profits
- **Database Logging**: Records trades, signals, and performance metrics
- **Reversal Monitor**: Monitors open positions for potential reversals

## Setup

1. Install MetaTrader 5 and required Python packages
2. Configure API keys in environment variables or config file
3. Adjust trading parameters in config.py
4. Run main.py to start the bot

## Configuration

Trading parameters can be configured in the constants section, including:

- Trading symbols and lot sizes
- Stop loss and take profit settings
- Indicator parameters
- API model selection
- Consensus requirements

## Usage

```bash
python main.py
```

## Requirements

- Python 3.9+
- MetaTrader 5
- pandas, numpy, pandas_ta
- Various AI API clients

## License

Private use only - not licensed for redistribution
