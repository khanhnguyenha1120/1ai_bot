"""
Market data handling functions for the AI trading bot.
Includes functions for fetching and processing market data from MetaTrader 5.
"""
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import pandas_ta as ta
import numpy as np
import MetaTrader5 as mt5

from core.utils.logging_setup import logger
from core.utils.constants import (
    DEFAULT_CANDLES_TO_REQUEST, ADDITIONAL_LOOKBACK_BUFFER,
    EMA_FAST_PERIOD, EMA_SLOW_PERIOD, STOCH_K_PERIOD, STOCH_D_PERIOD,
    BBANDS_PERIOD, BBANDS_STDDEV, RSI_PERIOD, ATR_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER
)

def check_mt5_connection() -> bool:
    """
    Check if MetaTrader 5 is connected.
    
    Returns:
        True if connected, False otherwise
    """
    logger.info("Checking MetaTrader 5 connection...")
    if not mt5.initialize():
        logger.error("MT5 initialize() failed - terminal may not be running")
        return False
        
    if not mt5.terminal_info():
        logger.error("MT5 terminal_info() failed - connection may be unstable")
        return False
        
    terminal_info = mt5.terminal_info()._asdict()
    logger.info(f"MT5 connected successfully. Terminal: {terminal_info.get('name')}, Build: {terminal_info.get('build')}, Connected: {terminal_info.get('connected')}")
    return True

def format_indicator_value(value, decimal_places: int = 2):
    """
    Safely formats indicator values, returning N/A for NaNs.
    
    Args:
        value: The value to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted value or "N/A" for NaN values
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimal_places}f}"

def fetch_ohlc_data(symbol: str, timeframe: int, num_candles: int = DEFAULT_CANDLES_TO_REQUEST) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data from MetaTrader 5.
    
    Args:
        symbol: The trading symbol
        timeframe: The timeframe to fetch
        num_candles: Number of candles to request
        
    Returns:
        DataFrame with OHLC data or None if failed
    """
    try:
        # Get timeframe name for logging
        tf_name = str(timeframe)
        for tf_const, tf_val in vars(mt5).items():
            if tf_const.startswith('TIMEFRAME_') and tf_val == timeframe:
                tf_name = tf_const.replace('TIMEFRAME_', '')
                break
                
        logger.info(f"Fetching {tf_name} data for {symbol} ({num_candles} candles requested, buffer: {ADDITIONAL_LOOKBACK_BUFFER})")
        
        # Request more candles than needed to account for non-trading days
        start_time = time.monotonic()
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles + ADDITIONAL_LOOKBACK_BUFFER)
        fetch_time = time.monotonic() - start_time
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get {tf_name} data for {symbol} (time: {fetch_time:.2f}s)")
            return None
            
        logger.info(f"Received {len(rates)} candles of {tf_name} data for {symbol} (time: {fetch_time:.2f}s)")
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Keep only the requested number of candles
        if len(df) > num_candles:
            df = df.iloc[-num_candles:]
            logger.info(f"Trimmed to {len(df)} candles of {tf_name} data for {symbol}")
            
        # Log date range
        if not df.empty:
            start_date = df.index.min().strftime('%Y-%m-%d %H:%M')
            end_date = df.index.max().strftime('%Y-%m-%d %H:%M')
            logger.info(f"{symbol} {tf_name} data range: {start_date} to {end_date}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    """
    Calculate technical indicators for the given DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        digits: Number of digits for the symbol
        
    Returns:
        DataFrame with added indicators
    """
    try:
        start_time = time.monotonic()
        logger.info(f"Calculating technical indicators for {len(df)} candles (digits: {digits})...")
        
        # Make a copy to avoid modifying the original
        df_with_indicators = df.copy()
        
        # EMAs
        df_with_indicators[f'ema_{EMA_FAST_PERIOD}'] = ta.ema(df_with_indicators['close'], length=EMA_FAST_PERIOD)
        df_with_indicators[f'ema_{EMA_SLOW_PERIOD}'] = ta.ema(df_with_indicators['close'], length=EMA_SLOW_PERIOD)
        
        # Stochastic
        stoch = ta.stoch(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], 
                         k=STOCH_K_PERIOD, d=STOCH_D_PERIOD)
        df_with_indicators['stoch_k'] = stoch['STOCHk_14_3_3']
        df_with_indicators['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Bollinger Bands
        bbands = ta.bbands(df_with_indicators['close'], length=BBANDS_PERIOD, std=BBANDS_STDDEV)
        df_with_indicators['bb_upper'] = bbands['BBU_20_2.0']
        df_with_indicators['bb_middle'] = bbands['BBM_20_2.0']
        df_with_indicators['bb_lower'] = bbands['BBL_20_2.0']
        
        # RSI
        df_with_indicators['rsi'] = ta.rsi(df_with_indicators['close'], length=RSI_PERIOD)
        
        # ATR
        df_with_indicators['atr'] = ta.atr(df_with_indicators['high'], df_with_indicators['low'], 
                                          df_with_indicators['close'], length=ATR_PERIOD)
        
        # MACD
        macd = ta.macd(df_with_indicators['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df_with_indicators['macd'] = macd[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df_with_indicators['macd_signal'] = macd[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df_with_indicators['macd_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        
        # SuperTrend
        supertrend = ta.supertrend(df_with_indicators['high'], df_with_indicators['low'], 
                                  df_with_indicators['close'], length=SUPERTREND_PERIOD, 
                                  multiplier=SUPERTREND_MULTIPLIER)
        df_with_indicators['supertrend'] = supertrend[f'SUPERT_{SUPERTREND_PERIOD}_{SUPERTREND_MULTIPLIER}']
        df_with_indicators['supertrend_direction'] = supertrend[f'SUPERTd_{SUPERTREND_PERIOD}_{SUPERTREND_MULTIPLIER}']
        
        # Round all indicator values to the appropriate number of digits
        for col in df_with_indicators.columns:
            if col not in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                df_with_indicators[col] = df_with_indicators[col].round(digits)
        
        calc_time = time.monotonic() - start_time
        logger.info(f"Indicators calculated successfully in {calc_time:.2f}s. Added {len(df_with_indicators.columns) - 7} indicators.")
                
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return df

def calculate_daily_pivots(symbol: str, digits: int) -> Dict[str, float]:
    """
    Calculate daily pivot points for the given symbol.
    
    Args:
        symbol: The trading symbol
        digits: Number of digits for the symbol
        
    Returns:
        Dictionary with pivot points or None if failed
    """
    try:
        logger.info(f"Calculating daily pivot points for {symbol}...")
        start_time = time.monotonic()
        
        # Get daily data
        logger.info(f"Fetching daily data for {symbol} pivot calculation...")
        daily_data = fetch_ohlc_data(symbol, mt5.TIMEFRAME_D1, 2)
        if daily_data is None or len(daily_data) < 2:
            logger.error(f"Failed to get daily data for {symbol} pivot calculation")
            return None
            
        logger.info(f"Successfully retrieved {len(daily_data)} daily candles for {symbol} pivot calculation")
        
        # Calculate pivot points
        # Use previous day's data (index -2)
        prev_day = daily_data.iloc[-2]
        high, low, close = prev_day['high'], prev_day['low'], prev_day['close']
        
        logger.info(f"Using previous day's data for {symbol} pivot calculation: H={high}, L={low}, C={close}")
        
        pivot = round((high + low + close) / 3, digits)
        r1 = round((2 * pivot) - low, digits)
        r2 = round(pivot + (high - low), digits)
        r3 = round(high + 2 * (pivot - low), digits)
        s1 = round((2 * pivot) - high, digits)
        s2 = round(pivot - (high - low), digits)
        s3 = round(low - 2 * (high - pivot), digits)
        
        pivot_points = {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }
        
        calc_time = time.monotonic() - start_time
        logger.info(f"Daily pivot points calculated for {symbol} in {calc_time:.2f}s: PP={pivot}, R1={r1}, S1={s1}")
        return pivot_points
        
    except Exception as e:
        logger.error(f"Error calculating pivot points for {symbol}: {e}", exc_info=True)
        return None

def fetch_multi_timeframe_data(symbol: str, digits: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple timeframes and calculate indicators.
    
    Args:
        symbol: The trading symbol
        digits: Number of digits for the symbol
        
    Returns:
        Dictionary with DataFrames for each timeframe
    """
    timeframes = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    start_time = time.monotonic()
    logger.info(f"Fetching multi-timeframe data for {symbol} across {len(timeframes)} timeframes...")
    
    result = {}
    successful_timeframes = 0
    
    # Fetch data for each timeframe
    for tf_name, tf_value in timeframes.items():
        logger.info(f"Processing {tf_name} timeframe for {symbol}...")
        
        df = fetch_ohlc_data(symbol, tf_value)
        if df is not None:
            df_with_indicators = calculate_indicators(df, digits)
            result[tf_name] = df_with_indicators
            successful_timeframes += 1
            logger.info(f"Successfully processed {tf_name} data for {symbol} with indicators")
        else:
            logger.error(f"Failed to fetch {tf_name} data for {symbol}")
    
    total_time = time.monotonic() - start_time
    logger.info(f"Multi-timeframe data fetching complete for {symbol}: {successful_timeframes}/{len(timeframes)} timeframes in {total_time:.2f}s")
            
    return result

def prepare_ai_prompt(symbol: str, dataframes: Dict[str, pd.DataFrame], daily_pivots: Dict[str, float]) -> str:
    """
    Prepare a prompt for AI models with market data.
    
    Args:
        symbol: The trading symbol
        dataframes: Dictionary with DataFrames for each timeframe
        daily_pivots: Dictionary with pivot points
        
    Returns:
        Formatted prompt string
    """
    try:
        start_time = time.monotonic()
        logger.info(f"Preparing AI prompt for {symbol} with {len(dataframes)} timeframes and pivot data...")
        
        # Log available timeframes
        timeframes_str = ", ".join(dataframes.keys())
        logger.info(f"Available timeframes for {symbol}: {timeframes_str}")
        prompt = f"=== {symbol} MARKET ANALYSIS ===\n\n"
        
        # Current price info
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            prompt += f"CURRENT PRICE: Bid={tick.bid}, Ask={tick.ask}\n\n"
        
        # Add pivot points
        if daily_pivots:
            prompt += "DAILY PIVOT POINTS:\n"
            prompt += f"R3: {daily_pivots['r3']}\n"
            prompt += f"R2: {daily_pivots['r2']}\n"
            prompt += f"R1: {daily_pivots['r1']}\n"
            prompt += f"PP: {daily_pivots['pivot']}\n"
            prompt += f"S1: {daily_pivots['s1']}\n"
            prompt += f"S2: {daily_pivots['s2']}\n"
            prompt += f"S3: {daily_pivots['s3']}\n\n"
        
        # Add data for each timeframe
        for tf_name, df in dataframes.items():
            if df is not None and not df.empty:
                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2] if len(df) > 1 else None
                
                prompt += f"=== {tf_name} TIMEFRAME ===\n"
                prompt += f"Last candle: O={last_candle['open']}, H={last_candle['high']}, L={last_candle['low']}, C={last_candle['close']}\n"
                
                # Add indicator values
                prompt += "INDICATORS:\n"
                prompt += f"EMA(9): {format_indicator_value(last_candle.get('ema_9'))}\n"
                prompt += f"EMA(20): {format_indicator_value(last_candle.get('ema_20'))}\n"
                prompt += f"RSI(14): {format_indicator_value(last_candle.get('rsi'))}\n"
                prompt += f"Stoch K: {format_indicator_value(last_candle.get('stoch_k'))}\n"
                prompt += f"Stoch D: {format_indicator_value(last_candle.get('stoch_d'))}\n"
                prompt += f"MACD: {format_indicator_value(last_candle.get('macd'))}\n"
                prompt += f"MACD Signal: {format_indicator_value(last_candle.get('macd_signal'))}\n"
                prompt += f"MACD Hist: {format_indicator_value(last_candle.get('macd_hist'))}\n"
                prompt += f"BB Upper: {format_indicator_value(last_candle.get('bb_upper'))}\n"
                prompt += f"BB Middle: {format_indicator_value(last_candle.get('bb_middle'))}\n"
                prompt += f"BB Lower: {format_indicator_value(last_candle.get('bb_lower'))}\n"
                prompt += f"ATR(14): {format_indicator_value(last_candle.get('atr'))}\n"
                
                # Add SuperTrend direction
                supertrend_dir = last_candle.get('supertrend_direction')
                if supertrend_dir is not None:
                    direction = "UP (Bullish)" if supertrend_dir > 0 else "DOWN (Bearish)"
                    prompt += f"SuperTrend: {direction}\n"
                
                prompt += "\n"
        
        # Add trading instructions
        prompt += "TRADING INSTRUCTIONS:\n"
        prompt += "1. Analyze the provided data across all timeframes.\n"
        prompt += "2. Look for strong trend confirmation across multiple indicators.\n"
        prompt += "3. Consider pivot points for support/resistance levels.\n"
        prompt += "4. Provide your trading signal as 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'.\n"
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error preparing AI prompt for {symbol}: {e}")
        return f"Error preparing market data: {str(e)}"
