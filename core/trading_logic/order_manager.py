"""
Order management functions for the AI trading bot.
Handles placing, modifying, and closing orders in MetaTrader 5.
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import MetaTrader5 as mt5

from core.utils.logging_setup import logger
from core.utils.constants import BOT_MAGIC_NUMBER, SYMBOL_SETTINGS
from core.db_logger import log_order_placement, log_bot_status

async def place_market_order(
    symbol: str, 
    order_type: str, 
    volume: float, 
    sl_price: float
) -> Optional[Tuple[int, float, float, float, datetime]]:
    await log_bot_status(status="ORDER_ATTEMPT", stage="order_manager_place", details={"symbol": symbol, "order_type": order_type, "volume": volume, "sl_price": sl_price})
    """
    Place a market order in MetaTrader 5.
    
    Args:
        symbol: The trading symbol
        order_type: "BUY" or "SELL"
        volume: Order volume
        sl_price: Stop loss price
        
    Returns:
        Tuple with (ticket_id, entry_price, sl_price, tp_price, timestamp) or None if failed
    """
    try:
        # Validate inputs
        if order_type not in ["BUY", "SELL"]:
            logger.error(f"Invalid order type: {order_type}")
            return None
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
            
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick info for {symbol}")
            return None
            
        # Set up order parameters
        action = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == "BUY" else tick.bid
        
        # Calculate TP based on SL (risk:reward ratio 1:2)
        sl_distance = abs(price - sl_price)
        tp_price = price + (sl_distance * 2) if order_type == "BUY" else price - (sl_distance * 2)
        
        # Round prices to symbol digits
        digits = symbol_info.digits
        price = round(price, digits)
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": BOT_MAGIC_NUMBER,
            "comment": f"AI Trading Bot {order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        logger.info(f"Sending {order_type} order for {symbol}: volume={volume}, price={price}, sl={sl_price}, tp={tp_price}")
        result = mt5.order_send(request)
        
        # Process result
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket_id = result.order
            timestamp = datetime.now(timezone.utc)
            logger.info(f"Order placed successfully: {symbol} {order_type}, Ticket: {ticket_id}")
            
            # Get the actual entry price from the order
            orders = mt5.orders_get(ticket=ticket_id)
            if orders:
                entry_price = orders[0].price_open
                confirmed_sl = orders[0].sl
                confirmed_tp = orders[0].tp
            else:
                entry_price = price
                confirmed_sl = sl_price
                confirmed_tp = tp_price
                
            await log_bot_status(status="ORDER_PLACED", stage="order_manager_place", details={"symbol": symbol, "order_type": order_type, "volume": volume, "ticket_id": ticket_id})
            return (ticket_id, entry_price, confirmed_sl, confirmed_tp, timestamp)
        else:
            error_code = result.retcode if result else "Unknown"
            error_msg = f"Failed to place order: {error_code}"
            logger.error(error_msg)
            await log_bot_status(status="ORDER_FAILED", stage="order_manager_place", details={"symbol": symbol, "order_type": order_type, "volume": volume, "error": error_msg})
            return None
            
    except Exception as e:
        logger.error(f"Error placing market order for {symbol}: {e}", exc_info=True)
        await log_bot_status(status="ERROR", stage="order_manager_place", details={"symbol": symbol, "order_type": order_type, "volume": volume, "error": str(e)})
        return None

async def modify_stop_loss(ticket_id: int, new_sl: float) -> bool:
    """
    Modify the stop loss of an existing order.
    
    Args:
        ticket_id: The order ticket ID
        new_sl: New stop loss price
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the order
        position = mt5.positions_get(ticket=ticket_id)
        if not position or len(position) == 0:
            logger.error(f"Position with ticket {ticket_id} not found")
            return False
            
        position = position[0]
        symbol = position.symbol
        symbol_info = mt5.symbol_info(symbol)
        
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False
            
        # Round the new SL to symbol digits
        digits = symbol_info.digits
        new_sl = round(new_sl, digits)
        
        # Create modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "sl": new_sl,
            "tp": position.tp,
            "position": ticket_id,
        }
        
        # Send modification request
        logger.info(f"Modifying SL for ticket {ticket_id} to {new_sl}")
        result = mt5.order_send(request)
        
        # Process result
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"SL modified successfully for ticket {ticket_id}")
            return True
        else:
            error_code = result.retcode if result else "Unknown"
            error_msg = f"Failed to modify SL: {error_code}"
            logger.error(error_msg)
            return False
            
    except Exception as e:
        logger.error(f"Error modifying SL for ticket {ticket_id}: {e}", exc_info=True)
        return False

async def close_position(ticket_id: int) -> bool:
    """
    Close an existing position.
    
    Args:
        ticket_id: The position ticket ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the position
        position = mt5.positions_get(ticket=ticket_id)
        if not position or len(position) == 0:
            logger.error(f"Position with ticket {ticket_id} not found")
            return False
            
        position = position[0]
        symbol = position.symbol
        
        # Determine close parameters
        position_type = position.type
        volume = position.volume
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick info for {symbol}")
            return False
            
        # Set price based on position type (opposite of entry)
        price = tick.bid if position_type == mt5.POSITION_TYPE_BUY else tick.ask
        
        # Create close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if position_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket_id,
            "price": price,
            "deviation": 10,
            "magic": BOT_MAGIC_NUMBER,
            "comment": "AI Trading Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close request
        logger.info(f"Closing position with ticket {ticket_id}")
        result = mt5.order_send(request)
        
        # Process result
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket_id} closed successfully")
            return True
        else:
            error_code = result.retcode if result else "Unknown"
            error_msg = f"Failed to close position: {error_code}"
            logger.error(error_msg)
            return False
            
    except Exception as e:
        logger.error(f"Error closing position {ticket_id}: {e}", exc_info=True)
        return False

async def manage_positions():
    """
    Manage existing positions (trailing stop, breakeven, etc.).
    """
    await log_bot_status(status="POSITION_MANAGE_START", stage="order_manager_manage_positions", details={})
    try:
        # Get all positions with our magic number
        logger.info(f"Checking for open positions with magic number {BOT_MAGIC_NUMBER}...")
        positions = mt5.positions_get(magic=BOT_MAGIC_NUMBER)
        if not positions:
            logger.info("No open positions found to manage")
            await log_bot_status(status="NO_POSITIONS", stage="order_manager_manage_positions", details={})
            return 0
            
        logger.info(f"Found {len(positions)} open positions to manage")
        positions_updated = 0
            
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            position_type = position.type
            entry_price = position.price_open
            current_sl = position.sl
            
            position_type_str = "BUY" if position_type == mt5.POSITION_TYPE_BUY else "SELL"
            logger.info(f"Managing {symbol} {position_type_str} position (ticket: {ticket}, entry: {entry_price}, current SL: {current_sl})")
            
            # Calculate current profit
            profit = position.profit
            logger.info(f"Current profit for {symbol} position {ticket}: {profit:.2f} USD")
            
            # Skip symbols not in our settings
            settings = SYMBOL_SETTINGS.get(symbol)
            if not settings:
                logger.warning(f"Symbol {symbol} not found in settings, skipping position management")
                continue
                
            # Get current price
            logger.info(f"Getting current price for {symbol}...")
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick info for {symbol}")
                continue
                
            current_price = tick.bid if position_type == mt5.POSITION_TYPE_BUY else tick.ask
            logger.info(f"Current price for {symbol}: {current_price}")
            
            # Get symbol info
            logger.info(f"Getting symbol info for {symbol}...")
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                continue
            logger.info(f"Symbol info retrieved for {symbol}")
                
            point = symbol_info.point
            pip_def = settings.get('pip_definition_in_points', 1)
            pip_value = point * pip_def
            logger.info(f"Symbol parameters: point={point}, pip_def={pip_def}, pip_value={pip_value}")
            
            # Calculate profit in pips
            profit_points = (current_price - entry_price) / point if position_type == mt5.POSITION_TYPE_BUY else (entry_price - current_price) / point
            profit_pips = profit_points / pip_def
            logger.info(f"Current profit for {symbol} position {ticket}: {profit_pips:.2f} pips")
            
            # Check for breakeven
            be_pips = settings.get('breakeven_pips', 10.0)
            be_target_pips = settings.get('be_target_profit_pips', 5.0)
            logger.info(f"Breakeven settings for {symbol}: trigger={be_pips} pips, target profit={be_target_pips} pips")
            
            if profit_pips >= be_pips:
                logger.info(f"Breakeven condition met for {symbol} position {ticket}: {profit_pips:.2f} pips >= {be_pips} pips")
                
                # Calculate breakeven price with small profit
                be_price = entry_price + (be_target_pips * pip_value) if position_type == mt5.POSITION_TYPE_BUY else entry_price - (be_target_pips * pip_value)
                logger.info(f"Calculated breakeven price for {symbol}: {be_price} (entry: {entry_price}, target profit: {be_target_pips} pips)")
                
                # Only modify if new SL is better than current
                if (position_type == mt5.POSITION_TYPE_BUY and (current_sl is None or be_price > current_sl)) or \
                   (position_type == mt5.POSITION_TYPE_SELL and (current_sl is None or be_price < current_sl)):
                    logger.info(f"Setting breakeven for {symbol} position {ticket} (profit: {profit_pips:.1f} pips, new SL: {be_price})")
                    result = await modify_stop_loss(ticket, be_price)
                    if result:
                        logger.info(f"Successfully set breakeven for {symbol} position {ticket}")
                        await log_bot_status(status="BREAKEVEN_SET", stage="order_manager_breakeven", details={"symbol": symbol, "ticket": ticket, "new_sl": be_price})
                        positions_updated += 1
                    else:
                        logger.warning(f"Failed to set breakeven for {symbol} position {ticket}")
                        await log_bot_status(status="BREAKEVEN_FAILED", stage="order_manager_breakeven", details={"symbol": symbol, "ticket": ticket, "attempted_sl": be_price})
                    continue  # Skip trailing check if we just set breakeven
                else:
                    logger.info(f"Breakeven already set or not beneficial for {symbol} position {ticket} (current SL: {current_sl}, calculated BE: {be_price})")
            
            # Check for trailing stop
            trailing_trigger = settings.get('trailing_trigger_pips', 15.0)
            trailing_distance = settings.get('trailing_distance_pips', 15.0)
            logger.info(f"Trailing stop settings for {symbol}: trigger={trailing_trigger} pips, distance={trailing_distance} pips")
            
            if profit_pips >= trailing_trigger:
                logger.info(f"Trailing stop condition met for {symbol} position {ticket}: {profit_pips:.2f} pips >= {trailing_trigger} pips")
                
                # Calculate new trailing stop
                trailing_price = current_price - (trailing_distance * pip_value) if position_type == mt5.POSITION_TYPE_BUY else current_price + (trailing_distance * pip_value)
                logger.info(f"Calculated trailing stop price for {symbol}: {trailing_price} (current price: {current_price}, distance: {trailing_distance} pips)")
                
                # Only modify if new SL is better than current
                if (position_type == mt5.POSITION_TYPE_BUY and (current_sl is None or trailing_price > current_sl)) or \
                   (position_type == mt5.POSITION_TYPE_SELL and (current_sl is None or trailing_price < current_sl)):
                    logger.info(f"Setting trailing stop for {symbol} position {ticket} (profit: {profit_pips:.1f} pips, new SL: {trailing_price})")
                    result = await modify_stop_loss(ticket, trailing_price)
                    if result:
                        logger.info(f"Successfully set trailing stop for {symbol} position {ticket}")
                        await log_bot_status(status="TRAILING_STOP_SET", stage="order_manager_trailing_stop", details={"symbol": symbol, "ticket": ticket, "new_sl": trailing_price})
                        positions_updated += 1
                    else:
                        logger.warning(f"Failed to set trailing stop for {symbol} position {ticket}")
                        await log_bot_status(status="TRAILING_STOP_FAILED", stage="order_manager_trailing_stop", details={"symbol": symbol, "ticket": ticket, "attempted_sl": trailing_price})
                else:
                    logger.info(f"Trailing stop already set or not beneficial for {symbol} position {ticket} (current SL: {current_sl}, calculated trailing: {trailing_price})")
            else:
                logger.info(f"Trailing stop condition not met for {symbol} position {ticket}: {profit_pips:.2f} pips < {trailing_trigger} pips")
            
    except Exception as e:
        logger.error(f"Error managing positions: {e}", exc_info=True)
        await log_bot_status(status="ERROR", stage="order_manager_manage_positions", details={"error": str(e)})
        return 0
        
    logger.info(f"Position management complete. Updated {positions_updated} positions.")
    await log_bot_status(status="POSITION_MANAGE_DONE", stage="order_manager_manage_positions", details={"positions_updated": positions_updated})
    return positions_updated
