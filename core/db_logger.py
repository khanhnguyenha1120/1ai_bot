"""
Async database logger for AI trading bot (refactored from fx/db_logger.py).
Handles connection pooling, AI performance logging, consensus, order placement, reversal events, and trailing stops.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone # Ensure timezone is imported
from typing import Optional, List, Counter, Union, Dict, Any # Added Dict, Any
import json
import random
import asyncpg # <<< THAY THẾ psycopg
from asyncpg.pool import Pool # <<< Import Pool từ asyncpg
from dotenv import load_dotenv
from decimal import Decimal, ROUND_HALF_UP

load_dotenv()
logger = logging.getLogger("DBLogger")
DB_CONN_STRING = os.getenv("DATABASE_URL")
if not DB_CONN_STRING:
    DB_CONN_STRING = "postgresql://neondb_owner:npg_cnFZho2yt6wK@ep-hidden-fire-a4xkzbku-pooler.us-east-1.aws.neon.tech/forex?sslmode=require"

pool: Optional[Pool] = None

async def init_db_pool():
    global pool
    if not DB_CONN_STRING:
        logger.critical("Database connection string is not configured. DB logging disabled.")
        return
    try:
        if pool is None:
            logger.info(f"Initializing asyncpg database connection pool for: ...{DB_CONN_STRING[-30:]}")
            pool = await asyncpg.create_pool(
                dsn=DB_CONN_STRING,
                min_size=1,
                max_size=5,
                command_timeout=60
            )
            logger.debug("Database pool created. Testing connection...")
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                logger.debug(f"Test query result: {result}")
                if result == 1:
                    logger.info("Asyncpg database connection pool initialized and tested successfully.")
                else:
                    logger.error("Asyncpg database pool test query failed.")
                    if pool: await pool.close(); pool = None
    except Exception as e:
        logger.error(f"Failed to initialize asyncpg database pool: {e}", exc_info=True)
        if pool:
            try: await pool.close()
            except Exception as close_err: logger.error(f"Error closing pool after init failure: {close_err}")
        pool = None

async def close_db_pool():
    global pool
    if pool:
        logger.info("Closing asyncpg database connection pool...")
        try:
            await pool.close()
            logger.info("Asyncpg database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing asyncpg pool: {e}")
        finally:
            pool = None
    else:
        logger.debug("close_db_pool called, but pool was already None.")

async def log_ai_performance(
    event_timestamp: datetime, model_name: str, symbol: str, status: str,
    latency_seconds: Optional[float], signal: Optional[str],
    error_info: Optional[str] = None, raw_result_snippet: Optional[str] = None,
    api_key_used: Optional[str] = None
):
    if not pool:
        logger.warning("log_ai_performance called but pool is None. Skipping DB log.")
        return
    sql = """
        INSERT INTO ai_performance_log
        (event_timestamp, model_name, symbol, status, latency_seconds, signal, error_info, raw_result_snippet, api_key_used)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """
    logger.debug(f"log_ai_performance SQL: {sql.strip()} | params: {event_timestamp}, {model_name}, {symbol}, {status}, {latency_seconds}, {signal}, {error_info}, ...")
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(sql, event_timestamp, model_name, symbol, status, latency_seconds,
                               signal, error_info, raw_result_snippet, api_key_used)
        logger.info(f"Logged AI performance for {model_name} - {symbol} - {status} | DB result: {result}")
    except Exception as e:
        logger.error(f"DB Error logging AI performance (asyncpg): {e}", exc_info=True)

async def log_ai_consensus(
    event_timestamp: datetime, symbol: str, consensus_reached: bool,
    final_signal: Optional[str], consensus_ratio: Optional[str],
    total_models_expected: int, total_models_responded: int,
    detailed_votes: Dict[str, Dict[str, Any]]
) -> Optional[int]:
    if not pool:
        logger.warning("log_ai_consensus called but pool is None. Skipping DB log.")
        return None
    buy_votes, sell_votes, nosignal_votes, error_votes, valid_signals_count = 0, 0, 0, 0, 0
    for result in detailed_votes.values():
        signal = result.get("signal")
        status = result.get("status")
        is_valid = False
        if status == "SIGNAL_OK":
            if signal == "BUY": buy_votes += 1; is_valid = True
            elif signal == "SELL": sell_votes += 1; is_valid = True
            elif signal == "NO SIGNAL": nosignal_votes += 1; is_valid = True
            else: error_votes += 1
        else:
            error_votes += 1
        if is_valid: valid_signals_count += 1

    sql = """
        INSERT INTO ai_consensus_log
        (event_timestamp, symbol, consensus_reached, final_signal, consensus_ratio, total_models_expected, total_models_responded, buy_votes, sell_votes, nosignal_votes, error_votes, detailed_votes)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
    """
    logger.debug(f"log_ai_consensus SQL: {sql.strip()} | params: {event_timestamp}, {symbol}, {consensus_reached}, {final_signal}, ...")
    logger.debug(f"Vote breakdown: BUY={buy_votes}, SELL={sell_votes}, NO_SIGNAL={nosignal_votes}, ERROR={error_votes}, valid={valid_signals_count}")
    try:
        async with pool.acquire() as conn:
            consensus_id = await conn.fetchval(sql, event_timestamp, symbol, consensus_reached, final_signal, consensus_ratio,
                                              total_models_expected, total_models_responded, buy_votes, sell_votes, nosignal_votes, error_votes, json.dumps(detailed_votes))
            logger.info(f"Logged AI consensus for {symbol} | consensus: {consensus_reached} | final: {final_signal} | id: {consensus_id}")
            return consensus_id
    except Exception as e:
        logger.error(f"DB Error logging AI consensus (asyncpg): {e}", exc_info=True)
        return None

async def log_order_placement(
    timestamp: datetime,
    ticket_id: int,
    order_type: str,
    action: str,
    symbol: str,
    volume: float,
    entry_price: float,
    initial_sl: Optional[float],
    initial_tp: Optional[float],
    consensus_log_id: Optional[int] = None
):
    """Logs a newly placed order using asyncpg. (No initial chart URL)"""
    if not pool: return
    sql = """
        INSERT INTO orders
        (timestamp, ticket_id, order_type, action, symbol, volume, entry_price,
         initial_sl, initial_tp, consensus_log_id) -- Removed m1_chart_url
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) -- Removed $10, $11 became $10
        ON CONFLICT (ticket_id) DO NOTHING
    """
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(sql,
                timestamp, ticket_id, order_type, action, symbol, volume, entry_price,
                initial_sl, initial_tp, consensus_log_id # Removed m1_chart_url
            )
            if status.endswith(" 1"): logger.debug(f"Logged order placement: Ticket {ticket_id} ({symbol} {action})")
            elif status.endswith(" 0"): logger.warning(f"Order placement log skipped for ticket {ticket_id} (ON CONFLICT DO NOTHING triggered).")
            else: logger.info(f"Order placement status for ticket {ticket_id}: {status}")
    except Exception as e:
        logger.error(f"DB Error logging order placement (asyncpg): {e}", exc_info=False)

async def update_order_final_chart_url(ticket_id: int, chart_url: str):
    global pool
    if not pool:
        logger.error(f"DB pool not available, cannot update final chart URL for ticket {ticket_id}")
        return
    if not chart_url:
        logger.warning(f"Empty final chart URL provided for ticket {ticket_id}, skipping DB update.")
        return
    sql = """
        UPDATE orders
        SET final_chart_url = $1
        WHERE ticket_id = $2
    """
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(sql, chart_url, ticket_id)
            if status == "UPDATE 1":
                logger.info(f"Successfully updated final_chart_url for ticket {ticket_id} in DB.")
            else:
                logger.warning(f"Could not find ticket {ticket_id} in DB to update final_chart_url. Status: {status}")
    except Exception as e:
        logger.error(f"DB Error updating final_chart_url for ticket {ticket_id} (asyncpg): {e}", exc_info=False)

async def log_trailing_stop(order_ticket_id: int, symbol: str, old_sl: Optional[float], new_sl: float, profit_pips: float):
    if not pool: return
    sql = """ INSERT INTO trailing_log (order_ticket_id, symbol, old_sl, new_sl, profit_pips) VALUES ($1, $2, $3, $4, $5) """
    try:
        async with pool.acquire() as conn: await conn.execute(sql, order_ticket_id, symbol, old_sl, new_sl, profit_pips)
        logger.debug(f"Logged trailing stop for ticket {order_ticket_id}")
    except Exception as e: logger.error(f"DB Error logging trailing stop (asyncpg): {e}", exc_info=False)

async def log_reversal_event_start(order_ticket_id: int, symbol: str, code_signals: List[str], code_score: int, ai_confirmation_required: bool) -> Optional[int]:
    if not pool: return None
    sql = """
        INSERT INTO reversal_log (timestamp, order_ticket_id, symbol, code_signals, code_score, ai_confirmation_required)
        VALUES (CURRENT_TIMESTAMP, $1, $2, $3, $4, $5) RETURNING id
    """
    try:
        async with pool.acquire() as conn:
            reversal_log_id = await conn.fetchval(sql, order_ticket_id, symbol, code_signals, code_score, ai_confirmation_required)
            if reversal_log_id: logger.debug(f"Logged reversal start for ticket {order_ticket_id}. ID: {reversal_log_id}")
            else: logger.error("Failed to get reversal log ID after insert (asyncpg).")
            return reversal_log_id
    except Exception as e: logger.error(f"DB Error logging reversal start (asyncpg): {e}", exc_info=False); return None

async def log_reversal_ai_opinion(reversal_log_id: int, model_name: str, decision: str):
    if not pool: return
    sql = """ INSERT INTO reversal_ai_opinion_log (reversal_log_id, model_name, decision) VALUES ($1, $2, $3) """
    try:
        async with pool.acquire() as conn: await conn.execute(sql, reversal_log_id, model_name, decision)
        logger.debug(f"Logged AI reversal opinion for ID {reversal_log_id}: {model_name} -> {decision}")
    except Exception as e: logger.error(f"DB Error logging AI reversal opinion (asyncpg): {e}", exc_info=False)

async def update_reversal_event_end(reversal_log_id: int, final_decision: str, action_taken: str):
    if not pool: return
    sql = """ UPDATE reversal_log SET final_decision = $1, action_taken = $2 WHERE id = $3 """
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(sql, final_decision, action_taken, reversal_log_id)
            if status == "UPDATE 1": logger.debug(f"Updated reversal log ID {reversal_log_id}: Decision={final_decision}, Action={action_taken}")
            else: logger.warning(f"Could not update reversal log ID {reversal_log_id} (not found or no change?). Status: {status}")
    except Exception as e: logger.error(f"DB Error updating reversal event end (asyncpg): {e}", exc_info=False)

async def check_order_exists(ticket_id: int) -> bool:
    if not pool:
        logger.warning(f"DB pool not available, cannot check existence for ticket {ticket_id}")
        return False
    sql = "SELECT 1 FROM orders WHERE ticket_id = $1 LIMIT 1"
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval(sql, ticket_id)
            return result == 1
    except Exception as e:
        logger.error(f"DB Error checking order existence for ticket {ticket_id}: {e}", exc_info=False)
        return False

async def update_order_close_details(
    ticket_id: int,
    close_price: float,
    close_timestamp: datetime,
    profit_pips: Optional[float],
    profit_usd: Optional[Decimal], # <<< Sử dụng Decimal cho USD >>>
    close_reason: str = 'CLOSED' # Lý do mặc định nếu không xác định được
):
    """Cập nhật thông tin chi tiết khi một lệnh đóng cửa vào bảng orders."""
    global pool
    if not pool:
        logger.error(f"DB pool not available, cannot update close details for ticket {ticket_id}")
        return
    if not close_timestamp: # Cần có thời gian đóng cửa
        logger.error(f"Missing close_timestamp for ticket {ticket_id}, cannot update close details.")
        return

    # Đảm bảo close_timestamp là timezone-aware (UTC)
    if close_timestamp.tzinfo is None:
        logger.warning(f"close_timestamp for ticket {ticket_id} was naive, assuming UTC.")
        close_timestamp = close_timestamp.replace(tzinfo=timezone.utc)
    else:
        close_timestamp = close_timestamp.astimezone(timezone.utc)

    # Chuyển đổi profit_usd sang float để lưu vào DB (kiểu REAL) nếu không phải None
    profit_usd_float = float(profit_usd) if profit_usd is not None else None

    sql = """
        UPDATE orders
        SET
            close_price = $1,
            close_timestamp = $2,
            profit_pips = $3,
            profit_usd = $4,
            close_reason = $5
        WHERE ticket_id = $6
        AND close_timestamp IS NULL -- Chỉ cập nhật nếu chưa được cập nhật trước đó
    """
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(sql,
                close_price,
                close_timestamp,
                profit_pips,
                profit_usd_float, # Sử dụng giá trị float đã chuyển đổi
                close_reason,
                ticket_id
            )
            if status == "UPDATE 1":
                logger.info(f"Successfully updated close details for ticket {ticket_id} in DB.")
            elif status == "UPDATE 0":
                logger.warning(f"Could not update close details for ticket {ticket_id} in DB (maybe already updated or not found?). Status: {status}")
            else:
                 logger.warning(f"Unexpected status when updating close details for ticket {ticket_id}: {status}")

    except Exception as e:
        logger.error(f"DB Error updating close details for ticket {ticket_id} (asyncpg): {e}", exc_info=False)

async def get_active_api_keys(key_type: str) -> List[str]:
    """
    Lấy danh sách các API key đang hoạt động (is_active=TRUE) từ database
    theo loại được chỉ định (key_type) và trả về theo thứ tự ngẫu nhiên.
    """
    active_keys = []
    if not pool:
        logger.warning(f"DB pool not available, cannot fetch API keys for type '{key_type}'")
        return active_keys

    sql = """
        SELECT api_key_value
        FROM api_keys
        WHERE key_type = $1 AND is_active = TRUE
    """
    try:
        async with pool.acquire() as conn:
            records = await conn.fetch(sql, key_type)
            # Lấy danh sách key từ kết quả record
            active_keys = [record['api_key_value'] for record in records]

            # <<< THÊM BƯỚC XÁO TRỘN (SHUFFLE) >>>
            if active_keys: # Chỉ xáo trộn nếu list không rỗng
                random.shuffle(active_keys)
                logger.info(f"Fetched and *shuffled* {len(active_keys)} active API keys from DB for type '{key_type}'.")
            else:
                logger.info(f"Fetched 0 active API keys from DB for type '{key_type}'.")
            # <<< KẾT THÚC BƯỚC XÁO TRỘN >>>

    except Exception as e:
        logger.error(f"DB Error fetching/shuffling active API keys for type '{key_type}': {e}", exc_info=False)
        active_keys = [] # Trả về list rỗng khi có lỗi

    if not active_keys and key_type in ['gemini', 'openrouter']: # Chỉ cảnh báo nếu không có key cho các dịch vụ chính
         logger.warning(f"No active API keys found in database for type '{key_type}'. Rotator might not work.")

    return active_keys
