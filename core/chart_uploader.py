import asyncio
import logging
import os
import io
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
from dotenv import load_dotenv
import aiohttp

load_dotenv()
logger = logging.getLogger("ChartUploader")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID", "3e68a21a7ad8d21")
IMGUR_UPLOAD_URL = "https://api.imgur.com/3/image"

async def generate_closed_order_chart(
        symbol: str, entry_time: datetime, close_time: datetime,
        entry_price: float, close_price: float, order_type: str,
        ticket_id: int,
        save_to_disk: bool = False,
        charts_dir: str = "closed_charts"
) -> Optional[bytes]:
    """
    Generate a basic candlestick chart for a closed order, showing entry/exit points.
    """
    logger.info(f"Generating BASIC closed order chart for {symbol} Ticket {ticket_id}")
    logger.debug(f"Chart generation params | entry_time: {entry_time}, close_time: {close_time}, entry_price: {entry_price}, close_price: {close_price}, order_type: {order_type}, save_to_disk: {save_to_disk}, charts_dir: {charts_dir}")
    if save_to_disk:
        try:
            os.makedirs(charts_dir, exist_ok=True)
            logger.debug(f"Created directory '{charts_dir}' for saving charts.")
        except OSError as error:
            logger.error(f"Cannot create directory '{charts_dir}': {error}")
            save_to_disk = False
    fig = None
    buf = None
    try:
        timeframe_code = mt5.TIMEFRAME_M1
        tf_str = "M1"
        buffer_seconds = 60 * 5
        if entry_time.tzinfo is None: entry_time = entry_time.replace(tzinfo=timezone.utc)
        else: entry_time = entry_time.astimezone(timezone.utc)
        if close_time.tzinfo is None: close_time = close_time.replace(tzinfo=timezone.utc)
        else: close_time = close_time.astimezone(timezone.utc)
        ts_from = int(entry_time.timestamp()) - buffer_seconds
        ts_to = int(close_time.timestamp()) + buffer_seconds
        logger.debug(f"Ticket {ticket_id}: Fetching M1 data from {datetime.fromtimestamp(ts_from, tz=timezone.utc)} to {datetime.fromtimestamp(ts_to, tz=timezone.utc)}")
        rates = await asyncio.to_thread(mt5.copy_rates_range, symbol, timeframe_code, ts_from, ts_to)
        logger.debug(f"Fetched {len(rates) if rates is not None else 0} M1 bars for {symbol} (Ticket {ticket_id})")
        if rates is None or len(rates) < 5:
            logger.error(f"Insufficient M1 data ({len(rates) if rates is not None else 'None'}) for chart Ticket {ticket_id}.")
            return None
        df = pd.DataFrame(rates)
        logger.debug(f"DataFrame shape after loading M1 rates: {df.shape}")
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        try:
            entry_idx_actual = df.index.get_indexer([entry_time], method='nearest')[0]
            entry_time_on_chart = df.index[entry_idx_actual]
            logger.debug(f"Found entry_time_on_chart: {entry_time_on_chart} for ticket {ticket_id}")
        except IndexError:
            logger.warning(f"Could not find nearest index for entry_time {entry_time} in chart data for {ticket_id}. Using first data point.")
            entry_time_on_chart = df.index[0]
        try:
            close_idx_actual = df.index.get_indexer([close_time], method='nearest')[0]
            close_time_on_chart = df.index[close_idx_actual]
            logger.debug(f"Found close_time_on_chart: {close_time_on_chart} for ticket {ticket_id}")
        except IndexError:
            logger.warning(f"Could not find nearest index for close_time {close_time} in chart data for {ticket_id}. Using last data point.")
            close_time_on_chart = df.index[-1]
        if close_time_on_chart < entry_time_on_chart:
            logger.debug(f"Swapping entry/close time on chart for ticket {ticket_id} (close_time_on_chart < entry_time_on_chart)")
            entry_time_on_chart, close_time_on_chart = close_time_on_chart, entry_time_on_chart
        display_start_time = entry_time_on_chart - timedelta(seconds=buffer_seconds)
        display_end_time = close_time_on_chart + timedelta(seconds=buffer_seconds)
        df_display = df[(df.index >= display_start_time) & (df.index <= display_end_time)].copy()
        logger.debug(f"df_display shape after filtering: {df_display.shape}")
        if df_display.empty:
            logger.error(f"DataFrame empty after time filtering for basic chart Ticket {ticket_id}.")
            return None
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')
        digits = 5
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            digits = symbol_info.digits
        hlines_dict = dict(
            hlines=[entry_price, close_price],
            colors=['blue', 'red'],
            linestyle='--',
            linewidths=0.8,
            alpha=0.7
        )
        vlines_dict = dict(
            vlines=[entry_time_on_chart, close_time_on_chart],
            colors=['blue', 'red'],
            linestyle=':',
            linewidths=1.0,
            alpha=0.7
        )
        profit_pips = 0.0
        point = symbol_info.point if symbol_info else 0.00001
        pip_def = 10 if 'JPY' in symbol else 1
        if symbol_info and point > 0:
            profit_points = (close_price - entry_price) / point if order_type == "BUY" else (entry_price - close_price) / point
            profit_pips = round(profit_points / pip_def, 2) if pip_def > 0 else 0.0
        else:
            logger.warning(f"Cannot calculate pips accurately for {symbol} (Ticket {ticket_id}) without valid point/pip_def.")
        entry_ts_str = entry_time.strftime('%H:%M:%S')
        close_ts_str = close_time.strftime('%H:%M:%S')
        plot_title = (f"{symbol} Ticket {ticket_id} ({tf_str}) - {order_type}\n"
                      f"Entry: {entry_price:.{digits}f} ({entry_ts_str}) | Close: {close_price:.{digits}f} ({close_ts_str})\n"
                      f"Profit: {profit_pips} pips (approx)")
        plt.rcParams['figure.dpi'] = 90
        plt.rcParams['savefig.dpi'] = 90
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.autolayout'] = True
        fig, axlist = mpf.plot(
            df_display,
            type='candle',
            style=s,
            title=plot_title,
            volume=False,
            figsize=(10, 6),
            hlines=hlines_dict,
            vlines=vlines_dict,
            returnfig=True
        )
        if axlist and len(axlist) > 0:
            ax = axlist[0]
            try: entry_idx_loc = df_display.index.get_loc(entry_time_on_chart)
            except KeyError: entry_idx_loc = 0
            try: close_idx_loc = df_display.index.get_loc(close_time_on_chart)
            except KeyError: close_idx_loc = len(df_display) - 1
            ax.text(entry_idx_loc, entry_price, f' Entry', color='blue', fontsize=8, va='bottom' if order_type=="BUY" else 'top', ha='right' if entry_idx_loc > len(df_display)*0.8 else 'left', backgroundcolor='white')
            ax.text(close_idx_loc, close_price, f' Close', color='red', fontsize=8, va='bottom' if close_price > entry_price else 'top', ha='right' if close_idx_loc > len(df_display)*0.8 else 'left', backgroundcolor='white')
        if fig is None:
            logger.error(f"Ticket {ticket_id}: Figure object is None after basic plotting!")
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight', pad_inches=0.2)
        if save_to_disk:
            entry_timestamp_str = entry_time.strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{ticket_id}_{entry_timestamp_str}_basic.png"
            filepath = os.path.join(charts_dir, filename)
            try:
                fig.savefig(filepath, format='png', dpi=90, bbox_inches='tight', pad_inches=0.2)
                logger.info(f"Basic closed order chart saved: {filepath}")
            except Exception as save_err:
                logger.error(f"Failed to save basic chart to file {filepath}: {save_err}")
        plt.close(fig)
        buf.seek(0)
        buffer_size = len(buf.getvalue())
        logger.info(f"Basic chart generated successfully for Ticket {ticket_id}. Size: {buffer_size} bytes.")
        if buffer_size < 1024:
            logger.warning(f"Ticket {ticket_id}: Generated chart size ({buffer_size}) is extremely small.")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Failed to generate BASIC chart for Ticket {ticket_id}: {e}", exc_info=True)
        if buf: buf.close()
        if fig and plt.fignum_exists(fig.number):
            try: plt.close(fig)
            except Exception: pass
        return None

async def upload_to_imgur(image_bytes: bytes, symbol: str, ticket_id: int) -> Optional[str]:
    """Uploads image bytes to Imgur anonymously using multipart/form-data."""
    if not IMGUR_CLIENT_ID:
        logger.error("Imgur Client ID not configured. Cannot upload chart.")
        return None
    if not image_bytes:
        logger.error(f"Imgur Upload Aborted: image_bytes is None for ticket {ticket_id}")
        return None
    try:
        image_len = len(image_bytes)
        logger.debug(f"Imgur Upload Check: Preparing to upload for ticket {ticket_id}. Image bytes length: {image_len}")
        if image_len < 1024:
            logger.warning(f"Imgur Upload Warning: image_bytes length ({image_len}) is very small for ticket {ticket_id}.")
    except Exception as len_err:
        logger.error(f"Imgur Upload Error: Could not get length of image_bytes for ticket {ticket_id}: {len_err}")
        return None
    headers = {'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'}
    form_data = aiohttp.FormData()
    form_data.add_field('image',
                        image_bytes,
                        filename=f'chart_{symbol}_{ticket_id}_basic.png',
                        content_type='image/png')
    form_data.add_field('title', f'Order Chart {symbol} Ticket {ticket_id} (Basic)')
    form_data.add_field('description', f'Basic chart for order {ticket_id} ({symbol})')
    logger.info(f"Uploading BASIC chart for ticket {ticket_id} ({symbol}) to Imgur (using FormData)...")
    logger.debug(f"Imgur upload request metadata | symbol: {symbol}, ticket_id: {ticket_id}, image_bytes_len: {len(image_bytes) if image_bytes else 0}")
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.debug(f"Sending POST request to {IMGUR_UPLOAD_URL} for ticket {ticket_id}")
            async with session.post(IMGUR_UPLOAD_URL, headers=headers, data=form_data) as response:
                response_text = await response.text()
                logger.debug(f"Imgur Response Status: {response.status} for ticket {ticket_id}")
                logger.debug(f"Imgur Raw Response Text (first 500 chars): {response_text[:500]}")
                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        logger.debug(f"Imgur JSON response keys: {list(data.keys())}")
                        if data.get('success') and data.get('data', {}).get('link'):
                            link = data['data']['link']
                            logger.info(f"Imgur upload successful for ticket {ticket_id}. Link: {link}")
                            return link
                        else:
                            error_msg = data.get('data', {}).get('error', 'Unknown error')
                            logger.error(f"Imgur upload failed for ticket {ticket_id}. Response indicates failure: {error_msg}")
                            if 'Unknown error' in error_msg: logger.error(f"Full Imgur error response data: {data.get('data')}")
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding Imgur JSON response for ticket {ticket_id}: {e}. Response Text: {response_text}", exc_info=True)
                        return None
                else:
                    if response.status == 400:
                        logger.error(f"Imgur upload failed (400 Bad Request) for ticket {ticket_id}. Check image data validity and format. Response: {response_text[:1000]}")
                    else:
                        logger.error(f"Imgur upload failed for ticket {ticket_id}. Status: {response.status}. Response: {response_text[:500]}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"Imgur upload timed out for ticket {ticket_id}.")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"Imgur upload network error for ticket {ticket_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading chart to Imgur for ticket {ticket_id}: {e}", exc_info=True)
        return None
