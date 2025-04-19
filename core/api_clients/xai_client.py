"""
xAI (Grok) API client for the AI trading bot.
Handles communication with the xAI API for trading signal generation.
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

from core.utils.logging_setup import logger
from core.utils.constants import XAI_API_KEY, XAI_MODEL
from core.models.api_key_rotator import APIKeyRotator

# Global variables for response tracking
xai_responses = []

def save_ai_response_by_model(response: str, model_id: str):
    """
    Save AI response for tracking and analysis.
    
    Args:
        response: The AI response text
        model_id: Identifier for the AI model
    """
    global xai_responses
    xai_responses.append({
        "model": model_id,
        "response": response,
        "timestamp": time.time()
    })
    # Keep only the last 100 responses
    if len(xai_responses) > 100:
        xai_responses = xai_responses[-100:]

async def call_xai_api_for_direction(prompt: str, symbol: str, 
                                    api_key: str = XAI_API_KEY, 
                                    model: str = XAI_MODEL) -> str:
    """
    Call the xAI (Grok) API to get a trading direction.
    
    Args:
        prompt: The prompt to send to the API
        symbol: The trading symbol
        api_key: The xAI API key
        model: The xAI model to use
        
    Returns:
        The normalized AI response string ("BUY", "SELL", "NO SIGNAL") or an error message
    """
    try:
        logger.info(f"--- Calling xAI API for {symbol} ---")
        logger.debug(f"xAI API Request | Symbol: {symbol} | Model: {model} | Prompt: {prompt[:200]}")
        start_time = time.monotonic()
        
        # Create system and user messages
        system_content = (
            f"You are an expert scalping trading assistant AI for {symbol}. "
            f"Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. "
            f"Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Call the API
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            timeout=25.0
        )
        
        # Process the response
        response_time = time.monotonic() - start_time
        logger.info(f"xAI API response time for {symbol}: {response_time:.2f} seconds")
        logger.debug(f"xAI API Raw Response for {symbol}: {str(completion)[:500]}")
        
        if completion and completion.choices and len(completion.choices) > 0:
            final_content = getattr(completion.choices[0].message, 'content', None)
            logger.debug(f"xAI API Response Content for {symbol}: {final_content}")
        else:
            logger.warning(f"xAI response for {symbol} is None or has empty choices.")
            final_content = None
        
        if final_content:
            save_ai_response_by_model(final_content, "xAI")
            # Normalize the response
            if "TYPE: BUY" in final_content.upper():
                logger.info(f"xAI signal for {symbol}: BUY")
                return "BUY"
            elif "TYPE: SELL" in final_content.upper():
                logger.info(f"xAI signal for {symbol}: SELL")
                return "SELL"
            elif "TYPE: NO SIGNAL" in final_content.upper():
                logger.info(f"xAI signal for {symbol}: NO SIGNAL")
                return "NO SIGNAL"
            else:
                logger.warning(f"xAI response for {symbol} did not contain expected format: {final_content}")
                return "INVALID_OUTPUT"
        else:
            logger.warning(f"xAI response for {symbol} had no content.")
            save_ai_response_by_model("INVALID_OUTPUT", "xAI")
            return "INVALID_OUTPUT"
            
    except RateLimitError as rate_err:
        elapsed = time.monotonic() - start_time
        logger.error(f"xAI API rate limit exceeded for {symbol}: {rate_err} in {elapsed:.2f}s")
        save_ai_response_by_model(f"Rate limit error: {str(rate_err)[:50]}", "xAI")
        return f"Error: Rate limit - {str(rate_err)[:50]}"
        
    except AuthenticationError as auth_err:
        elapsed = time.monotonic() - start_time
        logger.error(f"xAI API authentication error for {symbol}: {auth_err} in {elapsed:.2f}s")
        save_ai_response_by_model(f"Authentication error: {str(auth_err)[:50]}", "xAI")
        return f"Error: Auth - {str(auth_err)[:50]}"
        
    except APIError as api_err:
        elapsed = time.monotonic() - start_time
        logger.error(f"xAI API error for {symbol}: {api_err} in {elapsed:.2f}s")
        save_ai_response_by_model(f"API error: {str(api_err)[:50]}", "xAI")
        return f"Error: API - {str(api_err)[:50]}"
        
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling xAI API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=True)
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", "xAI")
        return f"Error: xAI fail - {str(e)[:50]}"
