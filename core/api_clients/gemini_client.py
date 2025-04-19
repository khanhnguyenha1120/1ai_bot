"""
Gemini API client for the AI trading bot.
Handles communication with Google's Gemini API for trading signal generation.
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
import google.generativeai as genai

from core.utils.logging_setup import logger
from core.models.api_key_rotator import APIKeyRotator

from config import Config

# Global variables for response tracking
gemini_responses = []

def save_ai_response_by_model(response: str, model_id: str = "gemini"):
    """
    Save AI response for tracking and analysis.
    
    Args:
        response: The AI response text
        model_id: Identifier for the AI model
    """
    global gemini_responses
    gemini_responses.append({
        "model": model_id,
        "response": response,
        "timestamp": time.time()
    })
    # Keep only the last 100 responses
    if len(gemini_responses) > 100:
        gemini_responses = gemini_responses[-100:]

async def call_gemini_api_for_direction(prompt: str, symbol: str, rotator: Optional[APIKeyRotator] = None) -> str:
    """
    Call the Gemini API to get a trading direction.
    
    Args:
        prompt: The prompt to send to the API
        symbol: The trading symbol
        rotator: The API key rotator
        
    Returns:
        The normalized AI response string ("BUY", "SELL", "NO SIGNAL") or an error message
    """
    try:
        logger.info(f"--- Calling Gemini API for {symbol} ---")
        logger.debug(f"Gemini API Request | Symbol: {symbol} | Prompt: {prompt[:200]}")
        
        if not rotator:
            logger.error(f"No Gemini rotator provided for {symbol}.")
            return "Error: No Gemini rotator available"
            
        # Get the client from the rotator
        client = rotator.get_model_for_usage()
        if not client:
            logger.error(f"Failed to get Gemini client from rotator for {symbol}.")
            return "Error: Failed to get Gemini client"
            
        start_time = time.monotonic()
        model_name = rotator._model_name
        current_key_info = f"...{rotator._current_key[-4:]}" if rotator._current_key else 'N/A'
        logger.debug(f"Using Gemini key {current_key_info} for {symbol}")
        logger.debug(f"Gemini API Model Name: {model_name}")
        
        # Create system and user prompts
        system_prompt = (
            f"You are an expert scalping trading assistant AI for {symbol}. "
            f"Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. "
            f"Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."
        )
        
        # Call the API
        async def api_call_task():
            try:
                model = client.GenerativeModel(model_name)
                response = await asyncio.to_thread(
                    model.generate_content,
                    [system_prompt, prompt]
                )
                return response
            except Exception as inner_e:
                logger.error(f"Inner Gemini API error: {inner_e}")
                raise
                
        response = await asyncio.wait_for(api_call_task(), timeout=30.0)
        response_time = time.monotonic() - start_time
        logger.info(f"Gemini API response time for {symbol}: {response_time:.2f} seconds")
        logger.debug(f"Gemini API Raw Response for {symbol}: {str(response)[:500]}")
        
        # Process the response
        if response and hasattr(response, 'text'):
            final_content = response.text
            logger.debug(f"Gemini API Response Content for {symbol}: {final_content}")
        else:
            logger.warning(f"Gemini response for {symbol} is None or has no text attribute.")
            final_content = None
        
        if final_content:
            save_ai_response_by_model(final_content, "gemini")
            # Normalize the response
            if "TYPE: BUY" in final_content.upper():
                logger.info(f"Gemini signal for {symbol}: BUY")
                return "BUY"
            elif "TYPE: SELL" in final_content.upper():
                logger.info(f"Gemini signal for {symbol}: SELL")
                return "SELL"
            elif "TYPE: NO SIGNAL" in final_content.upper():
                logger.info(f"Gemini signal for {symbol}: NO SIGNAL")
                return "NO SIGNAL"
            else:
                logger.warning(f"Gemini response for {symbol} did not contain expected format: {final_content}")
                return "INVALID_OUTPUT"
        else:
            logger.warning(f"Gemini response for {symbol} had no content.")
            save_ai_response_by_model("INVALID_OUTPUT", "gemini")
            return "INVALID_OUTPUT"
            
    except Exception as e:
        elapsed = time.monotonic() - start_time if 'start_time' in locals() else 0
        logger.error(f"Error calling Gemini API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=True)
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", "gemini")
        return f"Error: Gemini fail - {str(e)[:50]}"
        
    finally:
        if rotator:
            rotator.release_model_usage()
            logger.debug(f"Released Gemini client usage for {symbol}")
