"""
G4F (GPT4Free) API client for the AI trading bot.
Handles communication with GPT4o and other models via G4F.
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
from g4f.client import AsyncClient

from core.utils.logging_setup import logger
from core.utils.constants import MODEL_GPT4O_G4F

# Global variables for response tracking
g4f_responses = []

def save_ai_response_by_model(response: str, model_id: str = "gpt4o"):
    """
    Save AI response for tracking and analysis.
    
    Args:
        response: The AI response text
        model_id: Identifier for the AI model
    """
    global g4f_responses
    g4f_responses.append({
        "model": model_id,
        "response": response,
        "timestamp": time.time()
    })
    # Keep only the last 100 responses
    if len(g4f_responses) > 100:
        g4f_responses = g4f_responses[-100:]

async def call_gpt4o_api_for_direction(prompt: str, symbol: str, model: str = MODEL_GPT4O_G4F) -> str:
    """
    Call the GPT4o API via G4F to get a trading direction.
    
    Args:
        prompt: The prompt to send to the API
        symbol: The trading symbol
        model: The model to use
        
    Returns:
        The normalized AI response string ("BUY", "SELL", "NO SIGNAL") or an error message
    """
    try:
        logger.info(f"--- Calling G4F API (model: {model}) for {symbol} ---")
        logger.debug(f"G4F API Request | Symbol: {symbol} | Model: {model} | Prompt: {prompt[:200]}")
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
        async with AsyncClient() as client:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30
            )
            
        response_time = time.monotonic() - start_time
        logger.info(f"G4F API response time for {symbol}: {response_time:.2f} seconds")
        logger.debug(f"G4F API Raw Response for {symbol}: {str(response)[:500]}")
        
        # Process the response
        if response and response.choices and len(response.choices) > 0:
            final_content = getattr(response.choices[0].message, 'content', None)
            logger.debug(f"G4F API Response Content for {symbol}: {final_content}")
        else:
            logger.warning(f"G4F response for {symbol} is None or has empty choices.")
            final_content = None
        
        if final_content:
            save_ai_response_by_model(final_content, model)
            # Normalize the response
            if "TYPE: BUY" in final_content.upper():
                logger.info(f"G4F signal for {symbol}: BUY")
                return "BUY"
            elif "TYPE: SELL" in final_content.upper():
                logger.info(f"G4F signal for {symbol}: SELL")
                return "SELL"
            elif "TYPE: NO SIGNAL" in final_content.upper():
                logger.info(f"G4F signal for {symbol}: NO SIGNAL")
                return "NO SIGNAL"
            else:
                logger.warning(f"G4F response for {symbol} did not contain expected format: {final_content}")
                return "INVALID_OUTPUT"
        else:
            logger.warning(f"G4F response for {symbol} had no content.")
            save_ai_response_by_model("INVALID_OUTPUT", model)
            return "INVALID_OUTPUT"
            
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling G4F API for {symbol}: {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=True)
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", model)
        return f"Error: g4f fail - {str(e)[:50]}"
