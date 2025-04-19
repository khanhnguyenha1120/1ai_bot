"""
OpenRouter API client for the AI trading bot.
Handles communication with various AI models through OpenRouter.
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

from core.utils.logging_setup import logger
from core.utils.constants import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from core.models.api_key_rotator import APIKeyRotator

# Global variables for response tracking
openrouter_responses = []

def save_ai_response_by_model(response: str, model_id: str):
    """
    Save AI response for tracking and analysis.
    
    Args:
        response: The AI response text
        model_id: Identifier for the AI model
    """
    global openrouter_responses
    openrouter_responses.append({
        "model": model_id,
        "response": response,
        "timestamp": time.time()
    })
    # Keep only the last 100 responses
    if len(openrouter_responses) > 100:
        openrouter_responses = openrouter_responses[-100:]

async def call_openrouter_api_for_direction(prompt: str, symbol: str, model_id: str, rotator: APIKeyRotator) -> str:
    """
    Call the OpenRouter API to get a trading direction.
    
    Args:
        prompt: The prompt to send to the API
        symbol: The trading symbol
        model_id: The model ID to use
        rotator: The API key rotator
        
    Returns:
        The normalized AI response string ("BUY", "SELL", "NO SIGNAL") or an error message
    """
    REASONING_MODELS = [
        "llama-4", "llama-3.1", "llama-3.3", "nemotron-ultra", "nemotron-super", "reka-flash",
        "deepseek-chat", "claude-3", "gpt-4", "gemini-pro", "qwen-2.5-",
    ]
    is_reasoning_model = any(pattern in model_id.lower() for pattern in REASONING_MODELS)
    logger.info(f"--- Calling OpenRouter API ({model_id}) for {symbol} {'with reasoning capabilities' if is_reasoning_model else 'direct signal'} ---")
    
    if not rotator:
        logger.error(f"No OpenRouter rotator provided for model {model_id} (Symbol: {symbol}).")
        return ConnectionAbortedError(f"No rotator for {model_id}")
        
    client = rotator.get_model_for_usage()
    if not client:
        logger.error(f"Failed to get OpenRouter client from rotator for model {model_id} (Symbol: {symbol}).")
        if rotator:
            rotator.release_model_usage()
        return ConnectionAbortedError(f"No client from rotator for {model_id}")
        
    start_time = time.monotonic()
    current_key_info = f"...{rotator._current_key[-4:]}" if rotator._current_key else 'N/A'
    logger.debug(f"Using OpenRouter key {current_key_info} for {symbol} model {model_id}")
    
    try:
        async def api_call_task():
            if is_reasoning_model:
                system_content = (
                    f"You are an expert scalping trading assistant AI for {symbol}. "
                    f"Analyze the provided data carefully and provide your reasoning. "
                    f"Conclude with a clear trading signal by writing 'TYPE: BUY', 'TYPE: SELL', "
                    f"or 'TYPE: NO SIGNAL' on a new line at the end of your response."
                )
                max_tokens = 1024
            else:
                system_content = (
                    f"You are an expert scalping trading assistant AI for {symbol}. "
                    f"Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. "
                    f"Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."
                )
                max_tokens = 50
                
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=25.0
            )
            return completion
            
        completion = await asyncio.wait_for(api_call_task(), timeout=30.0)
        response_time = time.monotonic() - start_time
        logger.info(f"OpenRouter API ({model_id}) response time for {symbol}: {response_time:.2f} seconds")
        
        if completion and completion.choices and len(completion.choices) > 0:
            final_content = getattr(completion.choices[0].message, 'content', None)
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} is None or has empty choices.")
            final_content = None
            
        if final_content:
            logger.info(f"OpenRouter ({model_id}) response received for {symbol}: {final_content[:80]}...")
            save_ai_response_by_model(final_content, model_id)
            
            # Extract the signal from the response
            if "TYPE: BUY" in final_content.upper():
                logger.info(f"OpenRouter ({model_id}) signal for {symbol}: BUY")
                return "BUY"
            elif "TYPE: SELL" in final_content.upper():
                logger.info(f"OpenRouter ({model_id}) signal for {symbol}: SELL")
                return "SELL"
            elif "TYPE: NO SIGNAL" in final_content.upper():
                logger.info(f"OpenRouter ({model_id}) signal for {symbol}: NO SIGNAL")
                return "NO SIGNAL"
            else:
                logger.warning(f"OpenRouter ({model_id}) response for {symbol} did not contain expected format: {final_content[:100]}...")
                return "INVALID_OUTPUT"
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} had no content.")
            save_ai_response_by_model("INVALID_OUTPUT", model_id)
            return "INVALID_OUTPUT"
            
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling OpenRouter API ({model_id}, {symbol}): {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=True)
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", model_id)
        return e
        
    finally:
        if rotator:
            rotator.release_model_usage()
            logger.info(f"Released OpenRouter client usage for model {model_id} (Symbol: {symbol})")
