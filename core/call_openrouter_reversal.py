# ===============================================================================
# === MIGRATED: call_openrouter_reversal FUNCTION ===
# ===============================================================================
import os
import time
import asyncio
from typing import Any

async def call_openrouter_reversal(prompt: str, symbol: str, model_id: str, rotator, logger) -> str | Exception:
    """
    Calls the specified OpenRouter model via the OpenAI-compatible API for reversal decision (CLOSE/HOLD).
    Returns the raw AI response string or Exception.
    """
    REASONING_MODELS = [
        "llama-4", "llama-3.1", "llama-3.3", "nemotron-ultra", "nemotron-super", "reka-flash",
        "deepseek-chat", "claude-3", "gpt-4", "gemini-pro", "qwen-2.5-",
    ]
    is_reasoning_model = any(pattern in model_id.lower() for pattern in REASONING_MODELS)
    logger.info(f"--- Calling OpenRouter API ({model_id}) for {symbol} (reversal, {'with reasoning' if is_reasoning_model else 'direct'}) ---")
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
    current_key_info = f"...{rotator._current_key[-4:]}" if hasattr(rotator, '_current_key') and rotator._current_key else 'N/A'
    logger.debug(f"Using OpenRouter key {current_key_info} for {symbol} model {model_id}")
    try:
        async def api_call_task():
            if is_reasoning_model:
                system_content = (
                    f"You are an expert trading AI for {symbol}. Analyze the provided data carefully and provide your reasoning. "
                    f"Conclude with a clear reversal signal by writing 'TYPE: CLOSE' or 'TYPE: HOLD' on a new line at the end of your response."
                )
                max_tokens = 1024
            else:
                system_content = (
                    f"You are an expert trading AI for {symbol}. Analyze provided data. "
                    f"Respond ONLY with 'TYPE: CLOSE' or 'TYPE: HOLD'."
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
                temperature=0.5,
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
            return final_content
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} had no content.")
            return "INVALID_OUTPUT"
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling OpenRouter API ({model_id}, {symbol}): {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        return e
    finally:
        if rotator:
            rotator.release_model_usage()
            logger.debug(f"Released OpenRouter client usage for model {model_id} (Symbol: {symbol})")
# ===============================================================================
# === END OF call_openrouter_reversal FUNCTION ===
# ===============================================================================
