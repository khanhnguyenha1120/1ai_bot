"""
AI Signal and Consensus Logic
Migrated from ai_trading_v2.py: includes model call functions for direction, normalization, and helpers.
"""
import asyncio
import logging
from core.db_logger import log_bot_status
from datetime import datetime, timezone
from typing import Optional

# Import model/config from core orchestrator
from core.bot_orchestrator import (
    MODEL_GPT4O_G4F, XAI_MODEL, XAI_API_KEY, gemini_rotator, 
    escape_markdown_v2, send_telegram_message
)

# --- GPT-4o (g4f) Direction Call ---
async def call_gpt4o_api_for_direction(prompt: str, symbol: str) -> str:
    await log_bot_status(status="AI_MODEL_CALL", stage="call_gpt4o_api_for_direction", details={"symbol": symbol, "model": "gpt-4o-g4f"})
    logger = logging.getLogger("AISignal")
    logger.info(f"--- Calling gpt-4o API (g4f) for {symbol} ---")
    logger.debug(f"gpt-4o (g4f) request | symbol: {symbol} | prompt: {prompt[:200]}")
    try:
        from g4f.client import AsyncClient
        client = AsyncClient()
        messages = [
            {"role": "system", "content": f"You are an expert scalping trading assistant AI for {symbol}. Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."},
            {"role": "user", "content": prompt}
        ]
        response = await client.chat.completions.create(
            model=MODEL_GPT4O_G4F,
            messages=messages,
        )
        logger.debug(f"gpt-4o (g4f) raw response for {symbol}: {str(response)[:500]}")
        final_content = getattr(response.choices[0].message, 'content', None)
        logger.debug(f"gpt-4o (g4f) response content for {symbol}: {final_content}")
        if final_content:
            norm_output = normalize_ai_output(final_content)
            logger.info(f"gpt-4o (g4f) Response Interpreted for {symbol}: {norm_output}")
            await log_bot_status(status="AI_MODEL_RESULT", stage="call_gpt4o_api_for_direction", details={"symbol": symbol, "model": "gpt-4o-g4f", "result": norm_output})
            return norm_output
        else:
            logger.warning(f"gpt-4o (g4f) response for {symbol} no content.")
            await log_bot_status(status="AI_MODEL_NO_CONTENT", stage="call_gpt4o_api_for_direction", details={"symbol": symbol, "model": "gpt-4o-g4f"})
            return "INVALID_OUTPUT"
    except Exception as e:
        logger.error(f"Generic Error calling gpt-4o API (g4f) for {symbol}: {e}", exc_info=True)
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_gpt4o_api_for_direction", details={"symbol": symbol, "model": "gpt-4o-g4f", "error": str(e)})
        return f"Error: Generic g4f fail"

# --- xAI (Grok) Direction Call ---
async def call_xai_api_for_direction(prompt: str, symbol: str) -> str:
    await log_bot_status(status="AI_MODEL_CALL", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL})
    logger = logging.getLogger("AISignal")
    logger.info(f"--- Calling xAI API ({XAI_MODEL}) for {symbol} ---")
    logger.debug(f"xAI request | symbol: {symbol} | model: {XAI_MODEL} | prompt: {prompt[:200]}")
    if not XAI_API_KEY or "YOUR_XAI_API_KEY_HERE" in XAI_API_KEY:
        logger.error("XAI_API_KEY is not configured."); return "Error: XAI Key missing"
    try:
        from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
        messages = [
            {"role": "system", "content": f"You are an expert scalping trading assistant AI for {symbol}. Analyze provided data. Decide on immediate BUY/SELL scalping opportunity. Respond ONLY with 'TYPE: BUY', 'TYPE: SELL', or 'TYPE: NO SIGNAL'."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        completion = await asyncio.to_thread(
            client.chat.completions.create, model=XAI_MODEL, messages=messages,
            temperature=0.7, stream=False
        )
        logger.debug(f"xAI raw response for {symbol}: {str(completion)[:500]}")
        final_content = getattr(completion.choices[0].message, 'content', None)
        logger.debug(f"xAI response content for {symbol}: {final_content}")
        if final_content:
            norm_output = normalize_ai_output(final_content)
            logger.info(f"xAI Response Interpreted for {symbol}: {norm_output}")
            await log_bot_status(status="AI_MODEL_RESULT", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "result": norm_output})
            return norm_output
        else:
            logger.warning(f"xAI response for {symbol} no content.")
            await log_bot_status(status="AI_MODEL_NO_CONTENT", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL})
            return "INVALID_OUTPUT"
    except AuthenticationError as e:
        logger.error(f"xAI Auth Error {symbol}: {e}")
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "error": f"Auth: {e}"})
        return f"Error: xAI Auth fail"
    except RateLimitError as e:
        logger.error(f"xAI Rate Limit {symbol}: {e}")
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "error": f"RateLimit: {e}"})
        return f"Error: xAI Rate Limit"
    except APIConnectionError as e:
        logger.error(f"xAI Connection Error {symbol}: {e}")
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "error": f"Connect: {e}"})
        return f"Error: xAI Connect fail"
    except APIStatusError as e:
        logger.error(f"xAI Status Error {symbol}: Status={e.status_code}")
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "error": f"Status: {e.status_code}"})
        return f"Error: xAI Status {e.status_code}"
    except Exception as e:
        logger.error(f"Generic Error calling xAI API for {symbol}: {e}", exc_info=True)
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_xai_api_for_direction", details={"symbol": symbol, "model": XAI_MODEL, "error": str(e)})
        return f"Error: Generic xAI fail"

# --- Gemini Direction Call ---
async def call_gemini_api_for_direction(prompt: str, symbol: str) -> str:
    await log_bot_status(status="AI_MODEL_CALL", stage="call_gemini_api_for_direction", details={"symbol": symbol, "model": "gemini"})
    logger = logging.getLogger("AISignal")
    logger.info(f"--- Calling Gemini API for {symbol} ---")
    logger.debug(f"Gemini request | symbol: {symbol} | prompt: {prompt[:200]}")
    logger.info(f"--- Calling Gemini API with charts for {symbol} ---")
    if not gemini_rotator:
        logger.error(f"No Gemini rotator available for {symbol}")
        return "Error: No Gemini rotator"
    client = gemini_rotator.get_model_for_usage()
    if not client:
        logger.error(f"Failed to get Gemini client from rotator for {symbol}")
        return "Error: No Gemini client available"
    # Placeholder for Gemini API call logic
    try:
        # TODO: Implement Gemini API call logic here
        # Return a normalized output for now
        await log_bot_status(status="AI_MODEL_RESULT", stage="call_gemini_api_for_direction", details={"symbol": symbol, "model": "gemini", "result": "NO_SIGNAL"})
        return "NO_SIGNAL"
    finally:
        gemini_rotator.release_model_usage()

# --- Save AI response by model ---
def save_ai_response_by_model(response: str, model_id: str, folder: str = "ai_responses"):
    import os
    os.makedirs(folder, exist_ok=True)
    safe_model_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
    filename = f"{safe_model_id}.txt"
    filepath = os.path.join(folder, filename)
    content = f"{response}\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)

# --- OpenRouter API Direction Call ---
async def call_openrouter_api_for_direction(prompt: str, symbol: str, model_id: str, rotator) -> str | Exception:
    await log_bot_status(status="AI_MODEL_CALL", stage="call_openrouter_api_for_direction", details={"symbol": symbol, "model": model_id})
    """
    Calls the specified OpenRouter model via the OpenAI-compatible API for trading direction.
    Automatically detects if the model should use reasoning based on model_id.
    Returns normalized AI response string ("BUY", "SELL", "NO SIGNAL", "INVALID_OUTPUT") or Exception.
    """
    logger = logging.getLogger("AISignal")
    REASONING_MODELS = [
        "llama-4", "llama-3.1", "llama-3.3", "nemotron-ultra", "nemotron-super",
        "reka-flash", "deepseek-chat", "claude-3", "gpt-4", "gemini-pro", "qwen-2.5-"
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
    import time
    start_time = time.monotonic()
    current_key_info = f"...{rotator._current_key[-4:]}" if getattr(rotator, '_current_key', None) else 'N/A'
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
        if completion and getattr(completion, 'choices', None) and len(completion.choices) > 0:
            final_content = getattr(completion.choices[0].message, 'content', None)
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} is None or has empty choices.")
            final_content = None
        if final_content:
            save_ai_response_by_model(final_content, model_id)
            norm_output = normalize_ai_output(final_content)
            await log_bot_status(status="AI_MODEL_RESULT", stage="call_openrouter_api_for_direction", details={"symbol": symbol, "model": model_id, "result": norm_output})
            return norm_output
        else:
            logger.warning(f"OpenRouter ({model_id}) response for {symbol} had no content.")
            await log_bot_status(status="AI_MODEL_NO_CONTENT", stage="call_openrouter_api_for_direction", details={"symbol": symbol, "model": model_id})
            save_ai_response_by_model("INVALID_OUTPUT", model_id)
            return "INVALID_OUTPUT"
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"Error calling OpenRouter API ({model_id}, {symbol}): {type(e).__name__} - {e} in {elapsed:.2f}s", exc_info=False)
        await log_bot_status(status="AI_MODEL_ERROR", stage="call_openrouter_api_for_direction", details={"symbol": symbol, "model": model_id, "error": str(e)})
        save_ai_response_by_model(f"Exception: {type(e).__name__} - {e}", model_id)
        return e
    finally:
        if rotator:
            rotator.release_model_usage()
            logger.debug(f"Released OpenRouter client usage for model {model_id} (Symbol: {symbol})")

# --- Output Normalization Helper (improved) ---
def normalize_ai_output(output) -> str:
    # Optionally log normalization events
    # asyncio.create_task(log_bot_status(status="AI_NORMALIZE", stage="normalize_ai_output", details={"output": str(output)[:100]}))
    """
    Normalizes AI response string or handles exceptions.
    Accepts:
    - TYPE: BUY
    - TYPE: SELL
    - TYPE: NO SIGNAL / TYPE: NO_SIGNAL
    If not matching, returns "NO SIGNAL".
    Handles dict, exception, and string.
    """
    logger = logging.getLogger("AISignal")
    logger.debug(f"Normalizing AI output: {str(output)[:200]}")
    if isinstance(output, dict):
        if output:
            signal = next(iter(output.values()))
            return normalize_ai_output(signal)
        else:
            return "INVALID_OUTPUT"
    if isinstance(output, Exception):
        logger.warning(f"AI call resulted in exception: {output}")
        return f"Error: {type(output).__name__}"
    if not isinstance(output, str) or not output.strip():
        return "INVALID_OUTPUT"
    output_lines = output.strip().split('\n')
    for line in output_lines:
        line_upper = line.strip().upper()
        if line_upper == "TYPE: BUY":
            return "BUY"
        elif line_upper == "TYPE: SELL":
            return "SELL"
        elif line_upper == "TYPE: NO SIGNAL" or line_upper == "TYPE: NO_SIGNAL":
            return "NO SIGNAL"
    # fallback for ambiguous output
    logger.warning(f"AI response unclear or unexpected format: '{str(output)[:100]}' -> NO SIGNAL")
    return "NO_SIGNAL"


# --- Consensus Aggregation and Voting Helpers ---
from collections import Counter
from typing import Dict, Any, Tuple

def aggregate_ai_signals(model_results_dict: Dict[str, Dict[str, Any]], required_consensus: int = 3) -> Tuple[str, int, Dict[str, int]]:
    # Optionally log consensus aggregation
    # asyncio.create_task(log_bot_status(status="AI_CONSENSUS", stage="aggregate_ai_signals", details={"results": model_results_dict}))
    """
    Aggregates AI model results, counts votes, and determines consensus decision.
    Args:
        model_results_dict: Dict of model_key -> {"signal": ..., "status": ...}
        required_consensus: How many votes are needed for consensus
    Returns:
        consensus_decision: "BUY", "SELL", or "NO SIGNAL"
        consensus_count: Number of votes for the consensus_decision
        decision_counts: Counter dict of all signals
    """
    logger = logging.getLogger("AISignal")
    logger.debug(f"Aggregating AI signals: {model_results_dict} | required_consensus: {required_consensus}")
    decision_counts = Counter()
    for result_detail in model_results_dict.values():
        signal = result_detail.get("signal")
        if signal in ["BUY", "SELL", "NO SIGNAL"]:
            decision_counts[signal] += 1
    logger.debug(f"AI decision counts: {decision_counts}")
    consensus_decision = "NO SIGNAL"
    consensus_count = 0
    if decision_counts["BUY"] >= required_consensus:
        consensus_decision = "BUY"; consensus_count = decision_counts["BUY"]
        logger.info(f"Consensus reached: BUY ({consensus_count})")
    elif decision_counts["SELL"] >= required_consensus:
        consensus_decision = "SELL"; consensus_count = decision_counts["SELL"]
        logger.info(f"Consensus reached: SELL ({consensus_count})")
    else:
        logger.info(f"No consensus reached. Decision counts: {decision_counts}")
    return consensus_decision, consensus_count, dict(decision_counts)


def summarize_ai_votes(model_results_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Returns a human-readable summary of votes and statuses for logging/telegram.
    """
    lines = [
        f"- {name}: {details.get('signal', 'NO_RESULT')} (Status: {details.get('status', 'N/A')})"
        for name, details in model_results_dict.items()
    ]
    summary = "\n".join(lines)
    return summary


def count_valid_and_invalid_signals(model_results_dict: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """
    Returns the count of valid signals (BUY, SELL, NO SIGNAL) and invalid/error responses.
    """
    valid = sum(1 for d in model_results_dict.values() if d.get("signal") in ["BUY", "SELL", "NO SIGNAL"])
    invalid = len(model_results_dict) - valid
    return valid, invalid

