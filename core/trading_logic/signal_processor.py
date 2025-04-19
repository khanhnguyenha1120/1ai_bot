"""
Signal processing functions for the AI trading bot.
Handles consensus logic and signal normalization from multiple AI models.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from collections import Counter

from core.utils.logging_setup import logger
from core.utils.constants import AI_MODELS_TO_CALL
from core.api_clients.xai_client import call_xai_api_for_direction
from core.api_clients.openrouter_client import call_openrouter_api_for_direction
from core.api_clients.gemini_client import call_gemini_api_for_direction
from core.api_clients.g4f_client import call_gpt4o_api_for_direction

async def get_ai_signals(prompt: str, symbol: str, 
                        gemini_rotator=None, 
                        openrouter_rotators=None) -> Dict[str, str]:
    """
    Get trading signals from multiple AI models.
    
    Args:
        prompt: The prompt to send to the AI models
        symbol: The trading symbol
        gemini_rotator: Rotator for Gemini API keys
        openrouter_rotators: Dictionary of rotators for OpenRouter API keys
        
    Returns:
        Dictionary with model names as keys and signals as values
    """
    start_time = time.monotonic()
    logger.info(f"[{symbol}] Starting AI signal collection from multiple models")
    logger.debug(f"[{symbol}] Prompt length: {len(prompt)} characters")
    
    if openrouter_rotators is None:
        openrouter_rotators = {}
        logger.debug(f"[{symbol}] No OpenRouter rotators provided, using empty dictionary")
    else:
        logger.debug(f"[{symbol}] OpenRouter rotators available for models: {list(openrouter_rotators.keys())}")
    
    logger.debug(f"[{symbol}] Gemini rotator available: {'Yes' if gemini_rotator else 'No'}")
    
    # Count total models to call
    total_models = len(AI_MODELS_TO_CALL)
    logger.info(f"[{symbol}] Will attempt to call {total_models} AI models for trading signals")
        
    ai_results = {}
    successful_calls = 0
    error_calls = 0
    
    for model_key, model_cfg in AI_MODELS_TO_CALL.items():
        model_start_time = time.monotonic()
        logger.info(f"[{symbol}] Processing AI model: {model_key} | type: {model_cfg['type']}")
        try:
            if model_cfg['type'] == 'g4f':
                try:
                    logger.info(f"[{symbol}] Calling g4f API (model: {model_key})")
                    ai_result = await call_gpt4o_api_for_direction(prompt, symbol)
                    logger.debug(f"[{symbol}] Raw g4f result: {ai_result}")
                except Exception as e:
                    logger.error(f"[{symbol}] Error calling g4f API: {e}")
                    logger.debug(f"[{symbol}] g4f API call traceback", exc_info=True)
                    ai_result = f"Error: g4f fail - {str(e)[:50]}"
                    error_calls += 1
            elif model_cfg['type'] == 'xai':
                try:
                    logger.info(f"[{symbol}] Calling xAI API (model: {model_key})")
                    ai_result = await call_xai_api_for_direction(prompt, symbol)
                    logger.debug(f"[{symbol}] Raw xAI result: {ai_result}")
                except Exception as e:
                    logger.error(f"[{symbol}] Error calling xAI API: {e}")
                    logger.debug(f"[{symbol}] xAI API call traceback", exc_info=True)
                    ai_result = f"Error: xAI fail - {str(e)[:50]}"
                    error_calls += 1
            elif model_cfg['type'] == 'gemini':
                if not gemini_rotator:
                    logger.warning(f"[{symbol}] No Gemini rotator available")
                    ai_result = "NO_GEMINI_ROTATOR"
                    error_calls += 1
                else:
                    try:
                        logger.info(f"[{symbol}] Calling Gemini API (model: {model_key})")
                        ai_result = await call_gemini_api_for_direction(prompt, symbol, gemini_rotator)
                        logger.debug(f"[{symbol}] Raw Gemini result: {ai_result}")
                    except Exception as e:
                        logger.error(f"[{symbol}] Error calling Gemini API: {e}")
                        logger.debug(f"[{symbol}] Gemini API call traceback", exc_info=True)
                        ai_result = f"Error: Gemini fail - {str(e)[:50]}"
                        error_calls += 1
            elif model_cfg['type'] == 'openrouter':
                model_name = model_cfg['model_name']
                rotator = openrouter_rotators.get(model_name)
                if not rotator:
                    logger.warning(f"[{symbol}] No OpenRouter rotator for model {model_name}")
                    ai_result = "NO_OPENROUTER_ROTATOR"
                    error_calls += 1
                else:
                    try:
                        logger.info(f"[{symbol}] Calling OpenRouter API (model: {model_key}, model_name: {model_name})")
                        ai_result = await call_openrouter_api_for_direction(prompt, symbol, model_name, rotator)
                        logger.debug(f"[{symbol}] Raw OpenRouter result: {ai_result}")
                    except Exception as e:
                        logger.error(f"[{symbol}] Error calling OpenRouter API with model {model_name}: {e}")
                        logger.debug(f"[{symbol}] OpenRouter API call traceback", exc_info=True)
                        ai_result = f"Error: OpenRouter fail - {str(e)[:50]}"
                        error_calls += 1
            else:
                logger.error(f"[{symbol}] Unknown model type {model_cfg['type']} for {model_key}")
                ai_result = "INVALID_MODEL_TYPE"
                error_calls += 1
            
            # Normalize result if it's a valid string (not an error)
            if isinstance(ai_result, str) and not ai_result.startswith("Error:") and not ai_result.startswith("NO_"):
                # Check if result is valid
                if ai_result in ["BUY", "SELL", "NO SIGNAL"]:
                    logger.info(f"[{symbol}] {model_key} returned valid signal: {ai_result}")
                    successful_calls += 1
                else:
                    logger.warning(f"[{symbol}] {model_key} returned invalid signal format: {ai_result}")
                    ai_result = "INVALID_OUTPUT"
                    error_calls += 1
            
            # Store result
            ai_results[model_key] = ai_result
            model_elapsed = time.monotonic() - model_start_time
            logger.info(f"[{symbol}] {model_key} processing completed in {model_elapsed:.2f}s with result: {ai_result}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Unexpected error processing {model_key} API: {e}", exc_info=True)
            ai_results[model_key] = f"Error: Unexpected - {str(e)[:50]}"
            error_calls += 1
    
    total_elapsed = time.monotonic() - start_time
    logger.info(f"[{symbol}] AI signal collection complete in {total_elapsed:.2f}s. Success: {successful_calls}/{total_models}, Errors: {error_calls}/{total_models}")
    logger.debug(f"[{symbol}] All AI results: {ai_results}")
    return ai_results

def get_consensus_signal(ai_results: Dict[str, str], required_consensus: int = 2) -> Optional[str]:
    """
    Get consensus signal from multiple AI model results.
    
    Args:
        ai_results: Dictionary with model names as keys and signals as values
        required_consensus: Minimum number of models required for consensus
        
    Returns:
        Consensus signal ("BUY", "SELL") or None if no consensus
    """
    start_time = time.monotonic()
    logger.info(f"Evaluating consensus from {len(ai_results)} AI results (required consensus: {required_consensus})")
    
    # Filter out error results
    valid_results = {}
    error_results = {}
    for model, result in ai_results.items():
        if isinstance(result, str) and not result.startswith("Error:") and not result.startswith("NO_") and result != "INVALID_OUTPUT":
            valid_results[model] = result
        else:
            error_results[model] = result
    
    logger.info(f"Valid results: {len(valid_results)}/{len(ai_results)}, Error results: {len(error_results)}/{len(ai_results)}")
    if error_results:
        logger.debug(f"Error results: {error_results}")
    
    # Count signals from valid results
    signal_counts = Counter(valid_results.values())
    logger.info(f"Signal counts: {dict(signal_counts)}")
    
    # Calculate percentages for better insights
    total_valid = sum(signal_counts.values())
    if total_valid > 0:
        buy_percentage = (signal_counts.get("BUY", 0) / total_valid) * 100
        sell_percentage = (signal_counts.get("SELL", 0) / total_valid) * 100
        no_signal_percentage = (signal_counts.get("NO SIGNAL", 0) / total_valid) * 100
        logger.info(f"Signal percentages: BUY: {buy_percentage:.1f}%, SELL: {sell_percentage:.1f}%, NO SIGNAL: {no_signal_percentage:.1f}%")
    
    consensus_signal = None
    # Check for consensus
    for signal in ["BUY", "SELL"]:
        if signal_counts[signal] >= required_consensus:
            logger.info(f"Consensus reached for {signal} (count: {signal_counts[signal]}/{total_valid})")
            consensus_signal = signal
            break
    
    if not consensus_signal:
        logger.info(f"No consensus reached. Required: {required_consensus}, highest count: {signal_counts.most_common(1)[0] if signal_counts else 'None'}")
        # Log the models that voted for each signal
        for signal in ["BUY", "SELL", "NO SIGNAL"]:
            models_with_signal = [model for model, result in valid_results.items() if result == signal]
            if models_with_signal:
                logger.debug(f"Models voting for {signal}: {', '.join(models_with_signal)}")
    
    elapsed = time.monotonic() - start_time
    logger.debug(f"Consensus evaluation completed in {elapsed:.4f}s")
    return consensus_signal

def log_ai_results(symbol: str, ai_results: Dict[str, str], consensus_signal: Optional[str]):
    """
    Log AI results and consensus.
    
    Args:
        symbol: The trading symbol
        ai_results: Dictionary with model names as keys and signals as values
        consensus_signal: The consensus signal or None
    """
    logger.info(f"===== AI RESULTS SUMMARY FOR {symbol} =====")
    
    # Categorize results
    buy_models = []
    sell_models = []
    no_signal_models = []
    error_models = []
    invalid_models = []
    
    for model, result in ai_results.items():
        if isinstance(result, Exception):
            error_models.append(model)
            logger.warning(f"[{symbol}] {model} ERROR: {type(result).__name__} - {result}")
        elif result == "BUY":
            buy_models.append(model)
            logger.info(f"[{symbol}] {model} signal: BUY")
        elif result == "SELL":
            sell_models.append(model)
            logger.info(f"[{symbol}] {model} signal: SELL")
        elif result == "NO SIGNAL":
            no_signal_models.append(model)
            logger.info(f"[{symbol}] {model} signal: NO SIGNAL")
        elif result.startswith("Error:") or result.startswith("NO_"):
            error_models.append(model)
            logger.warning(f"[{symbol}] {model} ERROR: {result}")
        else:
            invalid_models.append(model)
            logger.warning(f"[{symbol}] {model} INVALID OUTPUT: {result}")
    
    # Log summary statistics
    total_models = len(ai_results)
    logger.info(f"[{symbol}] Total models queried: {total_models}")
    logger.info(f"[{symbol}] Models voting BUY: {len(buy_models)} ({', '.join(buy_models) if buy_models else 'None'})")
    logger.info(f"[{symbol}] Models voting SELL: {len(sell_models)} ({', '.join(sell_models) if sell_models else 'None'})")
    logger.info(f"[{symbol}] Models voting NO SIGNAL: {len(no_signal_models)} ({', '.join(no_signal_models) if no_signal_models else 'None'})")
    logger.info(f"[{symbol}] Models with errors: {len(error_models)} ({', '.join(error_models) if error_models else 'None'})")
    logger.info(f"[{symbol}] Models with invalid output: {len(invalid_models)} ({', '.join(invalid_models) if invalid_models else 'None'})")
    
    # Log consensus result
    if consensus_signal:
        logger.info(f"[{symbol}] CONSENSUS DECISION: {consensus_signal}")
    else:
        logger.info(f"[{symbol}] NO CONSENSUS REACHED")
    
    logger.info(f"===== END AI RESULTS SUMMARY FOR {symbol} =====")
