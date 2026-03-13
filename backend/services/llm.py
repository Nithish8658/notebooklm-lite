import os
import time
from google import genai
from google.genai import errors
from fastapi import HTTPException
from dotenv import load_dotenv
from metrics import log_metric
from functools import lru_cache

# ===== ENV =====
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

# Create client explicitly
client = genai.Client(api_key=GEMINI_API_KEY)

# Use the stable model name
MODEL_ID = "gemini-3.1-flash-lite-preview"  # Standard stable model for 2026

# Create shared clients
client = genai.Client(api_key=GEMINI_API_KEY)

@lru_cache(maxsize=128)
def _call_gemini_cached(prompt: str) -> str:
    """Internal cached version of the call."""
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=prompt,
        config={"automatic_function_calling": {"disable": True}}
    )
    
    # Log token usage if available
    usage = getattr(response, 'usage_metadata', None)
    if usage:
        log_metric({
            "category": "llm",
            "operation": "generate_content",
            "model_name": MODEL_ID,
            "duration_ms": 0,
            "metadata": {
                "prompt_tokens": usage.prompt_token_count,
                "candidates_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
                "cached": False
            }
        })
    
    return response.text

def call_gemini(prompt: str, use_cache: bool = True) -> str:
    """Synchronous version for backward compatibility."""
    try:
        if use_cache:
            return _call_gemini_cached(prompt)
        
        # Non-cached path
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        return response.text
    except errors.APIError as e:
        # Log the exact native error from Gemini API
        log_metric({
            "category": "llm",
            "operation": "generate_content_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": "APIError",
                "status_code": getattr(e, "code", None),
                "message": str(e)
            }
        })
        
        # Raise HTTPException with native status code
        raise HTTPException(
            status_code=getattr(e, "code", 502),
            detail=f"Gemini API Error: {str(e)}"
        )
    except Exception as e:
        # Fallback for other errors (network, etc.)
        log_metric({
            "category": "llm",
            "operation": "generate_content_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": type(e).__name__,
                "message": str(e)
            }
        })
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected Gemini Error: {str(e)}"
        )
async def call_gemini_async(prompt: str) -> str:
    """
    AC-10: Async non-cached Gemini call for concurrent usage.
    Reuses a single client connection for performance.
    """
    t_call_start = time.time()
    try:
        # Re-using the aio client from the global client instance
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        t_call_end = time.time()

        # Log token usage if available
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            print(f"      - Gemini usage: Prompt {usage.prompt_token_count} tokens, Candidates {usage.candidates_token_count} tokens (Total: {usage.total_token_count})")
            log_metric({
                "category": "llm",
                "operation": "generate_content",
                "model_name": MODEL_ID,
                "duration_ms": round((t_call_end - t_call_start) * 1000, 2),
                "metadata": {
                    "prompt_tokens": usage.prompt_token_count,
                    "candidates_tokens": usage.candidates_token_count,
                    "total_tokens": usage.total_token_count
                }
            })

        return response.text
    except errors.APIError as e:
        status_code = getattr(e, "code", 502)
        log_metric({
            "category": "llm",
            "operation": "generate_content_async_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": "APIError",
                "status_code": status_code,
                "message": str(e)
            }
        })
        raise HTTPException(status_code=status_code, detail=f"Gemini Async Error: {str(e)}")
    except Exception as e:
        log_metric({
            "category": "llm",
            "operation": "generate_content_async_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": type(e).__name__,
                "message": str(e)
            }
        })
        raise HTTPException(status_code=500, detail=f"Unexpected Async Error: {str(e)}")

async def call_gemini_stream(prompt: str):
    """
    AC-09: Async generator for streaming Gemini responses (Stateless/Non-blocking).
    """
    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=MODEL_ID,
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except errors.APIError as e:
        status_code = getattr(e, "code", None)
        log_metric({
            "category": "llm",
            "operation": "generate_content_stream_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": "APIError",
                "status_code": status_code,
                "message": str(e)
            }
        })
        print(f"Gemini Streaming API Error [{status_code}]: {e}")
        yield ""
    except Exception as e:
        log_metric({
            "category": "llm",
            "operation": "generate_content_stream_error",
            "model_name": MODEL_ID,
            "metadata": {
                "error_type": type(e).__name__,
                "message": str(e)
            }
        })
        print(f"Unexpected Streaming error: {e}")
        yield ""
