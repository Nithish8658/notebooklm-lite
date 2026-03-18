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

# Load all 10 keys into a pool
API_KEY_POOL = []
for i in range(1, 11):
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEY_POOL.append(key)

# Fallback to single key if pool is empty
if not API_KEY_POOL:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        API_KEY_POOL.append(GEMINI_API_KEY)

if not API_KEY_POOL:
    raise RuntimeError("No Gemini API keys found in .env (Expected GEMINI_API_KEY_1 to _10)")

print(f"LLM: Initialized Round-Robin pool with {len(API_KEY_POOL)} keys.")

# Global counter for rotation
_key_counter = 0

def get_next_api_key() -> str:
    global _key_counter
    key = API_KEY_POOL[_key_counter % len(API_KEY_POOL)]
    _key_counter += 1
    return key

# Create a default client (using the first key)
client = genai.Client(api_key=API_KEY_POOL[0])

# Use the stable model name
MODEL_ID = "gemini-3.1-flash-lite-preview"

async def call_gemini_async(prompt: str, custom_api_key: str = None) -> str:
    """
    AC-10: Async Gemini call with Global Round-Robin distribution.
    """
    t_call_start = time.time()
    
    # Selection: Custom Key > Round-Robin Pool
    selected_key = custom_api_key or get_next_api_key()
    target_client = genai.Client(api_key=selected_key)
        
    try:
        response = await target_client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        t_call_end = time.time()
        duration_ms = round((t_call_end - t_call_start) * 1000, 2)

        # Log usage
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            log_metric({
                "category": "llm",
                "operation": "generate_content",
                "model_name": MODEL_ID,
                "duration_ms": duration_ms,
                "metadata": {
                    "prompt_tokens": usage.prompt_token_count,
                    "candidates_tokens": usage.candidates_token_count,
                    "total_tokens": usage.total_token_count,
                    "key_used": selected_key[:10] + "..." 
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
