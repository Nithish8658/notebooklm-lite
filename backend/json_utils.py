import json
import re
from typing import List, Dict, Any, Optional

def extract_json_objects(raw: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from raw LLM output.
    1. Try to find markdown code blocks.
    2. Fallback to a string-aware balanced-brace scanner.
    """
    # 1. Try markdown blocks
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    for block in fenced:
        try:
            # We assume one JSON object per code block for this project's needs
            return [json.loads(block)]
        except json.JSONDecodeError:
            pass

    # 2. String-aware scan
    objs = []
    stack = []
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = raw[start : i + 1]
                    try:
                        objs.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
                    start = None

    return objs

def safe_json_load(raw: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON from a string, falling back to extraction if direct parsing fails.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty input")
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        objs = extract_json_objects(raw)
        if not objs:
            raise ValueError("No valid JSON found")
        return objs[0]
