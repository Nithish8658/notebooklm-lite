import json
import re
from typing import List, Dict, Any, Optional, Union

def extract_json_payloads(raw: str) -> List[Union[Dict[str, Any], List[Any]]]:
    """
    Extract JSON objects OR arrays from raw LLM output.
    1. Try to find markdown code blocks.
    2. Fallback to a string-aware balanced-bracket/brace scanner.
    """
    # 1. Try markdown blocks
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    for block in fenced:
        try:
            res = json.loads(block)
            return [res]
        except json.JSONDecodeError:
            pass

    # 2. String-aware scan for either { } or [ ]
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

        # Start of structure
        if ch in ("{", "["):
            if not stack:
                start = i
            stack.append(ch)
        
        # End of structure
        elif ch in ("}", "]"):
            if stack:
                opening = stack.pop()
                # Check for mismatch (e.g., { ] )
                if (opening == "{" and ch == "}") or (opening == "[" and ch == "]"):
                    if not stack and start is not None:
                        candidate = raw[start : i + 1]
                        try:
                            objs.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            pass
                        start = None
                else:
                    # Mismatch found, reset if not nested
                    if not stack:
                        start = None

    return objs

def safe_json_load(raw: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Attempts to parse JSON from a string, falling back to extraction if direct parsing fails.
    Returns either a dictionary or a list.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty input")
    
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback to extraction
        from json_utils import extract_json_payloads
        objs = extract_json_payloads(raw)
        if not objs:
            # Last ditch effort: find first { or [ and last } or ]
            first_brace = raw.find('{')
            first_bracket = raw.find('[')
            
            start_idx = -1
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                start_idx = first_brace
                end_char = '}'
            elif first_bracket != -1:
                start_idx = first_bracket
                end_char = ']'
                
            if start_idx != -1:
                end_idx = raw.rfind(end_char)
                if end_idx > start_idx:
                    try:
                        return json.loads(raw[start_idx:end_idx+1])
                    except json.JSONDecodeError:
                        pass

            raise ValueError("No valid JSON found in LLM response")
        return objs[0]
