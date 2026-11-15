"""Safe JSON parser and schema validator for Phase 8 LLM outputs."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_json_block(raw_text: str) -> Optional[str]:
    """Extract JSON string block from potential markdown code fences or raw text."""
    if not raw_text:
        return None

    # Check for ```json ... ``` markdown fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Find outermost curly braces
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start : end + 1].strip()

    return None


def parse_and_validate_llm_json(
    raw_content: Optional[str],
    required_keys: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Parse raw LLM response string into verified JSON dict.

    Returns:
        (is_valid, parsed_dict, error_message)
    """
    if not raw_content or not raw_content.strip():
        return False, {}, "Empty response received from LLM."

    json_str = extract_json_block(raw_content) or raw_content.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        return False, {}, f"JSON decoding error: {str(err)}"

    if not isinstance(data, dict):
        return False, {}, f"Expected JSON object (dict), got {type(data).__name__}."

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, data, f"Missing required JSON fields: {missing}"

    return True, data, None
