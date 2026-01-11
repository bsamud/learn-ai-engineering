"""
Utility Functions for Agentic AI
================================

Helper functions for environment setup, token counting, message formatting,
and other common operations.
"""

import os
import json
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# =============================================================================
# Environment & Configuration
# =============================================================================

def load_env(env_path: Optional[str] = None) -> dict:
    """
    Load environment variables from .env file.

    Parameters
    ----------
    env_path : str, optional
        Path to .env file. If not provided, searches in current and parent directories.

    Returns
    -------
    dict
        Dictionary with loaded API keys and their status.

    Example
    -------
    >>> env = load_env()
    >>> print(env)
    {'OPENAI_API_KEY': 'loaded', 'ANTHROPIC_API_KEY': 'loaded'}
    """
    if env_path:
        load_dotenv(env_path)
    else:
        # Search for .env file
        current = Path.cwd()
        for path in [current, current.parent, current.parent.parent]:
            env_file = path / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break

    # Check which keys are available
    keys_status = {}
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        value = os.getenv(key)
        if value:
            keys_status[key] = "loaded"
            print(f"  {key}: {'*' * 8}{value[-4:]}")
        else:
            keys_status[key] = "not found"
            print(f"  {key}: not found")

    return keys_status


def get_api_key(provider: str) -> str:
    """
    Get API key for a specific provider.

    Parameters
    ----------
    provider : str
        Provider name: 'openai' or 'anthropic'

    Returns
    -------
    str
        API key for the provider.

    Raises
    ------
    ValueError
        If API key is not found.
    """
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    if provider.lower() not in key_map:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    env_var = key_map[provider.lower()]
    key = os.getenv(env_var)

    if not key:
        raise ValueError(
            f"{env_var} not found. Please set it in your .env file or environment."
        )

    return key


# =============================================================================
# Token Counting
# =============================================================================

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.

    Parameters
    ----------
    text : str
        Text to count tokens for.
    model : str
        Model name for tokenization (affects token count).

    Returns
    -------
    int
        Number of tokens.

    Example
    -------
    >>> count_tokens("Hello, world!")
    4
    """
    try:
        import tiktoken

        # Map model names to encodings
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            # Default to cl100k_base for most models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except ImportError:
        # Fallback: rough estimate (4 chars per token)
        return len(text) // 4


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4"
) -> dict:
    """
    Estimate the cost of an API call.

    Parameters
    ----------
    input_tokens : int
        Number of input tokens.
    output_tokens : int
        Number of output tokens.
    model : str
        Model name for pricing.

    Returns
    -------
    dict
        Cost breakdown with input_cost, output_cost, and total.

    Example
    -------
    >>> estimate_cost(1000, 500, "gpt-4")
    {'input_cost': 0.03, 'output_cost': 0.06, 'total': 0.09}
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    }

    # Get pricing for model or use default
    model_lower = model.lower()
    prices = None
    for key in pricing:
        if key in model_lower:
            prices = pricing[key]
            break

    if prices is None:
        prices = {"input": 0.01, "output": 0.03}  # Default estimate

    input_cost = (input_tokens / 1000) * prices["input"]
    output_cost = (output_tokens / 1000) * prices["output"]

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total": round(input_cost + output_cost, 6),
    }


# =============================================================================
# Message Formatting
# =============================================================================

def format_messages(messages: list[dict]) -> str:
    """
    Format a list of messages for display.

    Parameters
    ----------
    messages : list[dict]
        List of message dictionaries with 'role' and 'content'.

    Returns
    -------
    str
        Formatted string representation of messages.

    Example
    -------
    >>> msgs = [{"role": "user", "content": "Hello"}]
    >>> print(format_messages(msgs))
    USER: Hello
    """
    output = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        # Handle tool calls
        if "tool_calls" in msg:
            tool_info = []
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "unknown")
                args = tc.get("function", {}).get("arguments", "{}")
                tool_info.append(f"  -> {name}({args})")
            content = "\n".join(tool_info)

        output.append(f"{role}: {content}")

    return "\n".join(output)


def create_message(role: str, content: str) -> dict:
    """
    Create a properly formatted message dictionary.

    Parameters
    ----------
    role : str
        Message role: 'system', 'user', 'assistant', or 'tool'.
    content : str
        Message content.

    Returns
    -------
    dict
        Message dictionary.
    """
    return {"role": role, "content": content}


# =============================================================================
# JSON Utilities
# =============================================================================

def safe_json_parse(text: str) -> Optional[dict]:
    """
    Safely parse JSON from text, handling common issues.

    Parameters
    ----------
    text : str
        Text that may contain JSON.

    Returns
    -------
    dict or None
        Parsed JSON or None if parsing fails.

    Example
    -------
    >>> safe_json_parse('{"key": "value"}')
    {'key': 'value'}
    >>> safe_json_parse('Invalid JSON')
    None
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        try:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            pass

    # Try to extract JSON from generic code blocks
    if "```" in text:
        try:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline > start:
                start = newline + 1
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return None


def pretty_json(obj: Any, indent: int = 2) -> str:
    """
    Convert object to pretty-printed JSON string.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    indent : int
        Indentation level.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    return json.dumps(obj, indent=indent, ensure_ascii=False)


# =============================================================================
# Display Utilities
# =============================================================================

def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_step(step_num: int, description: str) -> None:
    """Print a numbered step."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 40)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"[OK] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"[WARNING] {message}")


# =============================================================================
# Validation
# =============================================================================

def validate_tool_schema(schema: dict) -> tuple[bool, list[str]]:
    """
    Validate a JSON Schema for tool parameters.

    Parameters
    ----------
    schema : dict
        JSON Schema to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages)
    """
    errors = []

    if not isinstance(schema, dict):
        return False, ["Schema must be a dictionary"]

    if "type" not in schema:
        errors.append("Schema missing 'type' field")
    elif schema["type"] != "object":
        errors.append("Schema type must be 'object'")

    if "properties" not in schema:
        errors.append("Schema missing 'properties' field")
    elif not isinstance(schema["properties"], dict):
        errors.append("'properties' must be a dictionary")

    return len(errors) == 0, errors
