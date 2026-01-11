"""
LLM Utilities for AI Engineering
================================

Multi-provider LLM client with streaming, cost tracking, and structured outputs.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Type
from enum import Enum


# =============================================================================
# Pricing Configuration
# =============================================================================

# Pricing per 1M tokens (as of 2024)
PRICING = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Estimate API cost for a request.

    Parameters
    ----------
    input_tokens : int
        Number of input tokens
    output_tokens : int
        Number of output tokens
    model : str
        Model name

    Returns
    -------
    float
        Estimated cost in USD
    """
    # Find matching pricing
    model_lower = model.lower()
    prices = None
    for key in PRICING:
        if key in model_lower:
            prices = PRICING[key]
            break

    if prices is None:
        prices = {"input": 10.0, "output": 30.0}  # Default estimate

    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return round(input_cost + output_cost, 6)


# =============================================================================
# LLM Client
# =============================================================================


@dataclass
class UsageStats:
    """Track API usage statistics."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """Add a request to the stats."""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += estimate_cost(input_tokens, output_tokens, model)

    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"Calls: {self.total_calls} | "
            f"Tokens: {self.total_input_tokens + self.total_output_tokens:,} | "
            f"Cost: ${self.total_cost:.4f}"
        )


class LLMClient:
    """
    Multi-provider LLM client with usage tracking.

    Supports OpenAI and Anthropic APIs with a unified interface.

    Example
    -------
    >>> client = LLMClient(provider="openai", model="gpt-4o-mini")
    >>> response = client.chat("Hello!")
    >>> print(response)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = UsageStats()

        # Set default model
        if model is None:
            self.model = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-sonnet-20241022",
            }.get(self.provider, "gpt-4o-mini")
        else:
            self.model = model

        # Initialize client
        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]) -> None:
        """Initialize the provider client."""
        if self.provider == "openai":
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=key)

        elif self.provider == "anthropic":
            from anthropic import Anthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=key)

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Simple chat completion.

        Parameters
        ----------
        message : str
            User message
        system : str, optional
            System prompt
        **kwargs
            Additional parameters

        Returns
        -------
        str
            Model response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        return self.chat_messages(messages, **kwargs)

    def chat_messages(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """
        Chat completion with message history.

        Parameters
        ----------
        messages : list[dict]
            List of message dictionaries
        **kwargs
            Additional parameters

        Returns
        -------
        str
            Model response
        """
        if self.provider == "openai":
            return self._chat_openai(messages, **kwargs)
        else:
            return self._chat_anthropic(messages, **kwargs)

    def _chat_openai(self, messages: list[dict], **kwargs) -> str:
        """OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # Track usage
        if response.usage:
            self.stats.add(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                self.model,
            )

        return response.choices[0].message.content

    def _chat_anthropic(self, messages: list[dict], **kwargs) -> str:
        """Anthropic chat completion."""
        # Extract system message
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                anthropic_messages.append(msg)

        request = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if system:
            request["system"] = system.strip()

        response = self.client.messages.create(**request)

        # Track usage
        self.stats.add(
            response.usage.input_tokens,
            response.usage.output_tokens,
            self.model,
        )

        return response.content[0].text

    def stream(
        self,
        message: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion.

        Yields
        ------
        str
            Chunks of the response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        if self.provider == "openai":
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            with self.client.messages.stream(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text

    def get_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.stats = UsageStats()


# =============================================================================
# Structured Outputs
# =============================================================================


def get_json_response(
    client: LLMClient,
    message: str,
    system: Optional[str] = None,
) -> dict:
    """
    Get a JSON response from the LLM.

    Parameters
    ----------
    client : LLMClient
        The LLM client
    message : str
        User message
    system : str, optional
        System prompt

    Returns
    -------
    dict
        Parsed JSON response
    """
    full_system = (system or "") + "\nRespond with valid JSON only."

    response = client.chat(message, system=full_system)

    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        raise


def stream_response(
    client: LLMClient,
    message: str,
    system: Optional[str] = None,
    callback: Optional[callable] = None,
) -> str:
    """
    Stream response with optional callback.

    Parameters
    ----------
    client : LLMClient
        The LLM client
    message : str
        User message
    system : str, optional
        System prompt
    callback : callable, optional
        Function to call with each chunk

    Returns
    -------
    str
        Complete response
    """
    full_response = ""
    for chunk in client.stream(message, system):
        full_response += chunk
        if callback:
            callback(chunk)
        else:
            print(chunk, end="", flush=True)

    print()  # Newline at end
    return full_response
