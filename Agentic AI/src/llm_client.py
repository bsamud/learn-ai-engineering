"""
LLM Client - Multi-Provider Abstraction
=======================================

A unified interface for interacting with LLM providers (OpenAI, Anthropic).
Handles chat completions, tool use, and streaming.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from enum import Enum

# =============================================================================
# Data Classes
# =============================================================================


class Provider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class Message:
    """
    A chat message.

    Attributes
    ----------
    role : str
        Message role: 'system', 'user', 'assistant', or 'tool'
    content : str
        Message content
    name : str, optional
        Name for tool messages
    tool_call_id : str, optional
        ID for tool response messages
    tool_calls : list, optional
        Tool calls made by assistant
    """

    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI message format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic message format."""
        # Anthropic uses 'user' and 'assistant' roles
        if self.role == "system":
            # System messages handled separately in Anthropic
            return None
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    """
    A tool call from the LLM.

    Attributes
    ----------
    id : str
        Unique identifier for the tool call
    name : str
        Name of the tool to call
    arguments : dict
        Arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """
    Response from an LLM.

    Attributes
    ----------
    content : str
        Text content of the response
    tool_calls : list[ToolCall]
        Any tool calls in the response
    finish_reason : str
        Why the response ended ('stop', 'tool_calls', etc.)
    usage : dict
        Token usage information
    raw_response : Any
        The raw API response
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


# =============================================================================
# LLM Client
# =============================================================================


class LLMClient:
    """
    Unified LLM client for multiple providers.

    Provides a consistent interface for chat completions and tool use
    across OpenAI and Anthropic APIs.

    Parameters
    ----------
    provider : str
        Provider to use: 'openai' or 'anthropic'
    model : str, optional
        Model name. Defaults based on provider.
    api_key : str, optional
        API key. If not provided, reads from environment.
    temperature : float
        Sampling temperature (0-2)
    max_tokens : int
        Maximum tokens in response

    Example
    -------
    >>> client = LLMClient(provider="openai", model="gpt-4o-mini")
    >>> response = client.chat([Message(role="user", content="Hello!")])
    >>> print(response.content)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.provider = Provider(provider.lower())
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set default model based on provider
        if model is None:
            self.model = {
                Provider.OPENAI: "gpt-4o-mini",
                Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            }[self.provider]
        else:
            self.model = model

        # Initialize provider client
        self._init_client(api_key)

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _init_client(self, api_key: Optional[str]) -> None:
        """Initialize the provider-specific client."""
        if self.provider == Provider.OPENAI:
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = OpenAI(api_key=key)

        elif self.provider == Provider.ANTHROPIC:
            from anthropic import Anthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            self.client = Anthropic(api_key=key)

    def chat(
        self,
        messages: list[Message | dict],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Parameters
        ----------
        messages : list[Message | dict]
            Conversation messages
        tools : list[dict], optional
            Tool definitions for function calling
        **kwargs
            Additional provider-specific parameters

        Returns
        -------
        LLMResponse
            The model's response

        Example
        -------
        >>> response = client.chat([
        ...     Message(role="system", content="You are helpful."),
        ...     Message(role="user", content="What is 2+2?")
        ... ])
        >>> print(response.content)
        """
        # Normalize messages
        normalized = [
            m if isinstance(m, Message) else Message(**m) for m in messages
        ]

        if self.provider == Provider.OPENAI:
            return self._chat_openai(normalized, tools, **kwargs)
        else:
            return self._chat_anthropic(normalized, tools, **kwargs)

    def _chat_openai(
        self,
        messages: list[Message],
        tools: Optional[list[dict]],
        **kwargs,
    ) -> LLMResponse:
        """Handle OpenAI chat completion."""
        # Convert messages
        openai_messages = [m.to_openai_format() for m in messages]

        # Build request
        request = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add tools if provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Make request
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]

        # Parse tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        # Track usage
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        self.call_count += 1

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response,
        )

    def _chat_anthropic(
        self,
        messages: list[Message],
        tools: Optional[list[dict]],
        **kwargs,
    ) -> LLMResponse:
        """Handle Anthropic chat completion."""
        # Extract system message
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                formatted = msg.to_anthropic_format()
                if formatted:
                    anthropic_messages.append(formatted)

        # Build request
        request = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_content:
            request["system"] = system_content.strip()

        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = []
            for tool in tools:
                # Handle both OpenAI and direct formats
                if "function" in tool:
                    anthropic_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"].get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(tool)
            request["tools"] = anthropic_tools

        # Make request
        response = self.client.messages.create(**request)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        # Track usage
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.call_count += 1

        # Map stop reason
        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response,
        )

    def chat_stream(
        self,
        messages: list[Message | dict],
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion response.

        Parameters
        ----------
        messages : list[Message | dict]
            Conversation messages
        **kwargs
            Additional provider-specific parameters

        Yields
        ------
        str
            Chunks of the response as they arrive

        Example
        -------
        >>> for chunk in client.chat_stream([Message(role="user", content="Tell me a story")]):
        ...     print(chunk, end="", flush=True)
        """
        normalized = [
            m if isinstance(m, Message) else Message(**m) for m in messages
        ]

        if self.provider == Provider.OPENAI:
            yield from self._stream_openai(normalized, **kwargs)
        else:
            yield from self._stream_anthropic(normalized, **kwargs)

    def _stream_openai(
        self, messages: list[Message], **kwargs
    ) -> Generator[str, None, None]:
        """Stream from OpenAI."""
        openai_messages = [m.to_openai_format() for m in messages]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_anthropic(
        self, messages: list[Message], **kwargs
    ) -> Generator[str, None, None]:
        """Stream from Anthropic."""
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                formatted = msg.to_anthropic_format()
                if formatted:
                    anthropic_messages.append(formatted)

        request = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_content:
            request["system"] = system_content.strip()

        with self.client.messages.stream(**request) as stream:
            for text in stream.text_stream:
                yield text

    def get_usage_stats(self) -> dict:
        """
        Get usage statistics.

        Returns
        -------
        dict
            Usage stats including tokens and estimated cost.
        """
        from .utils import estimate_cost

        cost = estimate_cost(
            self.total_input_tokens,
            self.total_output_tokens,
            self.model,
        )

        return {
            "provider": self.provider.value,
            "model": self.model,
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": cost["total"],
        }

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_chat(
    prompt: str,
    system: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Quick one-off chat completion.

    Parameters
    ----------
    prompt : str
        User prompt
    system : str, optional
        System message
    provider : str
        Provider to use
    model : str, optional
        Model to use

    Returns
    -------
    str
        Model's response

    Example
    -------
    >>> response = quick_chat("What is the capital of France?")
    >>> print(response)
    The capital of France is Paris.
    """
    client = LLMClient(provider=provider, model=model)

    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=prompt))

    response = client.chat(messages)
    return response.content
