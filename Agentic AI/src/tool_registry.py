"""
Tool Registry - Tool Definition and Management
==============================================

Provides a framework for defining, registering, and executing tools
that LLMs can use for function calling.
"""

import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints
from functools import wraps

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Tool:
    """
    A tool that can be called by an LLM.

    Attributes
    ----------
    name : str
        Unique name for the tool
    description : str
        Description of what the tool does (shown to LLM)
    parameters : dict
        JSON Schema for the tool's parameters
    function : Callable
        The Python function to execute
    examples : list[dict]
        Example usages (optional)

    Example
    -------
    >>> tool = Tool(
    ...     name="add",
    ...     description="Add two numbers",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {
    ...             "a": {"type": "number", "description": "First number"},
    ...             "b": {"type": "number", "description": "Second number"}
    ...         },
    ...         "required": ["a", "b"]
    ...     },
    ...     function=lambda a, b: a + b
    ... )
    """

    name: str
    description: str
    parameters: dict
    function: Callable
    examples: list[dict] = field(default_factory=list)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        Parameters
        ----------
        **kwargs
            Arguments to pass to the tool function

        Returns
        -------
        Any
            Result from the tool function
        """
        return self.function(**kwargs)


@dataclass
class ToolResult:
    """
    Result from executing a tool.

    Attributes
    ----------
    tool_name : str
        Name of the tool that was executed
    success : bool
        Whether execution was successful
    result : Any
        The return value (if successful)
    error : str
        Error message (if failed)
    """

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None

    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if self.success:
            if isinstance(self.result, (dict, list)):
                return json.dumps(self.result, indent=2)
            return str(self.result)
        return f"Error: {self.error}"


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """
    Registry for managing tools.

    Provides methods to register, lookup, and execute tools.

    Example
    -------
    >>> registry = ToolRegistry()
    >>> registry.register(calculator_tool)
    >>> registry.register(search_tool)
    >>> result = registry.execute("add", a=1, b=2)
    >>> print(result)
    3
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Parameters
        ----------
        tool : Tool
            Tool to register

        Raises
        ------
        ValueError
            If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        print(f"  Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            print(f"  Unregistered tool: {name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Parameters
        ----------
        name : str
            Name of the tool to execute
        **kwargs
            Arguments to pass to the tool

        Returns
        -------
        ToolResult
            Result of the tool execution
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found. Available: {self.list_tools()}",
            )

        try:
            result = tool.execute(**kwargs)
            return ToolResult(
                tool_name=name,
                success=True,
                result=result,
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
            )

    def to_openai_format(self) -> list[dict]:
        """Get all tools in OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def to_anthropic_format(self) -> list[dict]:
        """Get all tools in Anthropic format."""
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    def describe(self) -> str:
        """Get a human-readable description of all tools."""
        lines = ["Available Tools:", "=" * 40]
        for tool in self._tools.values():
            lines.append(f"\n{tool.name}")
            lines.append(f"  Description: {tool.description}")
            if tool.parameters.get("properties"):
                lines.append("  Parameters:")
                for pname, pinfo in tool.parameters["properties"].items():
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    required = pname in tool.parameters.get("required", [])
                    req_str = " (required)" if required else ""
                    lines.append(f"    - {pname}: {ptype}{req_str}")
                    if pdesc:
                        lines.append(f"        {pdesc}")
        return "\n".join(lines)


# =============================================================================
# Decorators
# =============================================================================


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator to create a Tool from a function.

    Automatically extracts parameter info from type hints and docstring.

    Parameters
    ----------
    name : str, optional
        Tool name. Defaults to function name.
    description : str, optional
        Tool description. Defaults to function docstring.

    Example
    -------
    >>> @tool(name="calculator_add")
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers together.'''
    ...     return a + b
    ...
    >>> add.tool  # Access the Tool object
    >>> add(1, 2)  # Still works as normal function
    3
    """

    def decorator(func: Callable) -> Callable:
        # Get function info
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"

        # Build parameters from type hints
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        sig = inspect.signature(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type info
            param_type = hints.get(param_name, str)
            json_type = _python_type_to_json(param_type)

            # Get description from docstring (if available)
            param_desc = _extract_param_doc(func.__doc__, param_name)

            properties[param_name] = {
                "type": json_type,
                "description": param_desc,
            }

            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        # Create Tool
        tool_obj = Tool(
            name=tool_name,
            description=tool_desc.strip(),
            parameters=parameters,
            function=func,
        )

        # Wrap function to preserve both function and tool
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.tool = tool_obj
        return wrapper

    return decorator


def _python_type_to_json(py_type: type) -> str:
    """Convert Python type to JSON Schema type."""
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Handle Optional, List, etc.
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

    return type_map.get(py_type, "string")


def _extract_param_doc(docstring: Optional[str], param_name: str) -> str:
    """Extract parameter description from docstring."""
    if not docstring:
        return f"The {param_name} parameter"

    # Look for parameter in docstring (various formats)
    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        # Google/NumPy style: "param_name: description" or "param_name (type): description"
        if param_name in line and (":" in line or "--" in line):
            # Get the description part
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1 and param_name in parts[0]:
                    return parts[1].strip()
            if "--" in line:
                parts = line.split("--", 1)
                if len(parts) > 1:
                    return parts[1].strip()

    return f"The {param_name} parameter"


# =============================================================================
# Tool Builders
# =============================================================================


def create_tool(
    name: str,
    description: str,
    function: Callable,
    parameters: Optional[dict] = None,
) -> Tool:
    """
    Create a Tool with automatic parameter inference.

    Parameters
    ----------
    name : str
        Tool name
    description : str
        Tool description
    function : Callable
        Function to execute
    parameters : dict, optional
        JSON Schema for parameters. If not provided, inferred from function.

    Returns
    -------
    Tool
        The created tool

    Example
    -------
    >>> def multiply(x: int, y: int) -> int:
    ...     return x * y
    >>> tool = create_tool("multiply", "Multiply two numbers", multiply)
    """
    if parameters is None:
        # Infer from function
        hints = get_type_hints(function) if hasattr(function, "__annotations__") else {}
        sig = inspect.signature(function)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            json_type = _python_type_to_json(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"The {param_name} parameter",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        function=function,
    )


# =============================================================================
# Built-in Tools
# =============================================================================


def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_date() -> str:
    """Get the current date."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


# Create built-in tools
BUILTIN_TOOLS = {
    "current_time": Tool(
        name="current_time",
        description="Get the current date and time",
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_current_time,
    ),
    "current_date": Tool(
        name="current_date",
        description="Get the current date",
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_current_date,
    ),
}


def get_builtin_tools() -> list[Tool]:
    """Get all built-in tools."""
    return list(BUILTIN_TOOLS.values())
