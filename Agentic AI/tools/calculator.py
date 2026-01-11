"""
Calculator Tools
================

Mathematical operation tools for agents.
Great for learning tool use basics.
"""

import math
import sys
sys.path.append('..')

from src.tool_registry import Tool, ToolRegistry


# =============================================================================
# Calculator Functions
# =============================================================================


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return math.pow(base, exponent)


def sqrt(number: float) -> float:
    """Calculate square root."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


def percentage(value: float, percent: float) -> float:
    """Calculate percentage of a value."""
    return value * (percent / 100)


def average(numbers: list[float]) -> float:
    """Calculate average of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)


# =============================================================================
# Tool Definitions
# =============================================================================


add_tool = Tool(
    name="add",
    description="Add two numbers together",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    function=add,
)

subtract_tool = Tool(
    name="subtract",
    description="Subtract the second number from the first",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Number to subtract from"},
            "b": {"type": "number", "description": "Number to subtract"},
        },
        "required": ["a", "b"],
    },
    function=subtract,
)

multiply_tool = Tool(
    name="multiply",
    description="Multiply two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    function=multiply,
)

divide_tool = Tool(
    name="divide",
    description="Divide the first number by the second",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Numerator (number to divide)"},
            "b": {"type": "number", "description": "Denominator (number to divide by)"},
        },
        "required": ["a", "b"],
    },
    function=divide,
)

power_tool = Tool(
    name="power",
    description="Raise a number to a power (exponentiation)",
    parameters={
        "type": "object",
        "properties": {
            "base": {"type": "number", "description": "The base number"},
            "exponent": {"type": "number", "description": "The exponent"},
        },
        "required": ["base", "exponent"],
    },
    function=power,
)

sqrt_tool = Tool(
    name="sqrt",
    description="Calculate the square root of a number",
    parameters={
        "type": "object",
        "properties": {
            "number": {"type": "number", "description": "Number to find square root of"},
        },
        "required": ["number"],
    },
    function=sqrt,
)

percentage_tool = Tool(
    name="percentage",
    description="Calculate a percentage of a value (e.g., 20% of 150)",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The base value"},
            "percent": {"type": "number", "description": "The percentage to calculate"},
        },
        "required": ["value", "percent"],
    },
    function=percentage,
)

average_tool = Tool(
    name="average",
    description="Calculate the average (mean) of a list of numbers",
    parameters={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of numbers to average",
            },
        },
        "required": ["numbers"],
    },
    function=average,
)


# =============================================================================
# Registry
# =============================================================================


# Create a pre-configured registry with all calculator tools
calculator_registry = ToolRegistry()


def get_calculator_tools() -> list[Tool]:
    """Get all calculator tools."""
    return [
        add_tool,
        subtract_tool,
        multiply_tool,
        divide_tool,
        power_tool,
        sqrt_tool,
        percentage_tool,
        average_tool,
    ]


def register_all() -> ToolRegistry:
    """Create and return a registry with all calculator tools."""
    registry = ToolRegistry()
    for tool in get_calculator_tools():
        registry.register(tool)
    return registry


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Demo the calculator tools
    print("Calculator Tools Demo")
    print("=" * 40)

    registry = register_all()

    # Test each tool
    tests = [
        ("add", {"a": 5, "b": 3}),
        ("subtract", {"a": 10, "b": 4}),
        ("multiply", {"a": 6, "b": 7}),
        ("divide", {"a": 15, "b": 3}),
        ("power", {"base": 2, "exponent": 8}),
        ("sqrt", {"number": 144}),
        ("percentage", {"value": 200, "percent": 15}),
        ("average", {"numbers": [10, 20, 30, 40, 50]}),
    ]

    for tool_name, args in tests:
        result = registry.execute(tool_name, **args)
        print(f"{tool_name}({args}) = {result.result}")
