"""
Web Search Tools
================

Web search and information retrieval tools for agents.
Uses mock responses for learning - can be extended with real APIs.
"""

import json
from datetime import datetime
from typing import Optional
import sys
sys.path.append('..')

from src.tool_registry import Tool, ToolRegistry


# =============================================================================
# Mock Search Database
# =============================================================================

# Simulated search results for common queries
MOCK_SEARCH_DATABASE = {
    "python": [
        {
            "title": "Python.org - Official Website",
            "url": "https://www.python.org",
            "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
        },
        {
            "title": "Python Tutorial - W3Schools",
            "url": "https://www.w3schools.com/python/",
            "snippet": "Python is a popular programming language. Python can be used on a server to create web applications.",
        },
    ],
    "machine learning": [
        {
            "title": "Machine Learning - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "snippet": "Machine learning (ML) is a field of study in AI that focuses on algorithms that learn from data.",
        },
        {
            "title": "Google Machine Learning Crash Course",
            "url": "https://developers.google.com/machine-learning/crash-course",
            "snippet": "A self-study guide for aspiring machine learning practitioners with practical exercises.",
        },
    ],
    "llm": [
        {
            "title": "Large Language Models Explained",
            "url": "https://example.com/llm-explained",
            "snippet": "Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human language.",
        },
        {
            "title": "OpenAI - GPT Models",
            "url": "https://openai.com/gpt-4",
            "snippet": "GPT-4 is OpenAI's most advanced system, producing safer and more useful responses.",
        },
    ],
    "agents": [
        {
            "title": "AI Agents: The Future of Autonomous AI",
            "url": "https://example.com/ai-agents",
            "snippet": "AI agents are autonomous systems that can perceive, reason, and act to accomplish goals.",
        },
        {
            "title": "LangChain Agents Documentation",
            "url": "https://python.langchain.com/docs/modules/agents/",
            "snippet": "Agents use an LLM to determine which actions to take and in what order.",
        },
    ],
    "weather": [
        {
            "title": "Weather.com - Local Forecast",
            "url": "https://weather.com",
            "snippet": "Get the latest weather forecast for your area, including temperature, precipitation, and wind.",
        },
    ],
    "capital of france": [
        {
            "title": "Paris - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Paris",
            "snippet": "Paris is the capital and largest city of France, with a population of over 2 million.",
        },
    ],
}


# =============================================================================
# Search Functions
# =============================================================================


def web_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Search the web for information.

    This is a mock implementation for learning purposes.
    In production, you would integrate with a real search API.

    Parameters
    ----------
    query : str
        Search query
    num_results : int
        Maximum number of results to return

    Returns
    -------
    list[dict]
        Search results with title, url, and snippet
    """
    query_lower = query.lower()

    # Look for matching results
    results = []
    for key, data in MOCK_SEARCH_DATABASE.items():
        if key in query_lower or query_lower in key:
            results.extend(data)

    # If no direct match, return generic results
    if not results:
        results = [
            {
                "title": f"Search results for: {query}",
                "url": f"https://search.example.com/?q={query.replace(' ', '+')}",
                "snippet": f"Found information related to '{query}'. This is simulated data for learning purposes.",
            }
        ]

    return results[:num_results]


def get_webpage_content(url: str) -> str:
    """
    Fetch and extract text content from a webpage.

    This is a mock implementation for learning purposes.

    Parameters
    ----------
    url : str
        URL of the webpage to fetch

    Returns
    -------
    str
        Extracted text content
    """
    # Mock content based on URL patterns
    if "python.org" in url:
        return """
Python is a high-level, general-purpose programming language.
Its design philosophy emphasizes code readability with the use of significant indentation.

Key features:
- Easy to learn syntax
- Interpreted language
- Extensive standard library
- Object-oriented programming support
- Large community and ecosystem
"""

    if "wikipedia" in url:
        return """
This is simulated Wikipedia content.
In a real implementation, you would fetch and parse the actual webpage.
Wikipedia is a free online encyclopedia with millions of articles.
"""

    return f"Content from {url} (simulated for learning purposes)"


def get_current_weather(location: str) -> dict:
    """
    Get current weather for a location.

    This is a mock implementation for learning purposes.

    Parameters
    ----------
    location : str
        City name or location

    Returns
    -------
    dict
        Weather information
    """
    # Mock weather data
    import random

    weather_conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Clear"]

    return {
        "location": location,
        "temperature": random.randint(15, 30),
        "unit": "celsius",
        "condition": random.choice(weather_conditions),
        "humidity": random.randint(40, 80),
        "timestamp": datetime.now().isoformat(),
        "note": "This is simulated weather data for learning purposes",
    }


def get_news(topic: Optional[str] = None, num_articles: int = 3) -> list[dict]:
    """
    Get recent news articles.

    This is a mock implementation for learning purposes.

    Parameters
    ----------
    topic : str, optional
        Topic to filter news
    num_articles : int
        Number of articles to return

    Returns
    -------
    list[dict]
        News articles
    """
    mock_news = [
        {
            "title": "AI Technology Advances Rapidly in 2024",
            "source": "Tech News Daily",
            "summary": "New developments in artificial intelligence continue to reshape industries.",
            "date": "2024-01-15",
        },
        {
            "title": "Climate Summit Reaches Historic Agreement",
            "source": "World News",
            "summary": "World leaders commit to ambitious carbon reduction targets.",
            "date": "2024-01-14",
        },
        {
            "title": "Market Update: Tech Stocks Rally",
            "source": "Financial Times",
            "summary": "Technology sector leads market gains amid positive earnings reports.",
            "date": "2024-01-13",
        },
    ]

    if topic:
        # Filter by topic (simplified)
        mock_news = [n for n in mock_news if topic.lower() in n["title"].lower()]

    return mock_news[:num_articles]


# =============================================================================
# Tool Definitions
# =============================================================================


web_search_tool = Tool(
    name="web_search",
    description="Search the web for information on any topic. Returns titles, URLs, and snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 3)",
            },
        },
        "required": ["query"],
    },
    function=lambda query, num_results=3: web_search(query, num_results),
)

get_webpage_tool = Tool(
    name="get_webpage",
    description="Fetch and read the content of a webpage given its URL",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to fetch",
            },
        },
        "required": ["url"],
    },
    function=get_webpage_content,
)

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a specific location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location (e.g., 'New York', 'London')",
            },
        },
        "required": ["location"],
    },
    function=get_current_weather,
)

news_tool = Tool(
    name="get_news",
    description="Get recent news articles, optionally filtered by topic",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to filter news (optional)",
            },
            "num_articles": {
                "type": "integer",
                "description": "Number of articles to return (default: 3)",
            },
        },
        "required": [],
    },
    function=lambda topic=None, num_articles=3: get_news(topic, num_articles),
)


# =============================================================================
# Registry
# =============================================================================


search_registry = ToolRegistry()


def get_search_tools() -> list[Tool]:
    """Get all search/web tools."""
    return [
        web_search_tool,
        get_webpage_tool,
        weather_tool,
        news_tool,
    ]


def register_all() -> ToolRegistry:
    """Create and return a registry with all search tools."""
    registry = ToolRegistry()
    for tool in get_search_tools():
        registry.register(tool)
    return registry


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Web Search Tools Demo")
    print("=" * 40)

    registry = register_all()

    # Test web search
    print("\n1. Web Search:")
    result = registry.execute("web_search", query="python programming")
    print(json.dumps(result.result, indent=2))

    # Test weather
    print("\n2. Weather:")
    result = registry.execute("get_weather", location="San Francisco")
    print(json.dumps(result.result, indent=2))

    # Test news
    print("\n3. News:")
    result = registry.execute("get_news", topic="AI", num_articles=2)
    print(json.dumps(result.result, indent=2))
