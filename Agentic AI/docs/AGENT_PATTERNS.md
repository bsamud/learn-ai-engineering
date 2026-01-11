# Agent Patterns Reference

Quick reference for common agent patterns and implementations.

## Table of Contents

1. [ReAct Pattern](#react-pattern)
2. [Tool Calling](#tool-calling)
3. [Multi-Agent Patterns](#multi-agent-patterns)
4. [Memory Patterns](#memory-patterns)
5. [Error Handling](#error-handling)
6. [Safety Patterns](#safety-patterns)

---

## ReAct Pattern

### Basic Structure

```
Thought: [Reasoning about what to do]
Action: [tool_name]
Action Input: {"param": "value"}
Observation: [Result from tool]
... (repeat)
Thought: [Ready to answer]
Action: finish
Action Input: {"answer": "Final answer"}
```

### Implementation

```python
REACT_PROMPT = """
You are a helpful assistant. For each step:
1. Think about what to do
2. Take an action using available tools
3. Observe the result

Format:
Thought: [your reasoning]
Action: [tool name]
Action Input: {"param": "value"}

When ready to answer:
Thought: [why you're done]
Action: finish
Action Input: {"answer": "your answer"}
"""

def react_loop(task, tools, max_steps=5):
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": task}
    ]

    for step in range(max_steps):
        response = call_llm(messages)
        thought, action, args = parse_response(response)

        if action == "finish":
            return args["answer"]

        result = execute_tool(action, args)

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "Max steps reached"
```

---

## Tool Calling

### OpenAI Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```

### Anthropic Format

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    tools=tools
)
```

### Tool Result Handling

```python
# OpenAI
if message.tool_calls:
    for tool_call in message.tool_calls:
        result = execute_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

# Anthropic
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)
            }]
        })
```

---

## Multi-Agent Patterns

### Sequential Pipeline

```python
def pipeline(task, agents):
    context = ""
    for agent in agents:
        result = agent.run(task, context)
        context += f"\n{agent.name}: {result}"
    return result
```

### Parallel Execution

```python
import asyncio

async def parallel(task, agents):
    tasks = [agent.run(task) for agent in agents]
    results = await asyncio.gather(*tasks)
    return dict(zip([a.name for a in agents], results))
```

### Hierarchical (Manager/Workers)

```python
class ManagerAgent:
    def __init__(self, workers):
        self.workers = workers

    def run(self, task):
        # Decompose task
        subtasks = self.decompose(task)

        # Assign to workers
        results = {}
        for subtask in subtasks:
            worker = self.select_worker(subtask)
            results[worker.name] = worker.run(subtask)

        # Synthesize results
        return self.synthesize(results)
```

### Debate Pattern

```python
def debate(topic, pro_agent, con_agent, judge, rounds=3):
    history = []

    for round in range(rounds):
        pro_arg = pro_agent.run(f"Argue FOR: {topic}\nHistory: {history}")
        con_arg = con_agent.run(f"Argue AGAINST: {topic}\nHistory: {history}")
        history.extend([pro_arg, con_arg])

    verdict = judge.run(f"Judge this debate:\n{history}")
    return verdict
```

---

## Memory Patterns

### Conversation Memory

```python
class ConversationMemory:
    def __init__(self, max_messages=50):
        self.messages = []
        self.max_messages = max_messages

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self):
        return self.messages
```

### Summary Memory

```python
class SummaryMemory:
    def __init__(self, summarize_every=10):
        self.messages = []
        self.summary = ""
        self.summarize_every = summarize_every

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) >= self.summarize_every:
            self.summarize()

    def summarize(self):
        # Call LLM to summarize
        self.summary = call_llm(
            f"Summarize this conversation:\n{self.messages}"
        )
        self.messages = []

    def get_context(self):
        if self.summary:
            return [{"role": "system", "content": f"Previous context: {self.summary}"}] + self.messages
        return self.messages
```

---

## Error Handling

### Retry with Backoff

```python
import time

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

### Graceful Degradation

```python
def safe_execute(action, fallback="Unable to complete action"):
    try:
        return action()
    except ToolError as e:
        return f"Tool error: {e}. {fallback}"
    except APIError as e:
        return f"API error: {e}. {fallback}"
    except Exception as e:
        return fallback
```

### Self-Correction

```python
def execute_with_correction(task, validator, max_attempts=3):
    messages = [{"role": "user", "content": task}]

    for attempt in range(max_attempts):
        result = call_llm(messages)
        is_valid, feedback = validator(result)

        if is_valid:
            return result

        messages.append({"role": "assistant", "content": result})
        messages.append({"role": "user", "content": f"That's incorrect. {feedback}. Try again."})

    return result
```

---

## Safety Patterns

### Action Allowlist

```python
ALLOWED_ACTIONS = {"read", "search", "calculate"}

def safe_execute(action, args):
    if action not in ALLOWED_ACTIONS:
        return f"Action '{action}' not allowed"
    return execute(action, args)
```

### Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_calls=10, window=60):
        self.max_calls = max_calls
        self.window = window
        self.calls = defaultdict(list)

    def check(self, action):
        now = time.time()
        self.calls[action] = [t for t in self.calls[action] if now - t < self.window]

        if len(self.calls[action]) >= self.max_calls:
            return False

        self.calls[action].append(now)
        return True
```

### Human Approval

```python
REQUIRES_APPROVAL = {"delete", "send", "modify"}

def execute_with_approval(action, args, approve_callback):
    if any(trigger in action.lower() for trigger in REQUIRES_APPROVAL):
        if not approve_callback(action, args):
            return "Action rejected by user"
    return execute(action, args)
```

### Input Validation

```python
def validate_input(action, args, schema):
    # Check required fields
    for field in schema.get("required", []):
        if field not in args:
            return False, f"Missing required field: {field}"

    # Check types
    for field, value in args.items():
        if field in schema["properties"]:
            expected_type = schema["properties"][field]["type"]
            if not isinstance(value, TYPE_MAP[expected_type]):
                return False, f"Invalid type for {field}"

    return True, "Valid"
```

---

## Quick Reference

### When to Use Each Pattern

| Pattern | Use Case |
|---------|----------|
| ReAct | General reasoning + action tasks |
| Sequential Pipeline | Step-by-step processing |
| Parallel | Independent subtasks |
| Hierarchical | Complex tasks needing coordination |
| Debate | Exploring multiple perspectives |
| Self-correction | Quality-critical outputs |
| Human-in-loop | High-stakes actions |

### Common Tool Types

| Tool Type | Examples |
|-----------|----------|
| Information | search, lookup, retrieve |
| Calculation | math, convert, analyze |
| Action | create, update, delete |
| Communication | send, notify, post |
