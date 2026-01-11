"""
Agent Framework - Agent Implementations
=======================================

Provides base classes and implementations for AI agents including:
- Agent: Base class for all agents
- ReActAgent: Reasoning + Acting agent
- AgentOrchestrator: Multi-agent coordination
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from .llm_client import LLMClient, Message, LLMResponse
from .tool_registry import Tool, ToolRegistry, ToolResult


# =============================================================================
# Data Classes
# =============================================================================


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentStep:
    """
    A single step in agent execution.

    Attributes
    ----------
    step_num : int
        Step number (1-indexed)
    thought : str
        Agent's reasoning for this step
    action : str
        Action taken (tool name or 'finish')
    action_input : dict
        Input to the action
    observation : str
        Result of the action
    """

    step_num: int
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""

    def to_string(self) -> str:
        """Format step as string."""
        parts = [f"Step {self.step_num}:"]
        if self.thought:
            parts.append(f"  Thought: {self.thought}")
        if self.action:
            parts.append(f"  Action: {self.action}")
            if self.action_input:
                parts.append(f"  Input: {json.dumps(self.action_input)}")
        if self.observation:
            parts.append(f"  Observation: {self.observation}")
        return "\n".join(parts)


@dataclass
class AgentResult:
    """
    Result from agent execution.

    Attributes
    ----------
    success : bool
        Whether the agent completed successfully
    output : str
        Final output/answer from the agent
    steps : list[AgentStep]
        All steps taken during execution
    error : str
        Error message if failed
    metadata : dict
        Additional metadata (tokens, time, etc.)
    """

    success: bool
    output: str
    steps: list[AgentStep] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Get execution summary."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"[{status}] {len(self.steps)} steps | Output: {self.output[:100]}..."


# =============================================================================
# Base Agent
# =============================================================================


class Agent(ABC):
    """
    Abstract base class for agents.

    All agent implementations should inherit from this class and
    implement the `run` method.

    Parameters
    ----------
    name : str
        Agent name/identifier
    llm : LLMClient
        LLM client for reasoning
    tools : ToolRegistry
        Available tools
    system_prompt : str
        System prompt for the agent
    max_steps : int
        Maximum steps before stopping
    verbose : bool
        Whether to print execution details
    """

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_steps = max_steps
        self.verbose = verbose

        self.state = AgentState.IDLE
        self.memory: list[Message] = []
        self.steps: list[AgentStep] = []

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type."""
        pass

    @abstractmethod
    def run(self, task: str) -> AgentResult:
        """
        Execute the agent on a task.

        Parameters
        ----------
        task : str
            The task/query for the agent to handle

        Returns
        -------
        AgentResult
            The result of agent execution
        """
        pass

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.state = AgentState.IDLE
        self.memory = []
        self.steps = []

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(f"[{self.name}] {message}")


# =============================================================================
# ReAct Agent
# =============================================================================


REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves problems step by step.

For each step, you will:
1. THINK about what to do next
2. Take an ACTION using available tools
3. OBSERVE the result

When you have enough information to answer, use the 'finish' action with your final answer.

Available tools:
{tools}

Respond in this exact format for each step:

Thought: [Your reasoning about what to do]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

Or when ready to give final answer:

Thought: [Your reasoning about why you're done]
Action: finish
Action Input: {{"answer": "Your final answer here"}}

Important:
- Always think before acting
- Use tools when you need information
- Give a clear, direct final answer
"""


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) Agent.

    Implements the ReAct pattern where the agent alternates between
    thinking about what to do and taking actions.

    Example
    -------
    >>> llm = LLMClient(provider="openai")
    >>> tools = ToolRegistry()
    >>> tools.register(calculator_tool)
    >>> agent = ReActAgent("math_agent", llm, tools)
    >>> result = agent.run("What is 25 * 4?")
    >>> print(result.output)
    """

    def _default_system_prompt(self) -> str:
        """Return ReAct system prompt."""
        return REACT_SYSTEM_PROMPT

    def run(self, task: str) -> AgentResult:
        """Execute the agent on a task."""
        self.reset()
        self.state = AgentState.THINKING

        # Build system prompt with tools
        tools_desc = self._format_tools()
        system = self.system_prompt.format(tools=tools_desc)

        # Initialize conversation
        self.memory = [
            Message(role="system", content=system),
            Message(role="user", content=task),
        ]

        self._log(f"Starting task: {task}")

        # Main loop
        for step_num in range(1, self.max_steps + 1):
            self._log(f"\n--- Step {step_num} ---")

            step = AgentStep(step_num=step_num)

            try:
                # Get LLM response
                self.state = AgentState.THINKING
                response = self.llm.chat(self.memory)

                # Parse the response
                thought, action, action_input = self._parse_response(response.content)

                step.thought = thought
                step.action = action
                step.action_input = action_input

                self._log(f"Thought: {thought}")
                self._log(f"Action: {action}")

                # Check for finish
                if action.lower() == "finish":
                    answer = action_input.get("answer", str(action_input))
                    step.observation = "Task completed"
                    self.steps.append(step)
                    self.state = AgentState.COMPLETED

                    self._log(f"Final Answer: {answer}")

                    return AgentResult(
                        success=True,
                        output=answer,
                        steps=self.steps,
                        metadata=self.llm.get_usage_stats(),
                    )

                # Execute the action
                self.state = AgentState.ACTING
                result = self.tools.execute(action, **action_input)

                observation = result.to_string()
                step.observation = observation

                self._log(f"Observation: {observation[:200]}...")

                # Add to memory
                self.memory.append(
                    Message(role="assistant", content=response.content)
                )
                self.memory.append(
                    Message(role="user", content=f"Observation: {observation}")
                )

            except Exception as e:
                step.observation = f"Error: {str(e)}"
                self._log(f"Error: {e}")
                self.state = AgentState.ERROR

            self.steps.append(step)

        # Max steps reached
        self.state = AgentState.COMPLETED
        return AgentResult(
            success=False,
            output="Max steps reached without completing task",
            steps=self.steps,
            error="Max steps exceeded",
            metadata=self.llm.get_usage_stats(),
        )

    def _format_tools(self) -> str:
        """Format tools for system prompt."""
        if not self.tools.get_all():
            return "No tools available."

        lines = []
        for tool in self.tools.get_all():
            params = []
            for pname, pinfo in tool.parameters.get("properties", {}).items():
                ptype = pinfo.get("type", "any")
                params.append(f"{pname}: {ptype}")
            params_str = ", ".join(params) if params else "none"
            lines.append(f"- {tool.name}({params_str}): {tool.description}")

        return "\n".join(lines)

    def _parse_response(self, content: str) -> tuple[str, str, dict]:
        """Parse LLM response into thought, action, and input."""
        thought = ""
        action = ""
        action_input = {}

        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.lower().startswith("thought:"):
                thought = line[8:].strip()

            elif line.lower().startswith("action:"):
                action = line[7:].strip()

            elif line.lower().startswith("action input:"):
                input_str = line[13:].strip()
                # Try to parse as JSON
                try:
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # Check remaining lines for JSON
                    remaining = "\n".join(lines[lines.index(line) + 1:])
                    try:
                        # Find JSON in remaining text
                        start = remaining.find("{")
                        end = remaining.rfind("}") + 1
                        if start >= 0 and end > start:
                            action_input = json.loads(remaining[start:end])
                    except:
                        action_input = {"raw": input_str}

        return thought, action, action_input


# =============================================================================
# Agent Orchestrator
# =============================================================================


@dataclass
class AgentTask:
    """A task assigned to an agent."""

    agent_name: str
    task: str
    dependencies: list[str] = field(default_factory=list)
    result: Optional[AgentResult] = None


class AgentOrchestrator:
    """
    Orchestrates multiple agents working together.

    Supports sequential and parallel agent execution with
    dependency management.

    Example
    -------
    >>> orchestrator = AgentOrchestrator()
    >>> orchestrator.register_agent("researcher", research_agent)
    >>> orchestrator.register_agent("writer", writer_agent)
    >>> results = orchestrator.run_sequential([
    ...     AgentTask("researcher", "Find info about topic"),
    ...     AgentTask("writer", "Write article based on research")
    ... ])
    """

    def __init__(self, verbose: bool = True):
        self._agents: dict[str, Agent] = {}
        self.verbose = verbose
        self.execution_log: list[dict] = []

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[name] = agent
        agent.name = name  # Ensure agent has correct name
        self._log(f"Registered agent: {name}")

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agents."""
        return list(self._agents.keys())

    def run_sequential(self, tasks: list[AgentTask]) -> dict[str, AgentResult]:
        """
        Run tasks sequentially.

        Each task can access results from previous tasks.

        Parameters
        ----------
        tasks : list[AgentTask]
            Tasks to execute in order

        Returns
        -------
        dict[str, AgentResult]
            Results keyed by agent name
        """
        results = {}
        context = {}

        for task in tasks:
            agent = self._agents.get(task.agent_name)
            if not agent:
                self._log(f"Agent not found: {task.agent_name}")
                continue

            # Build task with context from dependencies
            full_task = task.task
            if task.dependencies:
                dep_context = []
                for dep in task.dependencies:
                    if dep in results:
                        dep_context.append(f"[{dep}]: {results[dep].output}")
                if dep_context:
                    full_task = f"{task.task}\n\nContext:\n" + "\n".join(dep_context)

            self._log(f"\n{'='*50}")
            self._log(f"Running: {task.agent_name}")
            self._log(f"Task: {task.task[:100]}...")

            # Run agent
            result = agent.run(full_task)
            results[task.agent_name] = result
            task.result = result

            self.execution_log.append({
                "agent": task.agent_name,
                "task": task.task,
                "success": result.success,
                "output": result.output[:200],
            })

            self._log(f"Result: {result.summary()}")

        return results

    def run_parallel(self, tasks: list[AgentTask]) -> dict[str, AgentResult]:
        """
        Run tasks in parallel.

        Note: Currently runs sequentially but structured for future
        async implementation.

        Parameters
        ----------
        tasks : list[AgentTask]
            Tasks to execute (should have no dependencies)

        Returns
        -------
        dict[str, AgentResult]
            Results keyed by agent name
        """
        # TODO: Implement actual parallel execution with asyncio
        # For now, run sequentially
        self._log("Running tasks (sequential, parallel coming soon)...")
        return self.run_sequential(tasks)

    def run_hierarchical(
        self,
        manager_agent: str,
        worker_agents: list[str],
        task: str,
    ) -> AgentResult:
        """
        Run with a manager agent coordinating workers.

        The manager agent decides which workers to invoke and
        synthesizes their results.

        Parameters
        ----------
        manager_agent : str
            Name of the coordinating agent
        worker_agents : list[str]
            Names of worker agents
        task : str
            The main task to accomplish

        Returns
        -------
        AgentResult
            Result from the manager agent
        """
        manager = self._agents.get(manager_agent)
        if not manager:
            return AgentResult(
                success=False,
                output="",
                error=f"Manager agent not found: {manager_agent}",
            )

        # Add worker info to manager's context
        worker_info = []
        for name in worker_agents:
            agent = self._agents.get(name)
            if agent:
                worker_info.append(f"- {name}: {agent.system_prompt[:100]}...")

        enhanced_task = f"""{task}

You have access to these specialized workers:
{chr(10).join(worker_info)}

Coordinate their work to complete the task."""

        self._log(f"Manager {manager_agent} coordinating {len(worker_agents)} workers")

        return manager.run(enhanced_task)

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(f"[Orchestrator] {message}")

    def get_execution_summary(self) -> str:
        """Get a summary of all executions."""
        if not self.execution_log:
            return "No executions yet."

        lines = ["Execution Summary:", "=" * 40]
        for i, entry in enumerate(self.execution_log, 1):
            status = "OK" if entry["success"] else "FAIL"
            lines.append(f"{i}. [{status}] {entry['agent']}: {entry['task'][:50]}...")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_react_agent(
    name: str,
    provider: str = "openai",
    model: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    max_steps: int = 10,
    verbose: bool = True,
) -> ReActAgent:
    """
    Create a ReAct agent with common defaults.

    Parameters
    ----------
    name : str
        Agent name
    provider : str
        LLM provider ('openai' or 'anthropic')
    model : str, optional
        Model to use
    tools : list[Tool], optional
        Tools to register
    max_steps : int
        Maximum steps
    verbose : bool
        Print execution details

    Returns
    -------
    ReActAgent
        Configured ReAct agent
    """
    llm = LLMClient(provider=provider, model=model)

    registry = ToolRegistry()
    if tools:
        for tool in tools:
            registry.register(tool)

    return ReActAgent(
        name=name,
        llm=llm,
        tools=registry,
        max_steps=max_steps,
        verbose=verbose,
    )
