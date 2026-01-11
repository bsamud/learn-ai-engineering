# Agentic AI Track

Build autonomous AI agents that can reason, use tools, and complete complex tasks.

## Overview

This self-paced learning track teaches you how to build AI agents from the ground up. Starting with LLM fundamentals, you'll progressively learn tool use, reasoning patterns, multi-agent systems, and autonomous workflows.

**Skill Level:** Beginner-friendly (Python knowledge required)
**Duration:** 20-30 hours self-paced
**Prerequisites:** Basic Python, familiarity with APIs

## Learning Path

```
Module 1: Foundations          Module 2: Tool Use
┌─────────────────────┐       ┌─────────────────────┐
│ 01. LLM Basics      │──────▶│ 03. Tool Fundamentals│
│ 02. Prompt Eng.     │       │ 04. Function Calling │
└─────────────────────┘       └─────────────────────┘
                                        │
                    ┌───────────────────┘
                    ▼
Module 3: Agent Patterns       Module 4: Autonomous Systems
┌─────────────────────┐       ┌─────────────────────┐
│ 05. ReAct Agents    │──────▶│ 07. Auto Workflows  │
│ 06. Multi-Agent     │       │ 08. Capstone Agent  │
└─────────────────────┘       └─────────────────────┘
```

## Notebooks

| # | Notebook | Topics | Duration |
|---|----------|--------|----------|
| 01 | LLM Basics | API setup, tokens, parameters | 2 hrs |
| 02 | Prompt Engineering | Zero/few-shot, CoT, templates | 2-3 hrs |
| 03 | Tool Use Fundamentals | Tool schemas, definitions | 2-3 hrs |
| 04 | Function Calling | OpenAI/Anthropic formats, multi-turn | 2-3 hrs |
| 05 | ReAct Agents | Reasoning + Acting pattern | 3-4 hrs |
| 06 | Multi-Agent Systems | Orchestration, collaboration | 3-4 hrs |
| 07 | Autonomous Workflows | Planning, self-correction | 3-4 hrs |
| 08 | Capstone Agent | Build your own agent | 4-5 hrs |

## Project Structure

```
Agentic AI/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── notebooks/                   # Learning notebooks
│   ├── 01_llm_basics.ipynb
│   ├── 02_prompt_engineering.ipynb
│   ├── 03_tool_use_fundamentals.ipynb
│   ├── 04_function_calling.ipynb
│   ├── 05_react_agents.ipynb
│   ├── 06_multi_agent_systems.ipynb
│   ├── 07_autonomous_workflows.ipynb
│   └── 08_capstone_agent.ipynb
│
├── src/                         # Utility modules
│   ├── llm_client.py           # LLM provider abstraction
│   ├── tool_registry.py        # Tool management
│   ├── agent_framework.py      # Agent implementations
│   └── utils.py                # Helper functions
│
├── tools/                       # Example tool implementations
│   ├── calculator.py
│   ├── web_search.py
│   ├── file_operations.py
│   └── api_tools.py
│
├── data/examples/               # Example data files
│
├── docs/                        # Documentation
│   ├── LEARNING_GUIDE.md
│   └── AGENT_PATTERNS.md
│
└── evaluation/
    └── self_assessment.md       # Self-evaluation rubric
```

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your API keys:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Start Learning

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

Open `01_llm_basics.ipynb` and begin!

## Learning Approach

Each notebook follows a consistent structure:

1. **Concept Introduction** - Theory and context
2. **Guided Examples** - Pre-implemented code to run and study
3. **TODO Exercises** - Practice implementing yourself
4. **Checkpoints** - Verify understanding before proceeding
5. **Challenges** - Optional advanced exercises

## What You'll Build

By the end of this track, you'll have built:

- A multi-provider LLM client
- Custom tools (calculator, web search, file ops)
- A ReAct agent that reasons through problems
- A multi-agent system with specialized roles
- An autonomous workflow for complex tasks
- Your own capstone agent project

## Key Concepts Covered

### Tool Use
- JSON Schema for tool definitions
- Function calling across providers
- Tool result handling
- Error recovery

### Agent Patterns
- ReAct (Reasoning + Acting)
- Plan-and-Execute
- Reflection and self-correction
- Memory and state management

### Multi-Agent Systems
- Agent roles and specialization
- Sequential vs parallel execution
- Hierarchical orchestration
- Inter-agent communication

### Safety & Guardrails
- Input validation
- Output filtering
- Human-in-the-loop patterns
- Rate limiting and cost control

## Resources

- [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Docs](https://docs.anthropic.com/claude/docs/tool-use)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

## Support

- Check `docs/LEARNING_GUIDE.md` for detailed instructions
- Review `docs/AGENT_PATTERNS.md` for pattern references
- Use `evaluation/self_assessment.md` to track your progress

## License

Open source for educational purposes. Part of the AI Academy initiative.
