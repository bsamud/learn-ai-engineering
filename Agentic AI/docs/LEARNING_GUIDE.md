# Agentic AI Learning Guide

A comprehensive guide to completing the Agentic AI track.

## Learning Objectives

By completing this track, you will be able to:

1. **Understand LLMs**: Know how large language models work and how to interact with them
2. **Engineer Prompts**: Write effective prompts that get consistent, high-quality results
3. **Build Tools**: Create tools that extend LLM capabilities
4. **Implement Agents**: Build ReAct agents that reason and act
5. **Orchestrate Systems**: Coordinate multiple agents for complex tasks
6. **Design Workflows**: Create autonomous systems with safety guardrails
7. **Apply Knowledge**: Build your own agent for a real problem

## Prerequisites

- **Python**: Comfortable with Python 3.8+
- **APIs**: Basic understanding of REST APIs
- **Command Line**: Can run commands in terminal
- **Git**: Basic version control (optional but helpful)

## Time Estimates

| Module | Notebooks | Estimated Time |
|--------|-----------|----------------|
| Foundations | 01-02 | 4-5 hours |
| Tool Use | 03-04 | 4-6 hours |
| Agent Patterns | 05-06 | 6-8 hours |
| Autonomous | 07-08 | 6-8 hours |
| **Total** | 8 notebooks | **20-27 hours** |

## Module Breakdown

### Module 1: Foundations (Notebooks 01-02)

**Notebook 01: LLM Basics**
- Setting up API keys
- Making API calls
- Understanding tokens and context windows
- Experimenting with parameters (temperature, max_tokens)

**Notebook 02: Prompt Engineering**
- Zero-shot vs few-shot prompting
- System prompts and roles
- Chain-of-thought reasoning
- Creating prompt templates

**Checkpoint**: Before moving on, ensure you can:
- [ ] Make successful API calls to OpenAI or Anthropic
- [ ] Explain what tokens are and why they matter
- [ ] Write effective few-shot prompts
- [ ] Use chain-of-thought for reasoning tasks

### Module 2: Tool Use (Notebooks 03-04)

**Notebook 03: Tool Use Fundamentals**
- Understanding tool/function calling
- JSON Schema for tool definitions
- Writing good tool descriptions
- Using the ToolRegistry

**Notebook 04: Function Calling**
- OpenAI function calling format
- Anthropic tool use format
- Multi-turn conversations
- Error handling

**Checkpoint**: Before moving on, ensure you can:
- [ ] Define tools with proper JSON Schema
- [ ] Handle LLM tool call requests
- [ ] Manage multi-turn tool conversations
- [ ] Implement error handling for tools

### Module 3: Agent Patterns (Notebooks 05-06)

**Notebook 05: ReAct Agents**
- The ReAct pattern explained
- Thought → Action → Observation loop
- Building a ReAct agent from scratch
- Using the framework's ReActAgent

**Notebook 06: Multi-Agent Systems**
- Agent specialization
- Sequential and parallel execution
- Building an orchestrator
- Managing shared state

**Checkpoint**: Before moving on, ensure you can:
- [ ] Explain the ReAct pattern
- [ ] Implement a basic ReAct loop
- [ ] Create specialized agents
- [ ] Orchestrate multiple agents

### Module 4: Autonomous Systems (Notebooks 07-08)

**Notebook 07: Autonomous Workflows**
- Task planning and decomposition
- Self-correction patterns
- Human-in-the-loop controls
- Safety guardrails

**Notebook 08: Capstone Agent**
- Choose your project
- Design and implement
- Test and iterate
- Document your work

## Best Practices

### Coding

1. **Read Before Running**: Understand the code before executing
2. **Experiment**: Modify examples to see what happens
3. **Complete TODOs**: Don't skip the exercises
4. **Take Notes**: Document your learnings

### Learning

1. **Sequential Order**: Complete notebooks in order
2. **Checkpoints**: Verify understanding before moving on
3. **Breaks**: Take breaks between modules
4. **Practice**: Build small projects between notebooks

## Common Mistakes

### API Keys
- ❌ Hardcoding API keys in code
- ✅ Using .env files and environment variables

### Prompts
- ❌ Vague, open-ended prompts
- ✅ Specific prompts with clear constraints

### Tools
- ❌ Poor tool descriptions
- ✅ Clear descriptions that help LLM choose correctly

### Agents
- ❌ No step limits (infinite loops)
- ✅ Always set max_steps

### Error Handling
- ❌ Ignoring errors
- ✅ Proper try/except with meaningful messages

## Troubleshooting

### "API Key not found"
- Check your .env file exists
- Verify the key name matches exactly
- Restart Jupyter after changing .env

### "Rate limit exceeded"
- Add delays between API calls
- Use a smaller model for testing
- Check your API plan limits

### "Module not found"
- Run `pip install -r requirements.txt`
- Check sys.path includes parent directory
- Restart Jupyter kernel

### Agent gets stuck in loop
- Check your finish condition
- Verify tool results are being processed
- Add debugging print statements

## Evaluation Criteria

### Technical Skills (50%)
- Correct implementation
- Proper error handling
- Clean, readable code

### Understanding (30%)
- Can explain concepts
- Makes appropriate choices
- Adapts to new problems

### Documentation (20%)
- Clear explanations
- Good comments
- Complete capstone docs

## Resources

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com)
- [JSON Schema](https://json-schema.org/)

### Papers
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [Toolformer](https://arxiv.org/abs/2302.04761)

### Tools & Frameworks
- [LangChain](https://python.langchain.com/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [CrewAI](https://github.com/joaomdmoura/crewAI)

## Getting Help

1. **Check the docs**: Review AGENT_PATTERNS.md
2. **Re-read the notebook**: Often the answer is there
3. **Debug step by step**: Add print statements
4. **Search online**: Stack Overflow, GitHub issues
5. **Community**: Join AI/ML Discord servers

## Completion Checklist

- [ ] Completed all 8 notebooks
- [ ] Finished all exercises
- [ ] Built capstone agent
- [ ] Documented capstone project
- [ ] Self-evaluated using rubric

Congratulations on completing the Agentic AI track! 🎉
