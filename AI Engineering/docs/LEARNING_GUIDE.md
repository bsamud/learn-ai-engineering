# AI Engineering Learning Guide

A comprehensive guide to mastering production AI application development.

## Overview

This track teaches you how to build production-ready AI applications, from LLM integration to deployment. You'll learn industry best practices for RAG systems, fine-tuning, evaluation, and production deployment.

## Prerequisites

Before starting this track, you should have:

- **Python proficiency**: Comfortable with classes, functions, async programming
- **Basic ML knowledge**: Understanding of embeddings, vectors, similarity
- **API experience**: Familiarity with REST APIs and HTTP
- **Command line**: Basic terminal/shell usage

## Learning Path

### Module 1: LLM Integration (Notebooks 1-2)

**Goal**: Master working with LLM APIs

#### Notebook 01: Working with LLM APIs
- Setting up OpenAI and Anthropic
- Making API calls
- Streaming responses
- Error handling and retries
- Cost tracking

**Checkpoint**: You can call multiple LLM providers and handle errors gracefully.

#### Notebook 02: Structured Outputs
- JSON mode
- Pydantic validation
- Entity extraction
- Handling malformed responses

**Checkpoint**: You can reliably extract structured data from LLM outputs.

---

### Module 2: RAG Systems (Notebooks 3-5)

**Goal**: Build retrieval-augmented generation systems

#### Notebook 03: Embeddings & Vectors
- What embeddings are
- Creating embeddings
- Similarity search
- Vector databases (ChromaDB)

**Checkpoint**: You understand semantic search and can use vector databases.

#### Notebook 04: RAG Fundamentals
- RAG architecture
- Document loading and chunking
- Building the retrieval pipeline
- Generating answers with context

**Checkpoint**: You can build a basic RAG system end-to-end.

#### Notebook 05: Advanced RAG
- Chunking strategies
- Hybrid search (BM25 + vectors)
- Re-ranking
- Query transformation
- Evaluation metrics

**Checkpoint**: You can optimize RAG systems for production quality.

---

### Module 3: Fine-tuning (Notebooks 6-7)

**Goal**: Customize models for specific tasks

#### Notebook 06: Fine-tuning Basics
- When to fine-tune
- Data preparation
- OpenAI fine-tuning API
- Evaluation

**Checkpoint**: You know when and how to fine-tune models.

#### Notebook 07: LoRA & PEFT
- Parameter-efficient methods
- LoRA explained
- Training with Hugging Face PEFT
- Inference with adapters

**Checkpoint**: You can fine-tune models on consumer hardware.

---

### Module 4: Production (Notebooks 8-10)

**Goal**: Deploy AI applications to production

#### Notebook 08: Evaluation & Testing
- Building test datasets
- Automated evaluation
- LLM-as-Judge
- Regression testing

**Checkpoint**: You can build comprehensive evaluation frameworks.

#### Notebook 09: Production Deployment
- API design
- Caching strategies
- Rate limiting
- Monitoring and metrics
- Cost management

**Checkpoint**: You understand production deployment patterns.

#### Notebook 10: Capstone App
- Complete Document Q&A system
- All production features integrated
- Extension challenges

**Checkpoint**: You built a production-ready AI application.

---

## Study Tips

### 1. Run Every Cell
Don't just read the code. Run each cell and observe the output. Modify values to see what happens.

### 2. Complete the TODOs
The exercises are designed to reinforce learning. Try them before looking at solutions.

### 3. Build Your Own Project
Apply concepts to a project you care about. This accelerates learning.

### 4. Track Your Costs
LLM APIs cost money. Monitor usage and set up billing alerts.

### 5. Join the Community
Share your projects and learn from others.

---

## Common Issues

### API Key Errors
```
AuthenticationError: Invalid API key
```
- Check your `.env` file exists
- Verify the key is correct
- Ensure no extra spaces or quotes

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
- Make sure you're running from the `notebooks/` directory
- Check that `sys.path.append` runs first

### Rate Limits
```
RateLimitError: Rate limit exceeded
```
- Wait a moment and retry
- Implement exponential backoff
- Consider using caching

### Memory Issues
```
OutOfMemoryError
```
- Reduce batch sizes
- Use smaller models
- Enable quantization (QLoRA)

---

## Resources

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [ChromaDB](https://docs.trychroma.com)

### Papers
- "Retrieval-Augmented Generation" (RAG original paper)
- "LoRA: Low-Rank Adaptation" (efficient fine-tuning)
- "Chain-of-Thought Prompting" (reasoning)

### Tools
- [LangChain](https://python.langchain.com) - LLM framework
- [LlamaIndex](https://docs.llamaindex.ai) - RAG framework
- [Weights & Biases](https://wandb.ai) - Experiment tracking

---

## Self-Assessment Rubric

Use this to gauge your progress:

### Beginner (Notebooks 1-2)
- [ ] Can make LLM API calls
- [ ] Understands tokens and pricing
- [ ] Can extract structured data

### Intermediate (Notebooks 3-5)
- [ ] Understands embeddings and similarity
- [ ] Can build a RAG pipeline
- [ ] Knows when to use different chunking strategies

### Advanced (Notebooks 6-8)
- [ ] Knows when to fine-tune vs prompt
- [ ] Can use LoRA for efficient training
- [ ] Can design evaluation frameworks

### Production-Ready (Notebooks 9-10)
- [ ] Can design production APIs
- [ ] Implements caching and rate limiting
- [ ] Monitors costs and performance
- [ ] Built a complete AI application

---

## Getting Help

If you're stuck:

1. **Re-read the notebook** - Often the answer is there
2. **Check the src/ modules** - They have detailed docstrings
3. **Search the documentation** - Links provided above
4. **Open an issue** - We're here to help

Good luck on your AI Engineering journey!
