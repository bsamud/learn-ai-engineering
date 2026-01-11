# AI Engineering Track

Build production-ready AI applications from LLM integration to deployment.

## Overview

This self-paced learning track teaches you how to build real-world AI applications. From API integration and RAG systems to fine-tuning and production deployment.

**Skill Level:** Beginner-friendly (Python knowledge required)
**Duration:** 30-40 hours self-paced
**Prerequisites:** Basic Python, familiarity with APIs

## Learning Path

```
Module 1: LLM Integration        Module 2: RAG Systems
┌─────────────────────┐         ┌─────────────────────┐
│ 01. LLM APIs        │────────▶│ 03. Embeddings      │
│ 02. Structured Out  │         │ 04. Basic RAG       │
└─────────────────────┘         │ 05. Advanced RAG    │
                                └─────────────────────┘
                                          │
                    ┌─────────────────────┘
                    ▼
Module 3: Fine-tuning            Module 4: Production
┌─────────────────────┐         ┌─────────────────────┐
│ 06. FT Basics       │────────▶│ 08. Evaluation      │
│ 07. LoRA/PEFT       │         │ 09. Deployment      │
└─────────────────────┘         │ 10. Capstone App    │
                                └─────────────────────┘
```

## Notebooks

| # | Notebook | Topics | Duration |
|---|----------|--------|----------|
| 01 | LLM APIs | Multi-provider, streaming, cost | 2-3 hrs |
| 02 | Structured Outputs | JSON mode, Pydantic, Instructor | 2-3 hrs |
| 03 | Embeddings & Vectors | Embedding models, similarity, vector DBs | 3-4 hrs |
| 04 | RAG Fundamentals | Document loading, chunking, retrieval | 3-4 hrs |
| 05 | Advanced RAG | Hybrid search, re-ranking, evaluation | 3-4 hrs |
| 06 | Fine-tuning Basics | When to fine-tune, OpenAI API | 3-4 hrs |
| 07 | LoRA & PEFT | Efficient fine-tuning, HuggingFace | 4-5 hrs |
| 08 | Evaluation & Testing | Test suites, LLM-as-judge | 3-4 hrs |
| 09 | Production Deployment | FastAPI, caching, monitoring | 3-4 hrs |
| 10 | Capstone Application | Build your AI app | 5-6 hrs |

## Project Structure

```
AI Engineering/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── notebooks/                   # Learning notebooks
│   ├── 01_llm_apis.ipynb
│   ├── 02_structured_outputs.ipynb
│   ├── 03_embeddings_vectors.ipynb
│   ├── 04_rag_fundamentals.ipynb
│   ├── 05_advanced_rag.ipynb
│   ├── 06_finetuning_basics.ipynb
│   ├── 07_lora_peft.ipynb
│   ├── 08_evaluation_testing.ipynb
│   ├── 09_production_deployment.ipynb
│   └── 10_capstone_app.ipynb
│
├── src/                         # Utility modules
│   ├── llm_utils.py            # LLM client helpers
│   ├── embedding_utils.py      # Embedding utilities
│   ├── rag_pipeline.py         # RAG components
│   ├── finetuning_utils.py     # Fine-tuning helpers
│   └── evaluation.py           # Evaluation framework
│
├── data/
│   ├── documents/              # Sample documents for RAG
│   └── training_data/          # Sample training data
│
├── docs/                        # Documentation
│   ├── LEARNING_GUIDE.md
│   └── AI_ENG_CHEATSHEET.md
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
HUGGINGFACE_TOKEN=hf_your-token-here
```

### 3. Start Learning

```bash
jupyter notebook notebooks/
```

## What You'll Build

By the end of this track, you'll have built:

- Multi-provider LLM client with cost tracking
- Structured output extraction pipeline
- Complete RAG system with vector database
- Fine-tuned model for a custom task
- Production-ready API with FastAPI
- Your own AI application!

## Key Technologies

| Category | Technologies |
|----------|-------------|
| LLM APIs | OpenAI, Anthropic, HuggingFace |
| Embeddings | OpenAI, sentence-transformers |
| Vector DBs | ChromaDB, FAISS |
| Fine-tuning | PEFT, LoRA, Transformers |
| Evaluation | RAGAS, custom metrics |
| Deployment | FastAPI, Docker |

## Resources

- [OpenAI Cookbook](https://cookbook.openai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Support

- Check `docs/LEARNING_GUIDE.md` for detailed instructions
- Review `docs/AI_ENG_CHEATSHEET.md` for quick reference
- Use `evaluation/self_assessment.md` to track your progress

## License

Open source for educational purposes. Part of the AI Academy initiative.
