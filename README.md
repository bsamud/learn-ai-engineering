# AI Academy

**Open-source learning tracks for AI and Machine Learning Engineering.**

---

## Learning Tracks

| Track | Notebooks | Focus | Prerequisites |
|-------|-----------|-------|---------------|
| [**ML Engineering**](./MLE%20Capstone%20Project/) | 7 | MLOps, Model Deployment | Python, ML basics |
| [**Agentic AI**](./Agentic%20AI/) | 8 | AI Agents, Tool Use | Python basics |
| [**AI Engineering**](./AI%20Engineering/) | 10 | RAG, Fine-tuning, Production | Python basics |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/academy.git
cd academy
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies for your track
pip install -r "Agentic AI/requirements.txt"
# or
pip install -r "AI Engineering/requirements.txt"
# or
pip install -r "ML Engineering/requirements.txt"
```

### 3. Configure API Keys
```bash
# Copy the example env file
cp "Agentic AI/.env.example" "Agentic AI/.env"

# Edit with your API keys
nano "Agentic AI/.env"
```

### 4. Download Datasets (MLE Track Only)
```bash
# See DATA_SOURCES.md for full instructions
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p "ML Engineering/data/raw"
```

---

## Track Overviews

### Agentic AI
Build autonomous AI agents that can reason, use tools, and complete complex tasks.

**Topics**: LLM APIs, Prompt Engineering, Tool Use, Function Calling, ReAct Agents, Multi-Agent Systems, Autonomous Workflows

**No datasets required** - API-based learning

### AI Engineering
Build production-ready AI applications with RAG, fine-tuning, and deployment.

**Topics**: LLM Integration, Structured Outputs, Embeddings, RAG Systems, Fine-tuning, LoRA/PEFT, Evaluation, Production Deployment

**No datasets required** - Creates sample data dynamically

### ML Engineering
End-to-end machine learning engineering with MLOps best practices.

**Topics**: Classification, Regression, Feature Engineering, Model Deployment, Monitoring

**Requires Kaggle datasets** - See [DATA_SOURCES.md](./DATA_SOURCES.md)

---

## Required Datasets

| Dataset | Size | Track | Kaggle Link |
|---------|------|-------|-------------|
| Credit Card Fraud | 150 MB | MLE, Workshops | [Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Bank Churn | 1 MB | MLE | [Link](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) |
| Credit Risk | 5 MB | MLE | [Link](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |

See [DATA_SOURCES.md](./DATA_SOURCES.md) for complete download instructions.

---

## API Keys Required

| Provider | Tracks | Get Key |
|----------|--------|---------|
| OpenAI | Agentic AI, AI Engineering | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | Agentic AI, AI Engineering | [console.anthropic.com](https://console.anthropic.com/) |
| Kaggle | ML Engineering | [kaggle.com/settings](https://www.kaggle.com/settings) |

---

## Directory Structure

```
academy/
├── README.md                    # This file
├── DATA_SOURCES.md             # Dataset download instructions
├── Agentic AI/
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   ├── notebooks/              # 8 learning notebooks
│   ├── src/                    # Utility modules
│   ├── tools/                  # Example tools
│   └── docs/                   # Guides and references
├── AI Engineering/
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   ├── notebooks/              # 10 learning notebooks
│   ├── src/                    # Utility modules
│   └── docs/                   # Guides and references
└── ML Engineering/
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── notebooks/              # 7 learning notebooks
    ├── src/                    # Utility modules
    └── data/
        └── raw/                # Place datasets here
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

Created for the open-source community to help everyone learn AI and ML engineering.
