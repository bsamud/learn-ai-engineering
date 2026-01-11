# AI Engineering Cheatsheet

Quick reference for AI Engineering patterns and code snippets.

---

## LLM API Calls

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

# Basic chat
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# With JSON mode
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Anthropic

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

---

## Structured Outputs

### Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    occupation: Optional[str] = None

# Validate LLM output
data = json.loads(llm_response)
person = Person(**data)  # Raises ValidationError if invalid
```

### Extraction Pattern

```python
def extract_data(text: str, model_class: type[BaseModel]) -> BaseModel:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Extract: {model_class.model_json_schema()}"},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    return model_class(**json.loads(response.choices[0].message.content))
```

---

## Embeddings

### Create Embeddings

```python
from openai import OpenAI

client = OpenAI()

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)
embedding = response.data[0].embedding  # List of floats

# Batch
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Text 1", "Text 2", "Text 3"]
)
embeddings = [d.embedding for d in response.data]
```

### Cosine Similarity

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## Vector Databases

### ChromaDB

```python
import chromadb

# In-memory
client = chromadb.Client()

# Persistent
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"],
    metadatas=[{"source": "a"}, {"source": "b"}]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    where={"source": "a"}  # Optional filter
)
```

---

## RAG Pipeline

### Basic Pattern

```python
# 1. Load documents
from src.rag_pipeline import DocumentLoader, Chunker

loader = DocumentLoader()
docs = loader.load_directory("./documents")

# 2. Chunk
chunker = Chunker(chunk_size=500, overlap=50)
chunks = chunker.chunk_all(docs)

# 3. Index
from src.rag_pipeline import VectorStore

store = VectorStore(collection_name="my_docs")
store.add_documents(chunks)

# 4. Query
results = store.search("my question", k=5)

# 5. Generate
context = "\n\n".join([r["content"] for r in results])
prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
answer = llm.chat(prompt)
```

### Chunking Strategies

```python
# Fixed size
def chunk_fixed(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

# By separator
def chunk_by_separator(text, separator="\n\n"):
    return [s.strip() for s in text.split(separator) if s.strip()]
```

---

## Fine-tuning

### OpenAI Format (JSONL)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Create Fine-tuning Job

```python
from openai import OpenAI

client = OpenAI()

# Upload file
file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")

# Create job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)

# Check status
status = client.fine_tuning.jobs.retrieve(job.id)
print(status.status, status.fine_tuned_model)
```

### LoRA with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

peft_model = get_peft_model(base_model, config)
peft_model.print_trainable_parameters()
```

---

## Evaluation

### Basic Metrics

```python
# Precision, Recall, F1
def precision_recall_f1(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)

    precision = tp / len(retrieved_set) if retrieved_set else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1
```

### LLM-as-Judge

```python
def evaluate_answer(question, answer, criteria="accuracy, completeness"):
    prompt = f"""Evaluate this answer (0-1 score):
Question: {question}
Answer: {answer}
Criteria: {criteria}
Return JSON: {{"score": float, "feedback": str}}"""

    response = llm.chat(prompt)
    return json.loads(response)
```

---

## Production Patterns

### Caching

```python
import hashlib
from functools import lru_cache

# Simple hash-based cache
cache = {}

def cached_llm_call(prompt):
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if key not in cache:
        cache[key] = llm.chat(prompt)
    return cache[key]
```

### Rate Limiting

```python
import time

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.limit = calls_per_minute
        self.calls = []

    def acquire(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.limit:
            return False
        self.calls.append(now)
        return True
```

### Retry with Backoff

```python
import time
from openai import RateLimitError

def retry_with_backoff(fn, max_retries=3):
    for i in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            wait = 2 ** i
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

---

## Cost Estimation

### Token Pricing (approximate)

| Model | Input (1M tokens) | Output (1M tokens) |
|-------|-------------------|-------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3.5-sonnet | $3.00 | $15.00 |
| text-embedding-3-small | $0.02 | - |

### Estimate Tokens

```python
# Rough estimate: 1 token ≈ 4 characters
def estimate_tokens(text):
    return len(text) / 4

# Accurate (requires tiktoken)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = len(enc.encode(text))
```

---

## Quick Reference

### Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| Simple tasks | gpt-4o-mini |
| Complex reasoning | gpt-4o, claude-3.5-sonnet |
| Embeddings | text-embedding-3-small |
| Fine-tuning | gpt-4o-mini-2024-07-18 |
| Local/LoRA | Llama, Mistral |

### Chunk Sizes

| Content Type | Chunk Size | Overlap |
|--------------|------------|---------|
| General docs | 500 | 50 |
| Code | 1000 | 100 |
| FAQs | 200 | 20 |
| Long-form | 1500 | 200 |

### When to Use What

| Need | Solution |
|------|----------|
| General knowledge | Prompt engineering |
| Your documents | RAG |
| Consistent format | Fine-tuning |
| Specific behavior | Fine-tuning + RAG |
| Real-time data | Tool use / Functions |
