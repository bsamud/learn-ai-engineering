# AI Engineering Self-Assessment

Use this rubric to evaluate your understanding and skills after completing the track.

---

## Module 1: LLM Integration

### Notebook 01: LLM APIs

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| API Setup | Can't set up API keys | Needs help with configuration | Sets up independently | Troubleshoots others' issues |
| Making Calls | Can't make API calls | Makes basic calls | Handles all parameters | Optimizes for cost/latency |
| Streaming | Never used streaming | Can implement with help | Implements independently | Handles edge cases |
| Error Handling | No error handling | Basic try/catch | Retries with backoff | Production-grade handling |
| Cost Tracking | Unaware of costs | Knows pricing exists | Tracks usage | Optimizes spending |

**Self-Check Questions:**
1. Can you switch between OpenAI and Anthropic APIs?
2. Do you understand token limits and pricing?
3. Can you implement retry logic with exponential backoff?

---

### Notebook 02: Structured Outputs

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| JSON Mode | Never used | Can use with examples | Uses confidently | Knows limitations |
| Pydantic | Never used | Basic models | Complex validation | Custom validators |
| Extraction | Can't extract data | Extracts simple data | Complex entities | Handles edge cases |
| Error Recovery | No handling | Basic validation | Retry on failure | Self-correcting |

**Self-Check Questions:**
1. Can you define a Pydantic model for complex data?
2. How do you handle when LLM returns invalid JSON?
3. Can you extract multiple entity types from text?

---

## Module 2: RAG Systems

### Notebook 03: Embeddings & Vectors

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| Concept | Don't understand | Basic understanding | Can explain clearly | Teaches others |
| Creating | Can't create | Creates with help | Creates independently | Batch processing |
| Similarity | Don't understand | Knows concept | Implements search | Optimizes performance |
| Vector DBs | Never used | Basic CRUD | Filters & metadata | Production setup |

**Self-Check Questions:**
1. Can you explain embeddings to a non-technical person?
2. What's the difference between cosine and euclidean distance?
3. Can you set up a persistent vector database?

---

### Notebook 04: RAG Fundamentals

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| Architecture | Don't understand | Knows components | Builds end-to-end | Designs systems |
| Loading | Can't load docs | Loads basic files | Multiple formats | Custom loaders |
| Chunking | Don't understand | Fixed-size only | Multiple strategies | Optimizes for task |
| Retrieval | Can't implement | Basic search | Configured search | Hybrid approaches |
| Generation | Can't combine | Basic prompting | Context integration | Source citation |

**Self-Check Questions:**
1. Can you build a RAG pipeline from scratch?
2. How do you choose chunk size for different content?
3. Can you add source citations to generated answers?

---

### Notebook 05: Advanced RAG

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| Chunking | Fixed only | Knows options | Implements multiple | Semantic chunking |
| Hybrid Search | Don't know | Understands concept | Implements | Tunes parameters |
| Re-ranking | Never used | Basic implementation | LLM re-ranking | Cross-encoders |
| Query Transform | Don't know | HyDE concept | Implements | Multiple methods |
| Evaluation | No metrics | Knows metrics | Implements | A/B testing |

**Self-Check Questions:**
1. When would you use hybrid search over pure vector?
2. How does HyDE improve retrieval?
3. What metrics would you use to evaluate RAG quality?

---

## Module 3: Fine-tuning

### Notebook 06: Fine-tuning Basics

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| When to Use | Don't know | General idea | Clear decision tree | Advises others |
| Data Prep | Can't prepare | Basic formatting | Quality validation | Augmentation |
| Training | Never done | With guidance | Independent | Hyperparameter tuning |
| Evaluation | No evaluation | Basic metrics | Comprehensive | A/B testing |

**Self-Check Questions:**
1. When should you fine-tune vs. use prompting vs. RAG?
2. How many examples do you need for fine-tuning?
3. How do you evaluate if fine-tuning improved performance?

---

### Notebook 07: LoRA & PEFT

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| PEFT Concepts | Don't understand | Basic idea | Explains clearly | Compares methods |
| LoRA | Never used | Conceptual | Implements | Tunes rank/alpha |
| Training | Can't train | With guidance | Independent | Optimizes memory |
| Inference | Can't use | Basic loading | Merged models | Adapter switching |

**Self-Check Questions:**
1. Why does LoRA use much less memory than full fine-tuning?
2. How do you choose the rank (r) parameter?
3. When would you merge vs. keep adapters separate?

---

## Module 4: Production

### Notebook 08: Evaluation & Testing

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| Test Design | No tests | Basic cases | Comprehensive | Edge cases |
| Automation | Manual only | Basic automation | CI integration | Regression suite |
| LLM-as-Judge | Never used | Basic usage | Custom criteria | Calibration |
| Metrics | Don't know | Basic metrics | Task-specific | Custom metrics |

**Self-Check Questions:**
1. How do you create a representative test dataset?
2. What are the limitations of LLM-as-Judge?
3. How do you detect regression in model quality?

---

### Notebook 09: Production Deployment

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| API Design | No experience | Basic endpoints | RESTful patterns | Streaming APIs |
| Caching | Never used | Basic cache | Semantic cache | Cache invalidation |
| Rate Limiting | Don't know | Understands | Implements | Adaptive limits |
| Monitoring | No monitoring | Basic logging | Metrics dashboard | Alerting |
| Cost Control | Unaware | Tracks costs | Budget limits | Optimization |

**Self-Check Questions:**
1. How do you design an API for streaming LLM responses?
2. What caching strategy works best for LLM outputs?
3. How do you set up cost alerts and budgets?

---

### Notebook 10: Capstone Project

| Skill | Not Yet | Developing | Proficient | Expert |
|-------|---------|------------|------------|--------|
| Architecture | Can't design | Simple systems | Production-ready | Scalable |
| Integration | Can't combine | Basic integration | Seamless | Handles failures |
| Features | Minimal | Core features | Full system | Extensions |
| Documentation | None | Basic README | User guide | API docs |

**Self-Check Questions:**
1. Can you build a complete AI application end-to-end?
2. Does your application handle errors gracefully?
3. Can someone else use your application without help?

---

## Overall Progress Tracker

### Track Completion

- [ ] Module 1: LLM Integration (Notebooks 1-2)
- [ ] Module 2: RAG Systems (Notebooks 3-5)
- [ ] Module 3: Fine-tuning (Notebooks 6-7)
- [ ] Module 4: Production (Notebooks 8-10)

### Proficiency Level

Count your "Proficient" or "Expert" ratings above:

| Score | Level | Next Steps |
|-------|-------|------------|
| 0-10 | Beginner | Review fundamentals, complete more exercises |
| 11-20 | Developing | Practice with real projects |
| 21-30 | Proficient | Build portfolio projects |
| 31+ | Expert | Mentor others, contribute to community |

---

## Project Ideas to Demonstrate Skills

### Beginner Projects
1. FAQ chatbot with RAG
2. Text summarizer API
3. Sentiment analysis tool

### Intermediate Projects
1. Multi-document Q&A system
2. Content moderation pipeline
3. Structured data extractor

### Advanced Projects
1. Multi-modal RAG (text + images)
2. Fine-tuned domain expert
3. Autonomous research assistant

---

## Reflection Questions

After completing the track, reflect on:

1. **What was the most challenging concept?**
2. **What would you do differently next time?**
3. **What project will you build to practice?**
4. **How will you stay current with AI developments?**

---

## Certificate of Completion

Once you've completed all notebooks and achieved "Proficient" in most skills:

```
================================================
        AI ENGINEERING TRACK COMPLETED
================================================

Name: _______________________
Date: _______________________

Completed:
[ ] All 10 notebooks
[ ] Capstone project
[ ] Self-assessment

Next steps:
- Build portfolio projects
- Explore the Agentic AI track
- Contribute to open source

================================================
```

Congratulations on your AI Engineering journey!
