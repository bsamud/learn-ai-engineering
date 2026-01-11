#!/usr/bin/env python3
"""
Generate realistic commit history for AI Academy
Dates: September 20 - October 31, 2025
"""

import json
import random
from datetime import datetime, timedelta

# Seed for reproducibility (can change for different results)
random.seed(42)

def random_time(date_str):
    """Generate random time for a given date."""
    hour = random.randint(8, 22)  # 8 AM to 10 PM
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"

def get_dates_in_range(start_date, end_date):
    """Get list of dates between start and end."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

# Date ranges for each phase
PHASE1_START = datetime(2025, 9, 20)
PHASE1_END = datetime(2025, 9, 22)
PHASE2_START = datetime(2025, 9, 23)
PHASE2_END = datetime(2025, 10, 10)
PHASE3_START = datetime(2025, 10, 11)
PHASE3_END = datetime(2025, 10, 28)
PHASE4_START = datetime(2025, 10, 29)
PHASE4_END = datetime(2025, 10, 31)

commits = []
commit_id = 1

# =============================================================================
# PHASE 1: Initial Setup (Sep 20-22)
# =============================================================================
phase1_dates = get_dates_in_range(PHASE1_START, PHASE1_END)

commits.append({
    "id": commit_id,
    "date": random_time(phase1_dates[0]),
    "message": "Initial commit: AI Academy project structure",
    "files": [".gitignore", "LICENSE"],
    "phase": 1
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase1_dates[1]),
    "message": "Add ML Engineering notebooks and resources",
    "files": ["ML Engineering/"],
    "phase": 1
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase1_dates[2]),
    "message": "Add workshop notebooks for fraud detection and SageMaker",
    "files": [
        "fraud_detection_xgboost_1hour.ipynb",
        "sagemaker_ml_workshop.ipynb",
        "sagemaker_xgboost_fintech_workshop.ipynb"
    ],
    "phase": 1
})
commit_id += 1

# =============================================================================
# PHASE 2: Agentic AI Track (Sep 23 - Oct 10)
# =============================================================================
phase2_dates = get_dates_in_range(PHASE2_START, PHASE2_END)
date_idx = 0

# Structure and requirements
commits.append({
    "id": commit_id,
    "date": random_time(phase2_dates[date_idx]),
    "message": "Add Agentic AI track structure and requirements",
    "files": [
        "Agentic AI/README.md",
        "Agentic AI/requirements.txt"
    ],
    "phase": 2
})
commit_id += 1
date_idx += 1

# src/ modules
src_commits = [
    ("Add utility functions for Agentic AI", ["Agentic AI/src/__init__.py", "Agentic AI/src/utils.py"]),
    ("Implement LLM client with multi-provider support", ["Agentic AI/src/llm_client.py"]),
    ("Add tool registry for function calling", ["Agentic AI/src/tool_registry.py"]),
    ("Implement agent framework with ReAct pattern", ["Agentic AI/src/agent_framework.py"]),
]

for msg, files in src_commits:
    commits.append({
        "id": commit_id,
        "date": random_time(phase2_dates[date_idx]),
        "message": msg,
        "files": files,
        "phase": 2
    })
    commit_id += 1
    date_idx += 1

# tools/
commits.append({
    "id": commit_id,
    "date": random_time(phase2_dates[date_idx]),
    "message": "Add calculator tool with math operations",
    "files": ["Agentic AI/tools/calculator.py"],
    "phase": 2
})
commit_id += 1
date_idx += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase2_dates[date_idx]),
    "message": "Add web search tool with mock API",
    "files": ["Agentic AI/tools/web_search.py"],
    "phase": 2
})
commit_id += 1
date_idx += 1

# Notebooks 01-08 with randomized commits per notebook
agentic_notebooks = [
    ("01_llm_basics", "LLM fundamentals", ["API setup", "token counting", "parameters", "exercises"]),
    ("02_prompt_engineering", "prompt engineering", ["zero-shot examples", "few-shot patterns", "CoT prompting", "templates"]),
    ("03_tool_use_fundamentals", "tool use basics", ["JSON schema", "tool definitions", "parsing", "exercises"]),
    ("04_function_calling", "function calling", ["OpenAI format", "Anthropic format", "multi-turn", "error handling"]),
    ("05_react_agents", "ReAct agent", ["pattern explanation", "implementation", "examples", "exercises"]),
    ("06_multi_agent_systems", "multi-agent systems", ["orchestration patterns", "sequential agents", "parallel execution"]),
    ("07_autonomous_workflows", "autonomous workflows", ["planning", "self-correction", "guardrails", "human-in-loop"]),
    ("08_capstone_agent", "capstone project template", ["project structure", "evaluation rubric"]),
]

for nb_name, topic, commit_parts in agentic_notebooks:
    # Random number of commits (2-6) per notebook
    num_commits = random.randint(2, min(6, len(commit_parts)))
    selected_parts = random.sample(commit_parts, num_commits)

    # First commit: initial notebook
    commits.append({
        "id": commit_id,
        "date": random_time(phase2_dates[min(date_idx, len(phase2_dates)-1)]),
        "message": f"Add {nb_name}: {topic} notebook",
        "files": [f"Agentic AI/notebooks/{nb_name}.ipynb"],
        "phase": 2
    })
    commit_id += 1
    date_idx += 1

    # Additional commits for this notebook
    for part in selected_parts[1:]:
        if date_idx < len(phase2_dates):
            commits.append({
                "id": commit_id,
                "date": random_time(phase2_dates[date_idx]),
                "message": f"Update {nb_name}: add {part}",
                "files": [f"Agentic AI/notebooks/{nb_name}.ipynb"],
                "phase": 2
            })
            commit_id += 1
            date_idx = min(date_idx + 1, len(phase2_dates) - 1)

# Documentation
commits.append({
    "id": commit_id,
    "date": random_time(phase2_dates[-2]),
    "message": "Add Agentic AI learning guide",
    "files": ["Agentic AI/docs/LEARNING_GUIDE.md"],
    "phase": 2
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase2_dates[-1]),
    "message": "Add agent patterns reference documentation",
    "files": ["Agentic AI/docs/AGENT_PATTERNS.md"],
    "phase": 2
})
commit_id += 1

# =============================================================================
# PHASE 3: AI Engineering Track (Oct 11 - Oct 28)
# =============================================================================
phase3_dates = get_dates_in_range(PHASE3_START, PHASE3_END)
date_idx = 0

# Structure and requirements
commits.append({
    "id": commit_id,
    "date": random_time(phase3_dates[date_idx]),
    "message": "Add AI Engineering track structure and requirements",
    "files": [
        "AI Engineering/README.md",
        "AI Engineering/requirements.txt"
    ],
    "phase": 3
})
commit_id += 1
date_idx += 1

# src/ modules
ai_eng_src = [
    ("Add LLM utilities with cost tracking", ["AI Engineering/src/__init__.py", "AI Engineering/src/llm_utils.py"]),
    ("Implement embedding utilities and vector store", ["AI Engineering/src/embedding_utils.py"]),
    ("Add RAG pipeline components", ["AI Engineering/src/rag_pipeline.py"]),
    ("Implement fine-tuning utilities", ["AI Engineering/src/finetuning_utils.py"]),
    ("Add evaluation framework with LLM-as-judge", ["AI Engineering/src/evaluation.py"]),
]

for msg, files in ai_eng_src:
    commits.append({
        "id": commit_id,
        "date": random_time(phase3_dates[date_idx]),
        "message": msg,
        "files": files,
        "phase": 3
    })
    commit_id += 1
    date_idx += 1

# Notebooks 01-10
ai_eng_notebooks = [
    ("01_llm_apis", "LLM API integration", ["OpenAI setup", "Anthropic setup", "streaming", "cost tracking"]),
    ("02_structured_outputs", "structured outputs", ["JSON mode", "Pydantic models", "entity extraction"]),
    ("03_embeddings_vectors", "embeddings and vectors", ["embedding creation", "similarity search", "ChromaDB"]),
    ("04_rag_fundamentals", "RAG pipeline", ["document loading", "chunking", "retrieval", "generation"]),
    ("05_advanced_rag", "advanced RAG", ["hybrid search", "re-ranking", "evaluation metrics"]),
    ("06_finetuning_basics", "fine-tuning basics", ["data preparation", "OpenAI API", "evaluation"]),
    ("07_lora_peft", "LoRA and PEFT", ["concept explanation", "implementation", "training"]),
    ("08_evaluation_testing", "evaluation framework", ["test cases", "LLM-as-judge", "regression testing"]),
    ("09_production_deployment", "production deployment", ["API design", "caching", "monitoring"]),
    ("10_capstone_app", "capstone application", ["complete system", "production features", "challenges"]),
]

for nb_name, topic, commit_parts in ai_eng_notebooks:
    num_commits = random.randint(2, min(5, len(commit_parts)))
    selected_parts = random.sample(commit_parts, num_commits)

    commits.append({
        "id": commit_id,
        "date": random_time(phase3_dates[min(date_idx, len(phase3_dates)-1)]),
        "message": f"Add {nb_name}: {topic} notebook",
        "files": [f"AI Engineering/notebooks/{nb_name}.ipynb"],
        "phase": 3
    })
    commit_id += 1
    date_idx += 1

    for part in selected_parts[1:]:
        if date_idx < len(phase3_dates):
            commits.append({
                "id": commit_id,
                "date": random_time(phase3_dates[date_idx]),
                "message": f"Update {nb_name}: add {part}",
                "files": [f"AI Engineering/notebooks/{nb_name}.ipynb"],
                "phase": 3
            })
            commit_id += 1
            date_idx = min(date_idx + 1, len(phase3_dates) - 1)

# Documentation
commits.append({
    "id": commit_id,
    "date": random_time(phase3_dates[-3]),
    "message": "Add AI Engineering learning guide",
    "files": ["AI Engineering/docs/LEARNING_GUIDE.md"],
    "phase": 3
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase3_dates[-2]),
    "message": "Add AI Engineering cheatsheet",
    "files": ["AI Engineering/docs/AI_ENG_CHEATSHEET.md"],
    "phase": 3
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase3_dates[-1]),
    "message": "Add self-assessment rubric for AI Engineering",
    "files": ["AI Engineering/evaluation/self_assessment.md"],
    "phase": 3
})
commit_id += 1

# =============================================================================
# PHASE 4: Final Polish (Oct 29-31)
# =============================================================================
phase4_dates = get_dates_in_range(PHASE4_START, PHASE4_END)

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[0]),
    "message": "Add dataset sources documentation with Kaggle links",
    "files": ["DATA_SOURCES.md"],
    "phase": 4
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[0]),
    "message": "Add environment variable templates",
    "files": [
        "Agentic AI/.env.example",
        "AI Engineering/.env.example",
        "ML Engineering/.env.example"
    ],
    "phase": 4
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[1]),
    "message": "Update ML Engineering README with dataset instructions",
    "files": ["ML Engineering/README.md"],
    "phase": 4
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[1]),
    "message": "Create data directory structure",
    "files": [
        "ML Engineering/data/raw/.gitkeep",
        "ML Engineering/models/.gitkeep",
        "ML Engineering/reports/.gitkeep",
        "AI Engineering/data/documents/.gitkeep",
        "AI Engineering/data/training_data/.gitkeep",
        "Agentic AI/data/examples/.gitkeep"
    ],
    "phase": 4
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[2]),
    "message": "Add comprehensive README for AI Academy",
    "files": ["README.md"],
    "phase": 4
})
commit_id += 1

commits.append({
    "id": commit_id,
    "date": random_time(phase4_dates[2]),
    "message": "Final cleanup and documentation polish",
    "files": [
        "Agentic AI/README.md",
        "AI Engineering/README.md"
    ],
    "phase": 4
})
commit_id += 1

# =============================================================================
# Sort commits by date and output
# =============================================================================
commits.sort(key=lambda x: x["date"])

# Reassign IDs after sorting
for i, commit in enumerate(commits):
    commit["id"] = i + 1

# Output summary
print(f"Generated {len(commits)} commits")
print(f"Date range: {commits[0]['date'][:10]} to {commits[-1]['date'][:10]}")
print(f"\nPhase breakdown:")
for phase in [1, 2, 3, 4]:
    count = len([c for c in commits if c["phase"] == phase])
    print(f"  Phase {phase}: {count} commits")

# Save to JSON
output = {
    "metadata": {
        "total_commits": len(commits),
        "date_range": {
            "start": "2025-09-20",
            "end": "2025-10-31"
        },
        "generated_at": datetime.now().isoformat()
    },
    "commits": commits
}

with open("commit_plan.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to commit_plan.json")

# Also print first 10 and last 5 commits for preview
print("\n" + "="*60)
print("FIRST 10 COMMITS:")
print("="*60)
for c in commits[:10]:
    print(f"[{c['id']:2d}] {c['date']} | {c['message'][:60]}")

print("\n" + "="*60)
print("LAST 5 COMMITS:")
print("="*60)
for c in commits[-5:]:
    print(f"[{c['id']:2d}] {c['date']} | {c['message'][:60]}")
