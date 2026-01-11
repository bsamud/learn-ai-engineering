"""
Fine-tuning Utilities for AI Engineering
========================================

Data preparation and fine-tuning helpers for LLMs.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# =============================================================================
# Data Formatting
# =============================================================================


@dataclass
class TrainingExample:
    """A single training example."""

    instruction: str
    input: str = ""
    output: str = ""
    system: str = ""


def format_for_openai(
    examples: list[TrainingExample],
    output_path: str,
) -> str:
    """
    Format training data for OpenAI fine-tuning.

    Creates a JSONL file with the chat format.

    Parameters
    ----------
    examples : list[TrainingExample]
        Training examples
    output_path : str
        Path to save the JSONL file

    Returns
    -------
    str
        Path to the created file
    """
    with open(output_path, "w") as f:
        for ex in examples:
            messages = []

            if ex.system:
                messages.append({"role": "system", "content": ex.system})

            # Combine instruction and input for user message
            user_content = ex.instruction
            if ex.input:
                user_content += f"\n\n{ex.input}"

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex.output})

            f.write(json.dumps({"messages": messages}) + "\n")

    print(f"Created {len(examples)} examples at {output_path}")
    return output_path


def format_for_alpaca(
    examples: list[TrainingExample],
    output_path: str,
) -> str:
    """
    Format training data in Alpaca format.

    Parameters
    ----------
    examples : list[TrainingExample]
        Training examples
    output_path : str
        Path to save the JSON file

    Returns
    -------
    str
        Path to the created file
    """
    data = []
    for ex in examples:
        data.append({
            "instruction": ex.instruction,
            "input": ex.input,
            "output": ex.output,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created {len(examples)} examples at {output_path}")
    return output_path


def load_training_data(path: str) -> list[TrainingExample]:
    """
    Load training data from a file.

    Supports JSON and JSONL formats.
    """
    path = Path(path)
    examples = []

    if path.suffix == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data:
                    # OpenAI format
                    messages = data["messages"]
                    system = ""
                    instruction = ""
                    output = ""

                    for msg in messages:
                        if msg["role"] == "system":
                            system = msg["content"]
                        elif msg["role"] == "user":
                            instruction = msg["content"]
                        elif msg["role"] == "assistant":
                            output = msg["content"]

                    examples.append(
                        TrainingExample(
                            instruction=instruction,
                            output=output,
                            system=system,
                        )
                    )
                else:
                    # Alpaca-like format
                    examples.append(
                        TrainingExample(
                            instruction=data.get("instruction", ""),
                            input=data.get("input", ""),
                            output=data.get("output", ""),
                        )
                    )
    else:
        with open(path, "r") as f:
            data = json.load(f)

        for item in data:
            examples.append(
                TrainingExample(
                    instruction=item.get("instruction", ""),
                    input=item.get("input", ""),
                    output=item.get("output", ""),
                )
            )

    print(f"Loaded {len(examples)} training examples from {path}")
    return examples


# =============================================================================
# OpenAI Fine-tuning
# =============================================================================


class OpenAIFineTuner:
    """
    Helper for OpenAI fine-tuning.

    Example
    -------
    >>> tuner = OpenAIFineTuner()
    >>> job = tuner.create_job("training_data.jsonl", "gpt-3.5-turbo")
    >>> tuner.wait_for_completion(job.id)
    """

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    def upload_file(self, path: str) -> str:
        """Upload a training file."""
        with open(path, "rb") as f:
            response = self.client.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded file: {response.id}")
        return response.id

    def create_job(
        self,
        training_file: str,
        model: str = "gpt-3.5-turbo",
        epochs: Optional[int] = None,
        suffix: Optional[str] = None,
    ) -> dict:
        """
        Create a fine-tuning job.

        Parameters
        ----------
        training_file : str
            Path to training file or file ID
        model : str
            Base model
        epochs : int, optional
            Number of epochs
        suffix : str, optional
            Model name suffix
        """
        # Upload if it's a path
        if os.path.exists(training_file):
            file_id = self.upload_file(training_file)
        else:
            file_id = training_file

        hyperparameters = {}
        if epochs:
            hyperparameters["n_epochs"] = epochs

        job = self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            hyperparameters=hyperparameters if hyperparameters else None,
            suffix=suffix,
        )

        print(f"Created fine-tuning job: {job.id}")
        return job

    def get_status(self, job_id: str) -> dict:
        """Get job status."""
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "error": job.error,
        }

    def list_jobs(self, limit: int = 10) -> list[dict]:
        """List recent fine-tuning jobs."""
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        return [self.get_status(job.id) for job in jobs.data]

    def cancel_job(self, job_id: str) -> None:
        """Cancel a fine-tuning job."""
        self.client.fine_tuning.jobs.cancel(job_id)
        print(f"Cancelled job: {job_id}")


# =============================================================================
# Data Quality Checks
# =============================================================================


def validate_training_data(examples: list[TrainingExample]) -> dict:
    """
    Validate training data quality.

    Returns
    -------
    dict
        Validation results and statistics
    """
    results = {
        "total": len(examples),
        "valid": 0,
        "issues": [],
        "stats": {
            "avg_instruction_length": 0,
            "avg_output_length": 0,
            "empty_outputs": 0,
            "duplicates": 0,
        },
    }

    instruction_lengths = []
    output_lengths = []
    seen_instructions = set()

    for i, ex in enumerate(examples):
        issues = []

        # Check for empty fields
        if not ex.instruction.strip():
            issues.append("Empty instruction")
        if not ex.output.strip():
            issues.append("Empty output")
            results["stats"]["empty_outputs"] += 1

        # Check for duplicates
        if ex.instruction in seen_instructions:
            results["stats"]["duplicates"] += 1
        seen_instructions.add(ex.instruction)

        # Track lengths
        instruction_lengths.append(len(ex.instruction))
        output_lengths.append(len(ex.output))

        if issues:
            results["issues"].append({"index": i, "issues": issues})
        else:
            results["valid"] += 1

    # Calculate averages
    if instruction_lengths:
        results["stats"]["avg_instruction_length"] = sum(instruction_lengths) / len(
            instruction_lengths
        )
    if output_lengths:
        results["stats"]["avg_output_length"] = sum(output_lengths) / len(
            output_lengths
        )

    return results


def split_data(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split data into train/val/test sets.

    Returns
    -------
    tuple
        (train_examples, val_examples, test_examples)
    """
    import random

    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test
