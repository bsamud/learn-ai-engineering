"""
Evaluation Framework for AI Engineering
=======================================

Testing, evaluation, and metrics for AI applications.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from datetime import datetime


# =============================================================================
# Test Cases
# =============================================================================


@dataclass
class TestCase:
    """A single test case for evaluation."""

    id: str
    input: str
    expected: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    actual: Optional[str] = None
    passed: Optional[bool] = None
    score: Optional[float] = None
    feedback: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    total: int
    passed: int
    failed: int
    avg_score: float
    results: list[TestCase]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Get a summary string."""
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        return (
            f"Results: {self.passed}/{self.total} passed ({pass_rate:.1f}%) | "
            f"Avg Score: {self.avg_score:.2f}"
        )


# =============================================================================
# Evaluators
# =============================================================================


class Evaluator:
    """
    Evaluate AI system outputs.

    Example
    -------
    >>> evaluator = Evaluator(llm_client)
    >>> evaluator.add_test("q1", "What is 2+2?", expected="4")
    >>> results = evaluator.run(my_system)
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.test_cases: list[TestCase] = []

    def add_test(
        self,
        id: str,
        input: str,
        expected: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add a test case."""
        self.test_cases.append(
            TestCase(id=id, input=input, expected=expected, metadata=metadata)
        )

    def add_tests_from_json(self, path: str) -> None:
        """Load test cases from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        for item in data:
            self.add_test(
                id=item.get("id", f"test_{len(self.test_cases)}"),
                input=item["input"],
                expected=item.get("expected"),
                **item.get("metadata", {}),
            )

        print(f"Loaded {len(data)} test cases from {path}")

    def run(
        self,
        system: Callable[[str], str],
        evaluator_fn: Optional[Callable[[str, str, Optional[str]], tuple[bool, float, str]]] = None,
    ) -> EvaluationResult:
        """
        Run evaluation on all test cases.

        Parameters
        ----------
        system : Callable
            Function that takes input and returns output
        evaluator_fn : Callable, optional
            Custom evaluation function (input, actual, expected) -> (passed, score, feedback)
        """
        results = []

        for test in self.test_cases:
            print(f"Running test: {test.id}")

            try:
                # Get system output
                actual = system(test.input)
                test.actual = actual

                # Evaluate
                if evaluator_fn:
                    passed, score, feedback = evaluator_fn(
                        test.input, actual, test.expected
                    )
                elif test.expected:
                    passed, score, feedback = self._exact_match(actual, test.expected)
                else:
                    passed, score, feedback = True, 1.0, "No expected value to compare"

                test.passed = passed
                test.score = score
                test.feedback = feedback

            except Exception as e:
                test.passed = False
                test.score = 0.0
                test.feedback = f"Error: {str(e)}"

            results.append(test)

        # Calculate aggregates
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score or 0 for r in results) / total if total > 0 else 0

        return EvaluationResult(
            total=total,
            passed=passed,
            failed=total - passed,
            avg_score=avg_score,
            results=results,
        )

    def _exact_match(
        self,
        actual: str,
        expected: str,
    ) -> tuple[bool, float, str]:
        """Exact string match evaluation."""
        # Normalize strings
        actual_norm = actual.strip().lower()
        expected_norm = expected.strip().lower()

        if actual_norm == expected_norm:
            return True, 1.0, "Exact match"

        # Check if expected is contained in actual
        if expected_norm in actual_norm:
            return True, 0.8, "Expected found in output"

        return False, 0.0, f"Expected '{expected}', got '{actual}'"


# =============================================================================
# LLM-as-Judge Evaluator
# =============================================================================


class LLMJudge:
    """
    Use an LLM to evaluate outputs.

    Example
    -------
    >>> judge = LLMJudge(llm_client)
    >>> score, feedback = judge.evaluate(
    ...     question="What is Python?",
    ...     answer="Python is a programming language.",
    ...     criteria="accuracy, completeness"
    ... )
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def evaluate(
        self,
        question: str,
        answer: str,
        criteria: str = "accuracy, relevance, completeness",
        reference: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        Evaluate an answer using LLM.

        Returns
        -------
        tuple[float, str]
            (score 0-1, feedback)
        """
        prompt = f"""Evaluate the following answer based on these criteria: {criteria}

Question: {question}

Answer: {answer}
"""

        if reference:
            prompt += f"\nReference Answer: {reference}\n"

        prompt += """
Provide your evaluation in this exact JSON format:
{
    "score": <0.0 to 1.0>,
    "feedback": "<brief explanation>"
}

JSON Response:"""

        response = self.llm.chat(prompt)

        try:
            # Parse JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])
            return result["score"], result["feedback"]
        except:
            return 0.5, "Could not parse evaluation"

    def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> tuple[str, str]:
        """
        Compare two answers.

        Returns
        -------
        tuple[str, str]
            (winner "A" or "B" or "tie", explanation)
        """
        prompt = f"""Compare these two answers to the question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better? Respond with JSON:
{{
    "winner": "A" or "B" or "tie",
    "explanation": "<brief explanation>"
}}

JSON Response:"""

        response = self.llm.chat(prompt)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])
            return result["winner"], result["explanation"]
        except:
            return "tie", "Could not determine winner"


# =============================================================================
# RAG Evaluation Metrics
# =============================================================================


def calculate_retrieval_metrics(
    retrieved_docs: list[str],
    relevant_docs: list[str],
) -> dict:
    """
    Calculate retrieval metrics.

    Parameters
    ----------
    retrieved_docs : list[str]
        Documents retrieved by the system
    relevant_docs : list[str]
        Ground truth relevant documents

    Returns
    -------
    dict
        {precision, recall, f1}
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_mrr(
    retrieved_docs: list[str],
    relevant_doc: str,
) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Parameters
    ----------
    retrieved_docs : list[str]
        Ordered list of retrieved documents
    relevant_doc : str
        The relevant document

    Returns
    -------
    float
        MRR score (0-1)
    """
    try:
        rank = retrieved_docs.index(relevant_doc) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0
