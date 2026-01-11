#!/usr/bin/env python3
"""
Mock Inference for Testing
Simulates H2O MLOps API responses for development/testing.
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os


def simulate_model_prediction(row):
    """
    Simulate model prediction based on feature values.
    This mimics what the actual H2O model might predict.
    """
    base_prob = 0.22  # Base default probability

    # Risk factors that increase default probability
    if row['LIMIT_BAL'] < 50000:
        base_prob *= 1.4

    # Payment delays
    max_delay = max([row[f'PAY_{i}'] for i in [0, 2, 3, 4, 5, 6]])
    if max_delay > 0:
        base_prob *= (1 + max_delay * 0.25)

    # High utilization
    utilization = row['BILL_AMT1'] / row['LIMIT_BAL'] if row['LIMIT_BAL'] > 0 else 0
    if utilization > 0.8:
        base_prob *= 1.3

    # Education
    if row['EDUCATION'] >= 3:
        base_prob *= 1.15

    # Age
    if row['AGE'] < 25:
        base_prob *= 1.2

    # Cap probability
    prob_default = min(base_prob, 0.95)

    # Add some noise to simulate model uncertainty
    noise = np.random.normal(0, 0.05)
    prob_default = np.clip(prob_default + noise, 0.01, 0.99)

    return prob_default


def mock_score_dataset(input_path, output_path, metrics_path=None):
    """
    Score the inference dataset using mock predictions.

    Args:
        input_path: Path to inference_dataset.csv
        output_path: Path to save scores.csv
        metrics_path: Path to save metrics (optional)
    """
    print("=" * 60)
    print("Mock Inference Scoring (for Development/Testing)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} samples")

    # Simulate scoring with latency
    print("\nScoring samples...")
    latencies = []
    results = []

    np.random.seed(int(time.time()))  # Different results each run

    for idx, row in df.iterrows():
        start_time = time.time()

        # Simulate model prediction
        prob_default = simulate_model_prediction(row)
        prob_no_default = 1 - prob_default
        prediction = 1 if prob_default >= 0.5 else 0

        # Simulate network latency (5-50ms per sample)
        simulated_latency = np.random.uniform(5, 50)
        time.sleep(simulated_latency / 1000)  # Convert to seconds

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        results.append({
            'ID': row['ID'],
            'prediction': prediction,
            'probability_default': round(prob_default, 6),
            'probability_no_default': round(prob_no_default, 6)
        })

        if (idx + 1) % 100 == 0:
            print(f"\r  Processed {idx + 1}/{len(df)} samples...", end="", flush=True)

    print(f"\r  Processed {len(df)}/{len(df)} samples... Done!")

    # Create results DataFrame
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(output_path, index=False)

    print(f"\nScores saved to: {output_path}")
    print(f"  - Total predictions: {len(scores_df):,}")
    print(f"  - Predicted defaults: {(scores_df['prediction'] == 1).sum():,}")
    print(f"  - Predicted non-defaults: {(scores_df['prediction'] == 0).sum():,}")
    print(f"  - Average probability: {scores_df['probability_default'].mean():.4f}")

    # Calculate metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': 'MOCK_ENDPOINT',
        'samples_scored': len(df),
        'total_requests': len(df),
        'failed_requests': 0,
        'success_rate': 100.0,
        'avg_latency_ms': np.mean(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'total_latency_ms': np.sum(latencies)
    }

    # Print metrics
    print("\n" + "=" * 60)
    print("INFERENCE PERFORMANCE METRICS (SIMULATED)")
    print("=" * 60)
    print(f"Total Requests:     {metrics['total_requests']}")
    print(f"Success Rate:       {metrics['success_rate']:.2f}%")
    print(f"\nLatency Statistics (milliseconds):")
    print(f"  Average:          {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Minimum:          {metrics['min_latency_ms']:.2f} ms")
    print(f"  Maximum:          {metrics['max_latency_ms']:.2f} ms")
    print(f"  P50 (Median):     {metrics['p50_latency_ms']:.2f} ms")
    print(f"  P95:              {metrics['p95_latency_ms']:.2f} ms")
    print(f"  P99:              {metrics['p99_latency_ms']:.2f} ms")
    print("=" * 60)

    # Save metrics
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

    return scores_df, metrics


if __name__ == "__main__":
    # Default paths
    input_path = "data/inference_dataset.csv"
    output_path = "data/scores.csv"
    metrics_path = "reports/inference_metrics.json"

    # Run mock scoring
    scores_df, metrics = mock_score_dataset(input_path, output_path, metrics_path)
