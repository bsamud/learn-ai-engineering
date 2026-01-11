#!/usr/bin/env python3
"""
H2O MLOps Inference API Client
Scores inference dataset using deployed model endpoint.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
import argparse


class H2OMLOpsClient:
    """Client for H2O MLOps model scoring."""

    def __init__(self, endpoint_url, api_key=None):
        """
        Initialize the client.

        Args:
            endpoint_url: Full URL to the model scoring endpoint
                         e.g., https://api.example.com/v1/models/<model-id>/score
            api_key: Optional API key for authentication
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

        # Metrics tracking
        self.latencies = []
        self.total_requests = 0
        self.failed_requests = 0

    def score_single(self, row_data):
        """
        Score a single row of data.

        Args:
            row_data: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        payload = {
            "fields": list(row_data.keys()),
            "rows": [[row_data[k] for k in row_data.keys()]]
        }

        start_time = time.time()
        try:
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.latencies.append(latency)
            self.total_requests += 1

            if response.status_code == 200:
                return response.json()
            else:
                self.failed_requests += 1
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            self.failed_requests += 1
            self.total_requests += 1
            return {"error": str(e)}

    def score_batch(self, df, batch_size=100, show_progress=True):
        """
        Score a dataframe in batches.

        Args:
            df: Pandas DataFrame with inference data
            batch_size: Number of rows per API call
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with predictions and probabilities
        """
        results = []
        n_samples = len(df)
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"\nScoring {n_samples:,} samples in {n_batches} batches...")
        print(f"Endpoint: {self.endpoint_url}")
        print(f"Batch size: {batch_size}")
        print("-" * 60)

        start_time = time.time()

        for i in range(0, n_samples, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                print(f"\rBatch {batch_num}/{n_batches} ({i+len(batch_df)}/{n_samples} samples)...", end="", flush=True)

            # Prepare batch payload
            payload = {
                "fields": list(batch_df.columns),
                "rows": batch_df.values.tolist()
            }

            batch_start = time.time()
            try:
                response = requests.post(
                    self.endpoint_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                batch_latency = (time.time() - batch_start) * 1000
                self.latencies.append(batch_latency)
                self.total_requests += 1

                if response.status_code == 200:
                    result = response.json()
                    # Parse H2O MLOps response format
                    batch_results = self._parse_response(result, batch_df)
                    results.extend(batch_results)
                else:
                    self.failed_requests += 1
                    print(f"\nError in batch {batch_num}: HTTP {response.status_code}")
                    # Add empty results for failed batch
                    for idx in batch_df.index:
                        results.append({
                            'ID': df.loc[idx, 'ID'] if 'ID' in df.columns else idx,
                            'prediction': None,
                            'probability_default': None,
                            'probability_no_default': None
                        })
            except Exception as e:
                self.failed_requests += 1
                self.total_requests += 1
                print(f"\nException in batch {batch_num}: {str(e)}")
                for idx in batch_df.index:
                    results.append({
                        'ID': df.loc[idx, 'ID'] if 'ID' in df.columns else idx,
                        'prediction': None,
                        'probability_default': None,
                        'probability_no_default': None
                    })

        total_time = time.time() - start_time
        print(f"\n\nScoring completed in {total_time:.2f} seconds")

        return pd.DataFrame(results)

    def _parse_response(self, response, batch_df):
        """
        Parse H2O MLOps scoring response.

        Expected response format:
        {
            "fields": ["DEFAULT_PAYMENT_NEXT_MONTH.0", "DEFAULT_PAYMENT_NEXT_MONTH.1"],
            "score": [[0.8, 0.2], [0.3, 0.7], ...]
        }
        """
        results = []

        try:
            fields = response.get('fields', [])
            scores = response.get('score', response.get('scores', []))

            # Determine which field is the probability of default (class 1)
            prob_default_idx = 1  # Usually second column is P(class=1)
            prob_no_default_idx = 0

            for field in fields:
                if '.1' in str(field) or 'DEFAULT' in str(field).upper():
                    prob_default_idx = fields.index(field)
                    prob_no_default_idx = 1 - prob_default_idx
                    break

            for i, score_row in enumerate(scores):
                if i < len(batch_df):
                    row_id = batch_df.iloc[i]['ID'] if 'ID' in batch_df.columns else batch_df.index[i]

                    if len(score_row) >= 2:
                        prob_no_default = float(score_row[prob_no_default_idx])
                        prob_default = float(score_row[prob_default_idx])
                    else:
                        prob_default = float(score_row[0])
                        prob_no_default = 1 - prob_default

                    prediction = 1 if prob_default >= 0.5 else 0

                    results.append({
                        'ID': row_id,
                        'prediction': prediction,
                        'probability_default': prob_default,
                        'probability_no_default': prob_no_default
                    })
        except Exception as e:
            print(f"\nError parsing response: {e}")
            print(f"Response: {response}")

        return results

    def get_metrics(self):
        """Get performance metrics from scoring session."""
        if not self.latencies:
            return {}

        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / self.total_requests * 100,
            'avg_latency_ms': np.mean(self.latencies),
            'min_latency_ms': np.min(self.latencies),
            'max_latency_ms': np.max(self.latencies),
            'p50_latency_ms': np.percentile(self.latencies, 50),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'total_latency_ms': np.sum(self.latencies)
        }

    def print_metrics(self):
        """Print formatted performance metrics."""
        metrics = self.get_metrics()
        if not metrics:
            print("No metrics available")
            return

        print("\n" + "=" * 60)
        print("INFERENCE PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Requests:     {metrics['total_requests']}")
        print(f"Failed Requests:    {metrics['failed_requests']}")
        print(f"Success Rate:       {metrics['success_rate']:.2f}%")
        print(f"\nLatency Statistics (milliseconds):")
        print(f"  Average:          {metrics['avg_latency_ms']:.2f} ms")
        print(f"  Minimum:          {metrics['min_latency_ms']:.2f} ms")
        print(f"  Maximum:          {metrics['max_latency_ms']:.2f} ms")
        print(f"  P50 (Median):     {metrics['p50_latency_ms']:.2f} ms")
        print(f"  P95:              {metrics['p95_latency_ms']:.2f} ms")
        print(f"  P99:              {metrics['p99_latency_ms']:.2f} ms")
        print(f"  Total Time:       {metrics['total_latency_ms']:.2f} ms")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Score inference dataset using H2O MLOps API')
    parser.add_argument('--endpoint', type=str, required=True,
                       help='H2O MLOps model endpoint URL')
    parser.add_argument('--input', type=str, default='data/inference_dataset.csv',
                       help='Path to inference dataset CSV')
    parser.add_argument('--output', type=str, default='data/scores.csv',
                       help='Path to save scoring results')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of samples per API request')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for authentication (optional)')
    parser.add_argument('--save-metrics', type=str, default='reports/inference_metrics.json',
                       help='Path to save performance metrics')

    args = parser.parse_args()

    print("=" * 60)
    print("H2O MLOps Inference Client")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load inference dataset
    print(f"\nLoading inference data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} features")

    # Initialize client
    client = H2OMLOpsClient(args.endpoint, args.api_key)

    # Score the dataset
    scores_df = client.score_batch(df, batch_size=args.batch_size)

    # Save scores
    scores_df.to_csv(args.output, index=False)
    print(f"\nScores saved to: {args.output}")
    print(f"  - Total predictions: {len(scores_df):,}")
    print(f"  - Predicted defaults: {(scores_df['prediction'] == 1).sum():,}")
    print(f"  - Predicted non-defaults: {(scores_df['prediction'] == 0).sum():,}")

    # Print and save metrics
    client.print_metrics()

    # Save metrics to JSON
    import os
    os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
    metrics = client.get_metrics()
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['endpoint'] = args.endpoint
    metrics['samples_scored'] = len(df)

    with open(args.save_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {args.save_metrics}")


if __name__ == "__main__":
    main()
