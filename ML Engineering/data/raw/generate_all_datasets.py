#!/usr/bin/env python3
"""
Generate all datasets for MLOps Capstone Project.
Creates training, inference, and ground truth datasets.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_credit_default_data(n_samples=10000, default_ratio=0.22):
    """Generate credit card default prediction dataset."""

    # ID
    ids = list(range(1, n_samples + 1))

    # LIMIT_BAL (credit limit in NT dollars)
    limit_bal = np.random.choice(
        [10000, 20000, 30000, 50000, 80000, 100000, 120000, 150000,
         200000, 250000, 300000, 360000, 500000, 800000, 1000000],
        size=n_samples,
        p=[0.02, 0.05, 0.08, 0.12, 0.10, 0.08, 0.07, 0.08,
           0.12, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    )

    # SEX (1=male, 2=female)
    sex = np.random.choice([1, 2], size=n_samples, p=[0.4, 0.6])

    # EDUCATION
    education = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples,
                                  p=[0.10, 0.45, 0.35, 0.05, 0.03, 0.02])

    # MARRIAGE
    marriage = np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.52, 0.03])

    # AGE
    age = np.random.normal(loc=35, scale=9, size=n_samples)
    age = np.clip(age, 21, 79).astype(int)

    # PAY_0 to PAY_6: Payment status
    pay_status = []
    for i in range(7):
        pay = np.random.choice(
            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            size=n_samples,
            p=[0.05, 0.35, 0.35, 0.10, 0.08, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002]
        )
        pay_status.append(pay)

    # BILL_AMT1 to BILL_AMT6
    bill_amounts = []
    for i in range(6):
        bill = limit_bal * np.random.uniform(0, 1.2, size=n_samples)
        bill = bill + np.random.normal(0, 5000, size=n_samples)
        bill = np.round(bill, 0).astype(int)
        bill_amounts.append(bill)

    # PAY_AMT1 to PAY_AMT6
    pay_amounts = []
    for i in range(6):
        pay = bill_amounts[i] * np.random.uniform(0, 0.5, size=n_samples)
        pay = np.maximum(0, pay)
        pay = np.round(pay, 0).astype(int)
        pay_amounts.append(pay)

    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_status[0],
        'PAY_2': pay_status[1],
        'PAY_3': pay_status[2],
        'PAY_4': pay_status[3],
        'PAY_5': pay_status[4],
        'PAY_6': pay_status[5],
        'BILL_AMT1': bill_amounts[0],
        'BILL_AMT2': bill_amounts[1],
        'BILL_AMT3': bill_amounts[2],
        'BILL_AMT4': bill_amounts[3],
        'BILL_AMT5': bill_amounts[4],
        'BILL_AMT6': bill_amounts[5],
        'PAY_AMT1': pay_amounts[0],
        'PAY_AMT2': pay_amounts[1],
        'PAY_AMT3': pay_amounts[2],
        'PAY_AMT4': pay_amounts[3],
        'PAY_AMT5': pay_amounts[4],
        'PAY_AMT6': pay_amounts[5]
    })

    # Generate DEFAULT_PAYMENT_NEXT_MONTH based on risk factors
    default_prob = np.full(n_samples, default_ratio)

    for i in range(n_samples):
        # Risk factors
        if df.loc[i, 'LIMIT_BAL'] < 50000:
            default_prob[i] *= 1.5

        max_delay = max([df.loc[i, f'PAY_{j}'] for j in [0, 2, 3, 4, 5, 6]])
        if max_delay > 0:
            default_prob[i] *= (1 + max_delay * 0.3)

        utilization = df.loc[i, 'BILL_AMT1'] / df.loc[i, 'LIMIT_BAL']
        if utilization > 0.8:
            default_prob[i] *= 1.4

        if df.loc[i, 'EDUCATION'] >= 3:
            default_prob[i] *= 1.2

        if df.loc[i, 'AGE'] < 25:
            default_prob[i] *= 1.3

        default_prob[i] = min(default_prob[i], 0.9)

    # Assign default labels
    df['DEFAULT_PAYMENT_NEXT_MONTH'] = (np.random.random(n_samples) < default_prob).astype(int)

    # Adjust to target ratio
    actual_ratio = df['DEFAULT_PAYMENT_NEXT_MONTH'].sum() / n_samples
    if actual_ratio < default_ratio * 0.9:
        n_to_flip = int((default_ratio - actual_ratio) * n_samples)
        non_default_idx = df[df['DEFAULT_PAYMENT_NEXT_MONTH'] == 0].index.tolist()
        flip_idx = random.sample(non_default_idx, min(n_to_flip, len(non_default_idx)))
        df.loc[flip_idx, 'DEFAULT_PAYMENT_NEXT_MONTH'] = 1

    return df


def main():
    print("=" * 60)
    print("MLOps Capstone Project - Dataset Generator")
    print("=" * 60)

    # Generate training dataset (10,000 samples)
    print("\n[1/3] Generating training dataset...")
    train_df = generate_credit_default_data(n_samples=10000, default_ratio=0.22)
    train_df.to_csv('training_dataset.csv', index=False)

    print(f"  - Shape: {train_df.shape}")
    print(f"  - Default ratio: {train_df['DEFAULT_PAYMENT_NEXT_MONTH'].mean():.2%}")
    print(f"  - Saved to: training_dataset.csv")

    # Generate inference dataset (2,000 samples with different seed)
    print("\n[2/3] Generating inference dataset...")
    np.random.seed(123)  # Different seed for inference data
    random.seed(123)

    inference_full = generate_credit_default_data(n_samples=2000, default_ratio=0.22)

    # Adjust IDs to continue from training set
    inference_full['ID'] = inference_full['ID'] + 10000

    # Create ground truth (with labels)
    ground_truth_df = inference_full[['ID', 'DEFAULT_PAYMENT_NEXT_MONTH']].copy()
    ground_truth_df.columns = ['ID', 'actual_default']
    ground_truth_df.to_csv('ground_truth.csv', index=False)

    print(f"  - Ground truth saved to: ground_truth.csv")
    print(f"  - Shape: {ground_truth_df.shape}")
    print(f"  - Actual default ratio: {ground_truth_df['actual_default'].mean():.2%}")

    # Create inference dataset (WITHOUT target column)
    inference_df = inference_full.drop(columns=['DEFAULT_PAYMENT_NEXT_MONTH'])
    inference_df.to_csv('inference_dataset.csv', index=False)

    print(f"\n[3/3] Saving inference dataset...")
    print(f"  - Shape: {inference_df.shape}")
    print(f"  - Columns: {len(inference_df.columns)} (target column removed)")
    print(f"  - Saved to: inference_dataset.csv")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  1. training_dataset.csv     - {train_df.shape[0]:,} samples for model training")
    print(f"  2. inference_dataset.csv    - {inference_df.shape[0]:,} samples for API scoring")
    print(f"  3. ground_truth.csv         - Actual labels for model monitoring")

    print("\nNext Steps:")
    print("  Step 2: Train model with training_dataset.csv in H2O Driverless AI")
    print("  Step 3: Deploy model to H2O MLOps and get endpoint")
    print("  Step 4: Score inference_dataset.csv using API endpoint")
    print("  Step 5: Compare scores.csv with ground_truth.csv for monitoring")
    print("  Step 6: Generate performance report")


if __name__ == "__main__":
    main()
