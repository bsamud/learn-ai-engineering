#!/usr/bin/env python3
"""
Generate synthetic credit card default payment dataset.
Based on UCI Default of Credit Card Clients Dataset structure.
"""

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_training_data(n_samples=10000, default_ratio=0.22):
    """Generate training dataset with credit card default data."""

    # ID
    ids = list(range(1, n_samples + 1))

    # LIMIT_BAL (credit limit in NT dollars, typically 10k - 1M)
    limit_bal = np.random.choice(
        [10000, 20000, 30000, 50000, 80000, 100000, 120000, 150000,
         200000, 250000, 300000, 360000, 500000, 800000, 1000000],
        size=n_samples,
        p=[0.02, 0.05, 0.08, 0.12, 0.10, 0.08, 0.07, 0.08,
           0.12, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    )

    # SEX (1=male, 2=female)
    sex = np.random.choice([1, 2], size=n_samples, p=[0.4, 0.6])

    # EDUCATION (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    education = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples,
                                  p=[0.10, 0.45, 0.35, 0.05, 0.03, 0.02])

    # MARRIAGE (1=married, 2=single, 3=others)
    marriage = np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.52, 0.03])

    # AGE (21-79 years)
    age = np.random.normal(loc=35, scale=9, size=n_samples)
    age = np.clip(age, 21, 79).astype(int)

    # PAY_0 to PAY_6: Payment status
    # -2=no consumption, -1=pay duly, 0=use of revolving credit,
    # 1=payment delay for one month, 2=payment delay for two months, etc.
    pay_status = []
    for i in range(7):  # PAY_0 through PAY_6
        pay = np.random.choice(
            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            size=n_samples,
            p=[0.05, 0.35, 0.35, 0.10, 0.08, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002]
        )
        pay_status.append(pay)

    # BILL_AMT1 to BILL_AMT6: Amount of bill statement
    bill_amounts = []
    for i in range(6):
        # Bill amount correlates with credit limit
        bill = limit_bal * np.random.uniform(0, 1.2, size=n_samples)
        # Add some noise and allow negative (overpayment)
        bill = bill + np.random.normal(0, 5000, size=n_samples)
        bill = np.round(bill, 0).astype(int)
        bill_amounts.append(bill)

    # PAY_AMT1 to PAY_AMT6: Amount of previous payment
    pay_amounts = []
    for i in range(6):
        # Payment amount based on bill amount with some randomness
        pay = bill_amounts[i] * np.random.uniform(0, 0.5, size=n_samples)
        pay = np.maximum(0, pay)  # No negative payments
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
        # Higher default risk for:
        # - Lower credit limits
        if df.loc[i, 'LIMIT_BAL'] < 50000:
            default_prob[i] *= 1.5

        # - Payment delays
        max_delay = max([df.loc[i, f'PAY_{j}'] for j in [0, 2, 3, 4, 5, 6]])
        if max_delay > 0:
            default_prob[i] *= (1 + max_delay * 0.3)

        # - High utilization (bill / limit)
        utilization = df.loc[i, 'BILL_AMT1'] / df.loc[i, 'LIMIT_BAL']
        if utilization > 0.8:
            default_prob[i] *= 1.4

        # - Lower education
        if df.loc[i, 'EDUCATION'] >= 3:
            default_prob[i] *= 1.2

        # - Younger age
        if df.loc[i, 'AGE'] < 25:
            default_prob[i] *= 1.3

        # Cap probability
        default_prob[i] = min(default_prob[i], 0.9)

    # Assign default labels
    df['DEFAULT_PAYMENT_NEXT_MONTH'] = (np.random.random(n_samples) < default_prob).astype(int)

    # Adjust to target ratio if needed
    actual_ratio = df['DEFAULT_PAYMENT_NEXT_MONTH'].sum() / n_samples
    if actual_ratio < default_ratio * 0.9:
        n_to_flip = int((default_ratio - actual_ratio) * n_samples)
        non_default_idx = df[df['DEFAULT_PAYMENT_NEXT_MONTH'] == 0].index.tolist()
        flip_idx = random.sample(non_default_idx, min(n_to_flip, len(non_default_idx)))
        df.loc[flip_idx, 'DEFAULT_PAYMENT_NEXT_MONTH'] = 1

    return df

# Generate training dataset
print("Generating training_dataset.csv...")
train_df = generate_training_data(n_samples=10000, default_ratio=0.22)
train_df.to_csv('training_dataset.csv', index=False)

print(f"Training dataset created:")
print(f"  - Shape: {train_df.shape}")
print(f"  - Columns: {list(train_df.columns)}")
print(f"  - Default ratio: {train_df['DEFAULT_PAYMENT_NEXT_MONTH'].sum() / len(train_df):.2%}")
print(f"  - Defaults: {train_df['DEFAULT_PAYMENT_NEXT_MONTH'].sum():,}")
print(f"  - Non-defaults: {(train_df['DEFAULT_PAYMENT_NEXT_MONTH'] == 0).sum():,}")
print(f"\nSaved to: training_dataset.csv")

# Show sample
print("\nSample data (first 3 rows):")
print(train_df.head(3).to_string())
