#!/usr/bin/env python3
"""
Prepare datasets from Kaggle creditcard.csv for MLOps Capstone Project.

Usage:
1. Download creditcard.csv from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place creditcard.csv in this directory (data/raw/)
3. Run: python prepare_datasets.py

This script creates:
- training_dataset.csv: 80% of data with fraud labels for H2O Driverless AI training
- inference_dataset.csv: 20% of data WITHOUT labels for model inference/scoring
- ground_truth.csv: Labels for inference dataset (for model monitoring)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main():
    # Check if creditcard.csv exists
    if not os.path.exists('creditcard.csv'):
        print("ERROR: creditcard.csv not found!")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\nPlace the creditcard.csv file in this directory and run again.")
        return

    print("Loading creditcard.csv...")
    df = pd.read_csv('creditcard.csv')

    print(f"Original dataset shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].sum() / len(df):.4%}")
    print(f"Total fraudulent transactions: {df['Class'].sum():,}")
    print(f"Total legitimate transactions: {(df['Class'] == 0).sum():,}")

    # Rename 'Class' to 'is_fraud' for clarity
    df = df.rename(columns={'Class': 'is_fraud'})

    # Add transaction IDs for tracking
    df.insert(0, 'transaction_id', [f'TXN_{i:08d}' for i in range(1, len(df) + 1)])

    # Split into training (80%) and inference (20%)
    # Use stratified split to maintain fraud ratio in both sets
    train_df, inference_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_fraud']
    )

    print(f"\nTraining set: {len(train_df):,} samples")
    print(f"Inference set: {len(inference_df):,} samples")

    # Save training dataset (with labels)
    train_df = train_df.reset_index(drop=True)
    train_df.to_csv('training_dataset.csv', index=False)
    print(f"\nSaved training_dataset.csv")
    print(f"  - Shape: {train_df.shape}")
    print(f"  - Fraud ratio: {train_df['is_fraud'].sum() / len(train_df):.4%}")
    print(f"  - Fraudulent: {train_df['is_fraud'].sum():,}")
    print(f"  - Legitimate: {(train_df['is_fraud'] == 0).sum():,}")

    # Create ground truth (labels for inference set)
    inference_df = inference_df.reset_index(drop=True)
    ground_truth_df = inference_df[['transaction_id', 'is_fraud']].copy()
    ground_truth_df.columns = ['transaction_id', 'actual_fraud']
    ground_truth_df.to_csv('ground_truth.csv', index=False)
    print(f"\nSaved ground_truth.csv")
    print(f"  - Shape: {ground_truth_df.shape}")
    print(f"  - Fraud ratio: {ground_truth_df['actual_fraud'].sum() / len(ground_truth_df):.4%}")

    # Save inference dataset (WITHOUT labels - students will score this)
    inference_df_no_labels = inference_df.drop(columns=['is_fraud'])
    inference_df_no_labels.to_csv('inference_dataset.csv', index=False)
    print(f"\nSaved inference_dataset.csv")
    print(f"  - Shape: {inference_df_no_labels.shape}")
    print(f"  - Note: Labels removed for inference")

    # Create a sample scores template
    sample_scores = pd.DataFrame({
        'transaction_id': inference_df_no_labels['transaction_id'].head(5),
        'fraud_probability': [0.0, 0.0, 0.0, 0.0, 0.0],
        'predicted_fraud': [0, 0, 0, 0, 0]
    })
    sample_scores.to_csv('sample_scores_template.csv', index=False)
    print(f"\nSaved sample_scores_template.csv (example format for API output)")

    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("1. training_dataset.csv - Use this to train model in H2O Driverless AI")
    print("2. inference_dataset.csv - Use this for model inference/scoring API")
    print("3. ground_truth.csv - Use this with scores.csv for model monitoring")
    print("4. sample_scores_template.csv - Expected format for inference output")
    print("\nNext steps:")
    print("1. Upload training_dataset.csv to H2O Driverless AI")
    print("2. Train your fraud detection model")
    print("3. Download the MOJO file")
    print("4. Create inference API using the MOJO")
    print("5. Score inference_dataset.csv to generate scores.csv")
    print("6. Compare scores.csv with ground_truth.csv for monitoring")

if __name__ == "__main__":
    main()
