# Dataset Sources for AI Academy

This document lists all required datasets and their download sources.

---

## Quick Start

### Option 1: Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (download kaggle.json from your Kaggle account settings)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download datasets
cd "ML Engineering/data/raw"
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction --unzip
kaggle datasets download -d laotse/credit-risk-dataset --unzip
```

### Option 2: Manual Download

Visit each Kaggle link below and click "Download" button.

---

## Required Datasets

### 1. Credit Card Fraud Detection

| Property | Value |
|----------|-------|
| **Filename** | `creditcard.csv` |
| **Size** | ~150 MB (compressed) |
| **Rows** | 284,807 transactions |
| **Kaggle Link** | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **Used By** | `05_xgboost_fraud_detection.ipynb`, `07_xgboost_fintech_advanced.ipynb`, ML Engineering notebooks |

**Description**: Anonymized credit card transactions with fraud labels. Highly imbalanced (0.172% fraud).

**Download Command**:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p "ML Engineering/data/raw"
```

---

### 2. Customer Churn

| Property | Value |
|----------|-------|
| **Filename** | `customer_churn.csv` or `Churn_Modelling.csv` |
| **Size** | ~1 MB |
| **Rows** | ~10,000 customers |
| **Kaggle Link** | https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction |
| **Alternative** | https://www.kaggle.com/datasets/shubh0799/churn-modelling |
| **Used By** | `01_classification_model.ipynb` |

**Description**: Customer demographics and churn status for binary classification.

**Download Command**:
```bash
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction --unzip -p "ML Engineering/data/raw"
```

---

### 3. Credit Risk / Loan Default

| Property | Value |
|----------|-------|
| **Filename** | `credit_risk.csv` |
| **Size** | ~5 MB |
| **Rows** | ~32,000 loan applications |
| **Kaggle Link** | https://www.kaggle.com/datasets/laotse/credit-risk-dataset |
| **Alternative** | https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset |
| **Used By** | `02_regression.ipynb`, `04_feature_engineering.ipynb` |

**Description**: Loan application data for credit risk assessment and default prediction.

**Download Command**:
```bash
kaggle datasets download -d laotse/credit-risk-dataset --unzip -p "ML Engineering/data/raw"
```

---

## Expected Directory Structure

After downloading, your data should be organized as:

```
academy/
├── ML Engineering/
│   └── data/
│       └── raw/
│           ├── creditcard.csv      # Fraud detection
│           ├── customer_churn.csv  # Customer churn
│           └── credit_risk.csv     # Loan default
├── AI Engineering/
│   └── data/
│       ├── documents/              # Sample docs (created by notebooks)
│       └── training_data/          # Fine-tuning data (created by notebooks)
└── Agentic AI/
    └── data/
        └── examples/               # Example prompts (optional)
```

---

## Track-Specific Requirements

### Agentic AI Track
**No datasets required** - All notebooks use API calls and in-memory examples.

**Requirements**:
- OpenAI API key
- Anthropic API key (optional)

### AI Engineering Track
**No pre-downloaded datasets required** - Notebooks create sample data dynamically.

**Requirements**:
- OpenAI API key
- Anthropic API key (optional)
- ChromaDB (auto-installed)

### ML Engineering
**Requires Kaggle datasets** - Download before running notebooks.

**Requirements**:
- creditcard.csv (fraud detection)
- customer_churn.csv (classification)
- credit_risk.csv (regression)

---

## Troubleshooting

### "Dataset not found" Error
1. Verify file exists in `data/raw/`
2. Check filename matches what notebook expects
3. Some datasets may need renaming after download

### Kaggle CLI Authentication
```bash
# Check if authenticated
kaggle datasets list

# If error, re-configure:
# 1. Go to kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Move downloaded kaggle.json to ~/.kaggle/
```

### Large File Issues
The creditcard.csv file is ~150MB. If you have slow internet:
1. Download manually from Kaggle website
2. Use smaller sample for testing first
3. Consider using Git LFS for version control

---

## License Information

| Dataset | License |
|---------|---------|
| Credit Card Fraud | Database Contents License (DbCL) |
| Customer Churn | CC0: Public Domain |
| Credit Risk | CC0: Public Domain |

Please review each dataset's license before commercial use.
