# MLOps Capstone Project: Credit Default Prediction

## Overview

This capstone project guides students through a complete MLOps pipeline for credit default prediction, including model training, deployment, inference API usage, and model monitoring.

## Problem Statement

Build an end-to-end MLOps pipeline to predict credit card payment defaults using H2O Driverless AI and H2O MLOps platform.

## Project Structure

```
ML Engineering/
├── README.md                           # This file
├── data/
│   ├── training_dataset.csv           # 10,000 samples for model training
│   ├── inference_dataset.csv          # 2,000 samples for API scoring
│   ├── ground_truth.csv               # Actual labels for monitoring
│   └── scores.csv                     # Model predictions (generated)
├── notebooks/
│   ├── 01_API_Inference_Scoring.ipynb # Step 4: Score data using API
│   ├── 02_Model_Monitoring.ipynb      # Step 5: Evaluate model performance
│   └── 03_Report_Template.ipynb       # Step 6: Generate final report
├── reports/
│   ├── inference_metrics.json         # API performance metrics
│   ├── monitoring_report.json         # Model monitoring results
│   └── *.png                          # Visualization plots
├── evaluation/
│   └── educator_rubric.md             # Grading rubric for educators
└── src/
    └── *.py                           # Utility scripts
```

## Dataset Description

### Training Dataset (training_dataset.csv)
- **Samples**: 10,000
- **Features**: 24 input features
- **Target**: DEFAULT_PAYMENT_NEXT_MONTH (0 = No default, 1 = Default)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| ID | int | Customer ID |
| LIMIT_BAL | int | Credit limit (NT dollars) |
| SEX | int | 1=Male, 2=Female |
| EDUCATION | int | 1=Graduate, 2=University, 3=High school, 4-6=Other |
| MARRIAGE | int | 1=Married, 2=Single, 3=Other |
| AGE | int | Age in years |
| PAY_0 to PAY_6 | int | Payment status (-2 to 8) |
| BILL_AMT1-6 | int | Bill amounts for 6 months |
| PAY_AMT1-6 | int | Payment amounts for 6 months |

## Project Steps

### Step 1: Data Preparation ✅
**Status**: Pre-completed by instructor

Datasets are provided:
- `training_dataset.csv` - For model training
- `inference_dataset.csv` - For API scoring (no labels)
- `ground_truth.csv` - Actual labels for monitoring

### Step 2: Model Training (H2O Driverless AI)
**Team Task**

1. Upload `training_dataset.csv` to H2O Driverless AI
2. Configure experiment:
   - Target: `DEFAULT_PAYMENT_NEXT_MONTH`
   - Problem type: Binary classification
3. Train the model
4. Download MOJO.zip file
5. Document training metrics and feature importance

### Step 3: Model Deployment (H2O MLOps)
**Team Task**

1. Create project in H2O MLOps UI
2. Upload MOJO.zip as new experiment
3. Register model
4. Create deployment
5. Obtain scoring endpoint URL

### Step 4: API Inference Scoring
**Team Task**

Use `notebooks/01_API_Inference_Scoring.ipynb`:

1. Configure your endpoint URL
2. Score `inference_dataset.csv` using the API
3. Save predictions to `scores.csv`
4. Record API performance metrics

**Output**: `data/scores.csv`, `reports/inference_metrics.json`

### Step 5: Model Monitoring
**Team Task**

Use `notebooks/02_Model_Monitoring.ipynb`:

1. Compare predictions with ground truth
2. Calculate classification metrics
3. Analyze confusion matrix
4. Assess business impact
5. Detect model drift

**Output**: `reports/monitoring_report.json`, visualizations

### Step 6: Report Generation
**Team Task**

Use `notebooks/03_Report_Template.ipynb`:

1. Summarize all project steps
2. Analyze model performance
3. Provide recommendations
4. Document lessons learned

**Output**: Complete project report

### Step 7: Educator Evaluation

Instructors evaluate reports using `evaluation/educator_rubric.md`

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Access to H2O Driverless AI console
- Access to H2O MLOps platform

### Download Datasets

If using alternative datasets from Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets (requires Kaggle API credentials)
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data/raw
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction --unzip -p data/raw
kaggle datasets download -d laotse/credit-risk-dataset --unzip -p data/raw
```

Or download manually from:
- Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Bank Churn: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
- Credit Risk: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

See [DATA_SOURCES.md](../DATA_SOURCES.md) for complete instructions.

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy requests scikit-learn matplotlib seaborn jupyter
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

## Team Assignments

This project is designed for **5 teams** working on the same problem statement.

**Team Deliverables**:
1. Completed notebooks with executed cells
2. Generated data files (scores.csv)
3. Performance reports (JSON files)
4. Final report with analysis and recommendations

## Evaluation Criteria

Total: 100 points + 10 bonus points

- Model Training: 25 points
- Model Deployment: 20 points
- API Inference: 20 points
- Model Monitoring: 25 points
- Report Quality: 10 points
- Bonus: Up to 10 points

See `evaluation/educator_rubric.md` for detailed rubric.

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Model Training | Day 1-2 | Trained model, MOJO file |
| Deployment | Day 3 | Active API endpoint |
| Inference & Monitoring | Day 4-5 | scores.csv, metrics |
| Report Writing | Day 6-7 | Final report |

## Support

For technical issues:
- Check H2O documentation
- Review notebook instructions
- Consult with your instructor

## License

This project is for educational purposes only.

---

**Good luck with your capstone project!**
