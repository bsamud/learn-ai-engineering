# FintelHub Capstone Project Guide

**ML Engineering Workshop - Capstone Project**

---

## Overview

This capstone project is designed to give you hands-on experience with the complete machine learning workflow, from data engineering to model deployment. You will build two production-ready models using real-world financial datasets.

---

## Learning Objectives

By completing this capstone, you will:

1. ✅ **Understand the ML lifecycle** - From raw data to deployed models
2. ✅ **Master data engineering** - Clean, transform, and prepare data for ML
3. ✅ **Build classification models** - Predict binary outcomes (fraud, churn, default)
4. ✅ **Build regression models** - Predict continuous values (amounts, scores)
5. ✅ **Evaluate models properly** - Use appropriate metrics and visualizations
6. ✅ **Package models** - Save and version models for deployment
7. ✅ **Follow best practices** - Write clean, reproducible ML code

---

## Project Structure

### Phase 1: Choose Your Problem (15 minutes)

**Classification Options:**
- **Fraud Detection**: Predict if a transaction is fraudulent
- **Churn Prediction**: Predict if a customer will leave the bank
- **Credit Default**: Predict if a borrower will default on a loan

**Regression Options:**
- **Transaction Amount Prediction**: Predict transaction amounts
- **Loan Amount Prediction**: Predict loan amounts
- **Credit Score Prediction**: Predict credit scores

**Recommendation**: Choose problems that interest you or relate to your career goals!

---

### Phase 2: Classification Model (2-3 hours)

#### Step 1: Data Loading & Exploration (30 minutes)

**Tasks:**
- Load your chosen dataset
- Understand the data structure
- Check data types and missing values
- Explore target variable distribution
- Perform basic statistical analysis

**Key Questions:**
- How many samples do you have?
- Is your target balanced or imbalanced?
- What features are available?
- Are there any obvious data quality issues?

#### Step 2: Data Engineering (45 minutes)

**Tasks:**
- Handle missing values
- Encode categorical variables
- Engineer new features (if applicable)
- Handle class imbalance (for classification)
- Split data into train and test sets

**Best Practices:**
- Always use stratified split for imbalanced classification
- Never look at test data until final evaluation
- Document your preprocessing choices

#### Step 3: Model Training (30 minutes)

**Tasks:**
- Train 3 different models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Compare model performance
- Select the best model

**Tips:**
- Start with default parameters
- Use the helper functions in `src/model_utils.py`
- Train all models on the same data

#### Step 4: Model Evaluation (30 minutes)

**Tasks:**
- Evaluate on test set
- Calculate metrics (accuracy, precision, recall, F1, ROC-AUC)
- Create confusion matrix
- Plot ROC curve
- Analyze feature importance

**What to Look For:**
- Is your model better than random guessing?
- Are precision and recall balanced?
- Which features are most important?
- Are there signs of overfitting?

#### Step 5: Model Packaging (15 minutes)

**Tasks:**
- Save your best model
- Document model performance
- Save preprocessing artifacts (scalers, encoders)

---

### Phase 3: Regression Model (2-3 hours)

Follow the same structure as classification, with these differences:

#### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

#### Visualizations
- Actual vs Predicted scatter plot
- Residual plot
- Residual distribution

#### No Class Imbalance
- Don't need SMOTE or class weights
- Use regular train-test split (no stratify)

---

### Phase 4: Documentation & Presentation (30 minutes)

**Deliverables:**
1. Completed Jupyter notebooks
2. Saved models in `models/` folder
3. Brief summary of your findings
4. Recommendations for model improvement

---

## Workflow Checklist

### For Each Model:

- [ ] Load and explore data
- [ ] Check for missing values
- [ ] Identify feature types (numerical, categorical)
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Create train-test split
- [ ] Scale features (if needed)
- [ ] Train multiple models
- [ ] Compare model performance
- [ ] Select best model
- [ ] Evaluate on test set
- [ ] Visualize results
- [ ] Analyze feature importance
- [ ] Save model
- [ ] Document findings

---

## Common Mistakes to Avoid

### Data Leakage
❌ **Wrong**: Scaling before splitting
```python
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
```

✅ **Correct**: Split first, then scale
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Ignoring Class Imbalance
❌ **Wrong**: Using accuracy for imbalanced data
- 95% accuracy might be bad if 95% of data is one class

✅ **Correct**: Use F1, precision, recall, or ROC-AUC

### Not Using Test Set Properly
❌ **Wrong**: Making decisions based on test set performance
- Tuning hyperparameters on test set
- Selecting features using test set

✅ **Correct**: Test set is only for final evaluation

### Overfitting
❌ **Wrong**: Perfect training performance, poor test performance

✅ **Correct**: Use cross-validation, simpler models, or regularization

---

## Tips for Success

### 1. Start Simple
- Begin with basic models (Logistic Regression, Linear Regression)
- Add complexity gradually
- Understand each step before moving forward

### 2. Use the Provided Tools
- Helper functions in `src/data_engineering.py`
- Model utilities in `src/model_utils.py`
- Reference the cheatsheet in `docs/ML_CHEATSHEET.md`

### 3. Visualize Everything
- Data distributions
- Missing values
- Correlations
- Model performance
- Feature importance

### 4. Document Your Work
- Add markdown cells explaining your reasoning
- Comment your code
- Note any challenges or insights

### 5. Experiment
- Try different preprocessing techniques
- Test various models
- Compare results
- Learn from failures

---

## Evaluation Rubric

Your capstone will be evaluated on:

### Technical Implementation (50%)
- [ ] Proper data preprocessing
- [ ] Correct train-test split
- [ ] Multiple models trained and compared
- [ ] Appropriate evaluation metrics used
- [ ] Model saved correctly

### Analysis & Insights (30%)
- [ ] Data exploration performed
- [ ] Results interpreted correctly
- [ ] Feature importance analyzed
- [ ] Model limitations discussed

### Code Quality (20%)
- [ ] Clean, readable code
- [ ] Proper use of helper functions
- [ ] Good documentation
- [ ] Notebooks run without errors

---

## Troubleshooting Guide

### "My model has low accuracy"
1. Check if your data is imbalanced
2. Try feature engineering
3. Use more complex models
4. Check for data quality issues

### "Training takes too long"
1. Reduce dataset size for experimentation
2. Use a subset of features
3. Start with simpler models
4. Reduce n_estimators for tree-based models

### "I'm getting an error"
1. Check the error message carefully
2. Verify data types
3. Ensure no missing values in unexpected places
4. Check the cheatsheet for correct syntax
5. Ask for help!

### "I don't understand a concept"
1. Review the cheatsheet
2. Check sklearn documentation
3. Look at the guided sections in notebooks
4. Ask your instructor or peers

---

## Next Steps After Completion

Once you've completed the capstone:

1. **Model Deployment** - Deploy your models to H2O platform
2. **Advanced Techniques** - Try ensemble methods, hyperparameter tuning
3. **Feature Engineering** - Create more sophisticated features
4. **Model Monitoring** - Learn about tracking model performance in production
5. **Portfolio Project** - Expand this into a full portfolio piece

---

## Resources

### Documentation
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Pandas: https://pandas.pydata.org/

### In This Project
- `README.md` - Project overview and setup
- `docs/ML_CHEATSHEET.md` - Quick reference guide
- `src/data_engineering.py` - Data preprocessing functions
- `src/model_utils.py` - Model training and evaluation functions

### Get Help
- Check the solution notebooks (after attempting!)
- Reference the guided sections
- Ask your instructor
- Collaborate with peers

---

## Final Notes

Remember:
- **Machine learning is iterative** - You won't get perfect results on the first try
- **Learning from mistakes** - Errors are opportunities to learn
- **Experiment freely** - Try different approaches and see what works
- **Document everything** - Your future self will thank you
- **Ask questions** - There are no stupid questions in learning

**Good luck, and enjoy building your ML models!** 🚀
