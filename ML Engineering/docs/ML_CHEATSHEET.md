# Machine Learning Cheatsheet

Quick reference guide for common ML tasks and code snippets.

---

## Table of Contents
1. [Data Loading & Exploration](#data-loading--exploration)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Code Snippets](#code-snippets)

---

## Data Loading & Exploration

### Load CSV Data
```python
import pandas as pd

# Load full dataset
df = pd.read_csv('data/raw/dataset.csv')

# Load limited rows (for large datasets)
df = pd.read_csv('data/raw/dataset.csv', nrows=100000)
```

### Basic Data Exploration
```python
# Dataset shape
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# First few rows
df.head()

# Dataset info
df.info()

# Statistical summary
df.describe()

# Check data types
df.dtypes

# Column names
df.columns.tolist()
```

### Check Missing Values
```python
# Count missing values per column
df.isnull().sum()

# Percentage of missing values
(df.isnull().sum() / len(df) * 100).round(2)
```

### Check Target Distribution
```python
# For classification
df['target'].value_counts()

# Visualize
df['target'].value_counts().plot(kind='bar')
```

---

## Data Preprocessing

### Handle Missing Values

**Strategy 1: Fill with Mean/Median/Mode**
```python
# Fill with mean
df['column_name'].fillna(df['column_name'].mean(), inplace=True)

# Fill with median
df['column_name'].fillna(df['column_name'].median(), inplace=True)

# Fill with mode (most frequent)
df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)
```

**Strategy 2: Drop Missing**
```python
# Drop rows with any missing value
df.dropna(inplace=True)

# Drop rows with missing value in specific column
df.dropna(subset=['column_name'], inplace=True)
```

### Encode Categorical Variables

**One-Hot Encoding** (for nominal categories)
```python
# One-hot encode
df_encoded = pd.get_dummies(df, columns=['category_column'], drop_first=True)
```

**Label Encoding** (for ordinal categories)
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_column'] = le.fit_transform(df['category_column'])
```

### Handle Imbalanced Data (Classification)

**Using SMOTE**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Using Class Weights**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
```

---

## Feature Engineering

### Feature Scaling

**StandardScaler** (mean=0, std=1)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**MinMaxScaler** (range 0-1)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for classification with imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Model Training

### Classification Models

**Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**XGBoost Classifier**
```python
from xgboost import XGBClassifier

model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Regression Models

**Linear Regression**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Ridge Regression**
```python
from sklearn.linear_model import Ridge

model = Ridge(random_state=42, alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**XGBoost Regressor**
```python
from xgboost import XGBRegressor

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## Model Evaluation

### Classification Metrics

**Basic Metrics**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
```

**Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**ROC-AUC Score**
```python
from sklearn.metrics import roc_auc_score, roc_curve

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Regression Metrics

**Basic Metrics**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE:     {rmse:.4f}")
print(f"MAE:      {mae:.4f}")
print(f"R² Score: {r2:.4f}")
```

**Actual vs Predicted Plot**
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()
```

**Residual Plot**
```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### Feature Importance

```python
# For tree-based models (Random Forest, XGBoost)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['Feature'],
         feature_importance.head(10)['Importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```

---

## Code Snippets

### Save and Load Models

**Using Joblib**
```python
import joblib

# Save model
joblib.dump(model, 'models/my_model.pkl')

# Load model
loaded_model = joblib.load('models/my_model.pkl')
```

**Using Pickle**
```python
import pickle

# Save model
with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('models/my_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Hyperparameter Tuning (Grid Search)

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

---

## Common Metrics Explained

### Classification Metrics

- **Accuracy**: Overall correctness of predictions (TP+TN)/(TP+TN+FP+FN)
- **Precision**: Of all positive predictions, how many were correct? TP/(TP+FP)
- **Recall (Sensitivity)**: Of all actual positives, how many did we catch? TP/(TP+FN)
- **F1 Score**: Harmonic mean of precision and recall: 2*(Precision*Recall)/(Precision+Recall)
- **ROC-AUC**: Area under the ROC curve, measures overall model performance across all thresholds

**When to use what?**
- **Precision**: When false positives are costly (e.g., spam detection)
- **Recall**: When false negatives are costly (e.g., fraud detection, disease diagnosis)
- **F1**: When you need balance between precision and recall

### Regression Metrics

- **RMSE (Root Mean Squared Error)**: Square root of average squared differences. Penalizes large errors.
- **MAE (Mean Absolute Error)**: Average absolute differences. More robust to outliers.
- **R² Score**: Proportion of variance explained by the model (0 to 1, higher is better)

---

## Quick Troubleshooting

### Model is Overfitting
- Use cross-validation
- Reduce model complexity (fewer features, simpler model)
- Add regularization (Ridge, Lasso)
- Get more training data

### Model is Underfitting
- Use more complex model
- Add more features
- Reduce regularization

### Imbalanced Dataset (Classification)
- Use stratified split
- Apply SMOTE or other resampling techniques
- Use class weights
- Use appropriate metrics (F1, ROC-AUC instead of accuracy)

### Slow Training
- Reduce dataset size for experimentation
- Use simpler model first
- Reduce number of features
- Use sampling for large datasets

---

## Useful Resources

- **Scikit-learn Documentation**: https://scikit-learn.org/stable/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Helper Functions**: See `src/data_engineering.py` and `src/model_utils.py`

---

**Remember**: Machine learning is iterative. Experiment, evaluate, and refine!
