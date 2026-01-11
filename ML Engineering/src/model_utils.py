"""
Model Utilities Helper Functions
This module provides reusable functions for model training, evaluation, and persistence.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

def train_classification_models(X_train, y_train, models=None):
    """
    Train multiple classification models.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    models : dict, optional
        Dictionary of model names and instances. If None, uses default models.

    Returns:
    --------
    dict
        Dictionary of trained models
    """
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }

    trained_models = {}

    print("Training classification models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully\n")

    return trained_models


def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a classification model and print metrics.

    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    model_name : str
        Name of the model for display

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")

    return metrics


def compare_classification_models(trained_models, X_test, y_test):
    """
    Compare multiple classification models.

    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target

    Returns:
    --------
    pd.DataFrame
        Comparison dataframe with metrics for all models
    """
    results = []

    for name, model in trained_models.items():
        metrics = evaluate_classification_model(model, X_test, y_test, name)
        metrics['Model'] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")

    return comparison_df


def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """
    Plot confusion matrix for a classification model.

    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    model_name : str
        Name of the model for display
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """
    Plot ROC curve for a classification model.

    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    model_name : str
        Name of the model for display
    """
    if not hasattr(model, 'predict_proba'):
        print(f"{model_name} does not support probability predictions")
        return

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def get_feature_importance(model, feature_names, top_n=10):
    """
    Get and plot feature importance for tree-based models.

    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display

    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nTop {top_n} Important Features:")
    print(importance_df.head(top_n).to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df.head(top_n)['Feature'], importance_df.head(top_n)['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance_df


# ============================================================================
# REGRESSION MODELS
# ============================================================================

def train_regression_models(X_train, y_train, models=None):
    """
    Train multiple regression models.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    models : dict, optional
        Dictionary of model names and instances. If None, uses default models.

    Returns:
    --------
    dict
        Dictionary of trained models
    """
    if models is None:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'XGBoost': XGBRegressor(random_state=42)
        }

    trained_models = {}

    print("Training regression models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully\n")

    return trained_models


def evaluate_regression_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a regression model and print metrics.

    Parameters:
    -----------
    model : sklearn model
        Trained regression model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    model_name : str
        Name of the model for display

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"RMSE:      {metrics['rmse']:.4f}")
    print(f"MAE:       {metrics['mae']:.4f}")
    print(f"R² Score:  {metrics['r2']:.4f}")
    print(f"{'='*50}\n")

    return metrics


def compare_regression_models(trained_models, X_test, y_test):
    """
    Compare multiple regression models.

    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target

    Returns:
    --------
    pd.DataFrame
        Comparison dataframe with metrics for all models
    """
    results = []

    for name, model in trained_models.items():
        metrics = evaluate_regression_model(model, X_test, y_test, name)
        metrics['Model'] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'rmse', 'mae', 'r2']]

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")

    return comparison_df


def plot_predictions(y_test, y_pred, model_name="Model"):
    """
    Plot actual vs predicted values for regression.

    Parameters:
    -----------
    y_test : pd.Series or np.array
        Actual test values
    y_pred : np.array
        Predicted values
    model_name : str
        Name of the model for display
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred, model_name="Model"):
    """
    Plot residuals for regression model.

    Parameters:
    -----------
    y_test : pd.Series or np.array
        Actual test values
    y_pred : np.array
        Predicted values
    model_name : str
        Name of the model for display
    """
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residual Plot - {model_name}')
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual Distribution - {model_name}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, filepath):
    """
    Save trained model to disk.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.

    Parameters:
    -----------
    filepath : str
        Path to the saved model

    Returns:
    --------
    sklearn model
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


def save_predictions(y_test, y_pred, filepath):
    """
    Save predictions to CSV file.

    Parameters:
    -----------
    y_test : pd.Series or np.array
        Actual values
    y_pred : np.array
        Predicted values
    filepath : str
        Path to save predictions
    """
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predictions_df.to_csv(filepath, index=False)
    print(f"✓ Predictions saved to {filepath}")
