"""
Evaluation Module for Twitter Bot Detection Models
Provides comprehensive metrics and visualization for model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Any
import os


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            # Use probability of positive class
            if len(y_proba.shape) > 1:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba_pos)
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics['roc_auc'] = 0.0
            metrics['avg_precision'] = 0.0
    
    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model results.
    
    Args:
        results: Dictionary of model names to their metrics
        
    Returns:
        DataFrame with model comparisons
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    df = df.round(4)
    return df


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Classification Report for {model_name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=['Genuine', 'Bot']))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Bot'],
                yticklabels=['Genuine', 'Bot'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_confusion_matrices(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                save_path: str = None):
    """
    Plot confusion matrices for all models in a grid.
    
    Args:
        y_true: True labels
        predictions: Dictionary of model names to predictions
        save_path: Path to save the plot (optional)
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Genuine', 'Bot'],
                   yticklabels=['Genuine', 'Bot'],
                   ax=axes[idx])
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true: np.ndarray, probabilities: Dict[str, np.ndarray],
                   save_path: str = None):
    """
    Plot ROC curves for all models.
    
    Args:
        y_true: True labels
        probabilities: Dictionary of model names to prediction probabilities
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, y_proba) in enumerate(probabilities.items()):
        # Get probability of positive class
        if len(y_proba.shape) > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        auc = roc_auc_score(y_true, y_proba_pos)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(y_true: np.ndarray, probabilities: Dict[str, np.ndarray],
                                save_path: str = None):
    """
    Plot Precision-Recall curves for all models.
    
    Args:
        y_true: True labels
        probabilities: Dictionary of model names to prediction probabilities
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, y_proba) in enumerate(probabilities.items()):
        if len(y_proba.shape) > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        ap = average_precision_score(y_true, y_proba_pos)
        
        plt.plot(recall, precision, color=colors[idx % len(colors)], lw=2,
                label=f'{model_name} (AP = {ap:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """
    Plot bar chart comparing all metrics across models.
    
    Args:
        results: Dictionary of model names to their metrics
        save_path: Path to save the plot (optional)
    """
    df = pd.DataFrame(results).T
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    df_plot = df[available_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_plot.index))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric in enumerate(available_metrics):
        offset = (idx - len(available_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, df_plot[metric], width, label=metric.replace('_', ' ').title(),
                     color=colors[idx % len(colors)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names: List[str], model_name: str,
                           top_n: int = 15, save_path: str = None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to show
        save_path: Path to save the plot (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not have feature importances")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices][::-1], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(results: Dict[str, Dict[str, float]], output_path: str = None) -> str:
    """
    Generate a text report of model comparison.
    
    Args:
        results: Dictionary of model names to their metrics
        output_path: Path to save the report (optional)
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 70)
    report.append("TWITTER BOT DETECTION - MODEL COMPARISON REPORT")
    report.append("Using Cresci-2017 Dataset")
    report.append("=" * 70)
    report.append("")
    
    # Find best model for each metric
    df = pd.DataFrame(results).T
    
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 70)
    
    for metric in df.columns:
        best_model = df[metric].idxmax()
        best_value = df[metric].max()
        report.append(f"{metric.upper():20s} | Best: {best_model:20s} | Score: {best_value:.4f}")
    
    report.append("")
    report.append("DETAILED RESULTS")
    report.append("-" * 70)
    
    for model_name, metrics in results.items():
        report.append(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            report.append(f"  {metric_name:20s}: {value:.4f}")
    
    report.append("")
    report.append("=" * 70)
    
    # Overall best model (based on F1 score)
    if 'f1_score' in df.columns:
        best_overall = df['f1_score'].idxmax()
        report.append(f"BEST OVERALL MODEL (by F1 Score): {best_overall}")
    
    report.append("=" * 70)
    
    report_str = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)
    
    return report_str


if __name__ == "__main__":
    # Test evaluation functions with dummy data
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, 100)
    
    # Simulate predictions from different models
    results = {}
    predictions = {}
    probabilities = {}
    
    for model_name in ['SVM', 'Random Forest', 'Gradient Boosting', 'CatBoost']:
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        predictions[model_name] = y_pred
        probabilities[model_name] = y_proba
        results[model_name] = evaluate_model(y_true, y_pred, y_proba)
    
    print(compare_models(results))
    print("\n" + generate_report(results))
