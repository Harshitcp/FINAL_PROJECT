"""
Package initialization for Twitter Bot Detection
"""

from .data_loader import load_cresci_2017_dataset, create_synthetic_dataset, preprocess_data
from .feature_engineering import extract_features, get_feature_names
from .models import BotDetectionModels, tune_model, get_hyperparameter_grids
from .evaluation import (
    evaluate_model, compare_models, print_classification_report,
    plot_confusion_matrix, plot_all_confusion_matrices,
    plot_roc_curves, plot_precision_recall_curves,
    plot_metrics_comparison, plot_feature_importance,
    generate_report
)

__all__ = [
    # Data loading
    'load_cresci_2017_dataset',
    'create_synthetic_dataset',
    'preprocess_data',
    # Feature engineering
    'extract_features',
    'get_feature_names',
    # Models
    'BotDetectionModels',
    'tune_model',
    'get_hyperparameter_grids',
    # Evaluation
    'evaluate_model',
    'compare_models',
    'print_classification_report',
    'plot_confusion_matrix',
    'plot_all_confusion_matrices',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_metrics_comparison',
    'plot_feature_importance',
    'generate_report'
]
