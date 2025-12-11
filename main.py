"""
Main Script for Twitter Bot Detection using Cresci-2017 Dataset
Compares SVM, Random Forest, Gradient Boosting, and CatBoost classifiers
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_cresci_2017_dataset, create_synthetic_dataset, preprocess_data
from feature_engineering import extract_features
from models import BotDetectionModels
from evaluation import (
    evaluate_model, compare_models, print_classification_report,
    plot_all_confusion_matrices, plot_roc_curves, plot_precision_recall_curves,
    plot_metrics_comparison, plot_feature_importance, generate_report
)

# Suppress warnings
warnings.filterwarnings('ignore')


def main():
    """
    Main function to run the Twitter bot detection pipeline.
    """
    print("=" * 70)
    print("TWITTER BOT DETECTION USING CRESCI-2017 DATASET")
    print("Comparing: SVM, Random Forest, Gradient Boosting, CatBoost")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = Path(__file__).parent / 'data'
    RESULTS_PATH = Path(__file__).parent / 'results'
    MODELS_PATH = Path(__file__).parent / 'models'
    PRETRAINED_PATH = Path(__file__).parent / 'pretrained_models'
    
    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True)
    MODELS_PATH.mkdir(exist_ok=True)
    DATA_PATH.mkdir(exist_ok=True)
    PRETRAINED_PATH.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading Dataset...")
    print("-" * 40)
    
    try:
        df = load_cresci_2017_dataset(DATA_PATH)
    except Exception as e:
        print(f"Could not load actual dataset: {e}")
        print("Using synthetic dataset for demonstration...")
        df = create_synthetic_dataset(n_samples=5000)
    
    # Preprocess data
    print("\n[2/6] Preprocessing Data...")
    print("-" * 40)
    df = preprocess_data(df)
    
    # Extract features
    print("\n[3/6] Extracting Features...")
    print("-" * 40)
    X = extract_features(df)
    y = df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  - Genuine (0): {np.sum(y == 0)}")
    print(f"  - Bot (1): {np.sum(y == 1)}")
    
    # Get feature names
    feature_names = X.columns.tolist()
    X = X.values
    
    # Split data
    print("\n[4/6] Splitting Data...")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    print("\n[5/6] Training Models...")
    print("-" * 40)
    
    bot_models = BotDetectionModels(random_state=42)
    
    # Store results
    results = {}
    predictions = {}
    probabilities = {}
    trained_models = {}
    
    # Train and evaluate each model
    model_configs = {
        'SVM': {'X_train': X_train_scaled, 'X_test': X_test_scaled, 'scale': False},
        'Random Forest': {'X_train': X_train, 'X_test': X_test, 'scale': False},
        'Gradient Boosting': {'X_train': X_train, 'X_test': X_test, 'scale': False},
        'CatBoost': {'X_train': X_train, 'X_test': X_test, 'scale': False}
    }
    
    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        model = bot_models.get_model(model_name)
        model.fit(config['X_train'], y_train)
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(config['X_test'])
        y_proba = model.predict_proba(config['X_test'])
        
        # Store predictions
        predictions[model_name] = y_pred
        probabilities[model_name] = y_proba
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_proba)
        results[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Compare models
    print("\n[6/6] Generating Results...")
    print("-" * 40)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    comparison_df = compare_models(results)
    print(comparison_df.to_string())
    
    # Print detailed classification reports
    for model_name, y_pred in predictions.items():
        print_classification_report(y_test, y_pred, model_name)
    
    # Generate and save report
    report = generate_report(results, str(RESULTS_PATH / 'comparison_report.txt'))
    print("\n" + report)
    
    # Save comparison results
    comparison_df.to_csv(RESULTS_PATH / 'model_comparison.csv')
    print(f"\nResults saved to: {RESULTS_PATH}")
    
    # Generate plots
    print("\nGenerating Visualizations...")
    
    try:
        # Confusion matrices
        plot_all_confusion_matrices(
            y_test, predictions,
            save_path=str(RESULTS_PATH / 'confusion_matrices.png')
        )
        
        # ROC curves
        plot_roc_curves(
            y_test, probabilities,
            save_path=str(RESULTS_PATH / 'roc_curves.png')
        )
        
        # Precision-Recall curves
        plot_precision_recall_curves(
            y_test, probabilities,
            save_path=str(RESULTS_PATH / 'pr_curves.png')
        )
        
        # Metrics comparison bar chart
        plot_metrics_comparison(
            results,
            save_path=str(RESULTS_PATH / 'metrics_comparison.png')
        )
        
        # Feature importance for tree-based models
        for model_name in ['Random Forest', 'Gradient Boosting', 'CatBoost']:
            plot_feature_importance(
                trained_models[model_name],
                feature_names,
                model_name,
                save_path=str(RESULTS_PATH / f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            )
            
    except Exception as e:
        print(f"Warning: Could not generate some plots: {e}")
    
    # Find and announce the best model
    print("\n" + "=" * 70)
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"🏆 BEST MODEL: {best_model[0]}")
    print(f"   F1 Score: {best_model[1]['f1_score']:.4f}")
    print(f"   ROC-AUC: {best_model[1]['roc_auc']:.4f}")
    print("=" * 70)
    
    # Save pretrained models
    print("\nSaving Pre-trained Models...")
    save_pretrained_models(trained_models, scaler, feature_names, PRETRAINED_PATH, results)
    print(f"Models saved to: {PRETRAINED_PATH}")
    
    return results, trained_models


def save_pretrained_models(trained_models, scaler, feature_names, save_path, results):
    """
    Save all trained models to the pretrained_models folder.
    """
    save_path = Path(save_path)
    
    # Save SVM model and scaler
    joblib.dump(trained_models['SVM'], save_path / 'svm_model.pkl')
    joblib.dump(scaler, save_path / 'svm_scaler.pkl')
    
    # Save Random Forest
    joblib.dump(trained_models['Random Forest'], save_path / 'random_forest_model.pkl')
    
    # Save Gradient Boosting
    joblib.dump(trained_models['Gradient Boosting'], save_path / 'gradient_boosting_model.pkl')
    
    # Save CatBoost
    trained_models['CatBoost'].save_model(str(save_path / 'catboost_model.cbm'))
    
    # Save feature names
    with open(save_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save model metadata
    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Cresci-2017",
        "features_count": len(feature_names),
        "models": {
            "SVM": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            "Random Forest": {"n_estimators": 100, "max_depth": 10},
            "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
            "CatBoost": {"iterations": 100, "depth": 6, "learning_rate": 0.1}
        },
        "performance": {name: {k: round(v, 4) for k, v in metrics.items()} 
                       for name, metrics in results.items()}
    }
    
    with open(save_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  ✓ svm_model.pkl")
    print("  ✓ svm_scaler.pkl")
    print("  ✓ random_forest_model.pkl")
    print("  ✓ gradient_boosting_model.pkl")
    print("  ✓ catboost_model.cbm")
    print("  ✓ feature_names.json")
    print("  ✓ model_metadata.json")


if __name__ == "__main__":
    results, models = main()
