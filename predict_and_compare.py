"""
Twitter Bot Detection - Model Prediction & Accuracy Comparison
Uses all pre-trained models (SVM, Random Forest, Gradient Boosting, CatBoost)
to make predictions and compare their accuracy scores.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. CatBoost model will be skipped.")


def create_synthetic_dataset(n_samples=5000):
    """
    Create synthetic dataset mimicking Cresci-2017 structure.
    """
    np.random.seed(42)
    
    n_genuine = n_samples // 2
    n_bots = n_samples - n_genuine
    
    # Genuine user features
    genuine_data = {
        'statuses_count': np.random.lognormal(mean=6, sigma=1.5, size=n_genuine).astype(int),
        'followers_count': np.random.lognormal(mean=4, sigma=2, size=n_genuine).astype(int),
        'friends_count': np.random.lognormal(mean=4, sigma=1.5, size=n_genuine).astype(int),
        'favourites_count': np.random.lognormal(mean=5, sigma=2, size=n_genuine).astype(int),
        'listed_count': np.random.lognormal(mean=1, sigma=1.5, size=n_genuine).astype(int),
        'default_profile': np.random.choice([1, 0], size=n_genuine, p=[0.2, 0.8]),
        'geo_enabled': np.random.choice([1, 0], size=n_genuine, p=[0.4, 0.6]),
        'profile_use_background_image': np.random.choice([1, 0], size=n_genuine, p=[0.7, 0.3]),
        'verified': np.random.choice([1, 0], size=n_genuine, p=[0.05, 0.95]),
        'label': 0
    }
    
    base_timestamp = pd.Timestamp('2010-01-01')
    genuine_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.randint(0, 3000)) 
        for _ in range(n_genuine)
    ]
    
    # Bot features
    bot_data = {
        'statuses_count': np.concatenate([
            np.random.lognormal(mean=8, sigma=1, size=n_bots//2).astype(int),
            np.random.lognormal(mean=2, sigma=1, size=n_bots - n_bots//2).astype(int)
        ]),
        'followers_count': np.random.lognormal(mean=2, sigma=2, size=n_bots).astype(int),
        'friends_count': np.random.lognormal(mean=6, sigma=1, size=n_bots).astype(int),
        'favourites_count': np.random.lognormal(mean=2, sigma=2, size=n_bots).astype(int),
        'listed_count': np.random.lognormal(mean=0.5, sigma=1, size=n_bots).astype(int),
        'default_profile': np.random.choice([1, 0], size=n_bots, p=[0.6, 0.4]),
        'geo_enabled': np.random.choice([1, 0], size=n_bots, p=[0.1, 0.9]),
        'profile_use_background_image': np.random.choice([1, 0], size=n_bots, p=[0.3, 0.7]),
        'verified': np.random.choice([1, 0], size=n_bots, p=[0.001, 0.999]),
        'label': 1
    }
    
    bot_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.choice([100, 500, 1000, 1500]) + np.random.randint(0, 30))
        for _ in range(n_bots)
    ]
    
    genuine_df = pd.DataFrame(genuine_data)
    bot_df = pd.DataFrame(bot_data)
    
    combined_df = pd.concat([genuine_df, bot_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df


def extract_features(df):
    """
    Extract features from the dataset.
    """
    features = pd.DataFrame()
    epsilon = 1e-6
    
    # Basic count features
    count_features = ['statuses_count', 'followers_count', 'friends_count', 
                     'favourites_count', 'listed_count']
    for feature in count_features:
        if feature in df.columns:
            features[feature] = df[feature].fillna(0).astype(float)
    
    # Boolean features
    bool_features = ['default_profile', 'geo_enabled', 'profile_use_background_image', 'verified']
    for feature in bool_features:
        if feature in df.columns:
            features[feature] = df[feature].fillna(0).astype(int)
    
    # Ratio features
    features['follower_friend_ratio'] = (
        features['followers_count'] / (features['friends_count'] + epsilon)
    ).clip(0, 100)
    
    features['tweets_per_follower'] = (
        features['statuses_count'] / (features['followers_count'] + epsilon)
    ).clip(0, 1000)
    
    features['favorites_per_tweet'] = (
        features['favourites_count'] / (features['statuses_count'] + epsilon)
    ).clip(0, 100)
    
    features['listed_per_follower'] = (
        features['listed_count'] / (features['followers_count'] + epsilon)
    ).clip(0, 10)
    
    # Account age features
    if 'created_at' in df.columns:
        created_at = pd.to_datetime(df['created_at'], errors='coerce')
        reference_date = pd.Timestamp.now()
        features['account_age_days'] = (reference_date - created_at).dt.days.fillna(0).clip(0)
        
        features['tweets_per_day'] = (
            features['statuses_count'] / (features['account_age_days'] + epsilon)
        ).clip(0, 1000)
        
        features['followers_per_day'] = (
            features['followers_count'] / (features['account_age_days'] + epsilon)
        ).clip(0, 1000)
    
    # Log transformations
    for feature in count_features:
        features[f'{feature}_log'] = np.log1p(features[feature])
    
    # Composite scores
    features['engagement_score'] = (
        np.log1p(features['followers_count']) + 
        np.log1p(features['listed_count']) * 2 + 
        np.log1p(features['favourites_count'])
    )
    
    features['activity_score'] = (
        np.log1p(features['statuses_count']) + 
        np.log1p(features['favourites_count'])
    )
    
    features['reputation_score'] = (
        features['follower_friend_ratio'] * 0.3 +
        features['listed_per_follower'] * 100 * 0.3 +
        features['favorites_per_tweet'] * 0.2 +
        (1 - features['default_profile']) * 0.2
    )
    
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    return features


def load_pretrained_models(models_path):
    """
    Load all pre-trained models from the pretrained_models folder.
    """
    models_path = Path(models_path)
    models = {}
    
    # Load SVM
    svm_path = models_path / 'svm_model.pkl'
    scaler_path = models_path / 'svm_scaler.pkl'
    if svm_path.exists() and scaler_path.exists():
        models['SVM'] = {
            'model': joblib.load(svm_path),
            'scaler': joblib.load(scaler_path),
            'needs_scaling': True
        }
        print("  [OK] Loaded SVM model")
    
    # Load Random Forest
    rf_path = models_path / 'random_forest_model.pkl'
    if rf_path.exists():
        models['Random Forest'] = {
            'model': joblib.load(rf_path),
            'scaler': None,
            'needs_scaling': False
        }
        print("  [OK] Loaded Random Forest model")
    
    # Load Gradient Boosting
    gb_path = models_path / 'gradient_boosting_model.pkl'
    if gb_path.exists():
        models['Gradient Boosting'] = {
            'model': joblib.load(gb_path),
            'scaler': None,
            'needs_scaling': False
        }
        print("  [OK] Loaded Gradient Boosting model")
    
    # Load CatBoost
    cb_path = models_path / 'catboost_model.cbm'
    if cb_path.exists() and CATBOOST_AVAILABLE:
        cb_model = CatBoostClassifier()
        cb_model.load_model(str(cb_path))
        models['CatBoost'] = {
            'model': cb_model,
            'scaler': None,
            'needs_scaling': False
        }
        print("  [OK] Loaded CatBoost model")
    
    return models


def evaluate_model(model_info, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics.
    """
    model = model_info['model']
    
    # Apply scaling if needed
    if model_info['needs_scaling'] and model_info['scaler'] is not None:
        X_test_transformed = model_info['scaler'].transform(X_test)
    else:
        X_test_transformed = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_transformed)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test_transformed)
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc
    }
    
    return metrics, y_pred


def print_confusion_matrix(y_test, y_pred, model_name):
    """
    Print confusion matrix for a model.
    """
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Genuine  Bot")
    print(f"  Actual Genuine   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Bot       {cm[1][0]:5d}  {cm[1][1]:5d}")


def main():
    """
    Main function to load models, make predictions, and compare accuracy.
    """
    print("=" * 70)
    print("TWITTER BOT DETECTION - MODEL PREDICTION & ACCURACY COMPARISON")
    print("=" * 70)
    
    # Paths
    script_dir = Path(__file__).parent
    models_path = script_dir / 'pretrained_models'
    
    # Step 1: Load pre-trained models
    print("\n[1/4] Loading Pre-trained Models...")
    print("-" * 40)
    
    if not models_path.exists():
        print(f"ERROR: Models folder not found at {models_path}")
        print("Please run 'python download_models.py' first.")
        return
    
    models = load_pretrained_models(models_path)
    
    if not models:
        print("ERROR: No models found!")
        return
    
    print(f"\nLoaded {len(models)} models: {', '.join(models.keys())}")
    
    # Step 2: Create/Load test dataset
    print("\n[2/4] Preparing Test Dataset...")
    print("-" * 40)
    
    df = create_synthetic_dataset(n_samples=2000)
    print(f"Dataset size: {len(df)} samples")
    print(f"  - Genuine users: {len(df[df['label'] == 0])}")
    print(f"  - Bots: {len(df[df['label'] == 1])}")
    
    # Extract features
    X = extract_features(df)
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTest set size: {len(X_test)} samples")
    
    # Step 3: Make predictions with all models
    print("\n[3/4] Making Predictions...")
    print("-" * 40)
    
    all_results = {}
    all_predictions = {}
    
    for model_name, model_info in models.items():
        print(f"\n► {model_name}:")
        metrics, y_pred = evaluate_model(model_info, X_test, y_test, model_name)
        all_results[model_name] = metrics
        all_predictions[model_name] = y_pred
        
        print(f"  Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
        print_confusion_matrix(y_test, y_pred, model_name)
    
    # Step 4: Compare all models
    print("\n" + "=" * 70)
    print("[4/4] MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.round(4)
    
    print("\n" + comparison_df.to_string())
    
    # Find best model for each metric
    print("\n" + "-" * 70)
    print("BEST MODELS BY METRIC:")
    print("-" * 70)
    
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"  {metric:12s}: {best_model:20s} ({best_score:.4f})")
    
    # Overall winner
    print("\n" + "=" * 70)
    best_f1_model = comparison_df['F1 Score'].idxmax()
    best_f1_score = comparison_df['F1 Score'].max()
    best_accuracy_model = comparison_df['Accuracy'].idxmax()
    best_accuracy = comparison_df['Accuracy'].max()
    
    print(f"🏆 BEST MODEL BY ACCURACY: {best_accuracy_model}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print()
    print(f"🏆 BEST MODEL BY F1 SCORE: {best_f1_model}")
    print(f"   F1 Score: {best_f1_score:.4f}")
    print("=" * 70)
    
    # Save results
    results_path = script_dir / 'results'
    results_path.mkdir(exist_ok=True)
    
    comparison_df.to_csv(results_path / 'model_accuracy_comparison.csv')
    print(f"\nResults saved to: {results_path / 'model_accuracy_comparison.csv'}")
    
    return all_results, comparison_df


def predict_single_user(user_data, models_path=None):
    """
    Predict if a single user is a bot using all models.
    
    Args:
        user_data: Dictionary with user features
        models_path: Path to pretrained models
        
    Returns:
        Dictionary with predictions from all models
    """
    if models_path is None:
        models_path = Path(__file__).parent / 'pretrained_models'
    
    # Load models
    models = load_pretrained_models(models_path)
    
    # Create DataFrame
    df = pd.DataFrame([user_data])
    
    # Extract features
    X = extract_features(df)
    
    predictions = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        if model_info['needs_scaling'] and model_info['scaler'] is not None:
            X_transformed = model_info['scaler'].transform(X.values)
        else:
            X_transformed = X.values
        
        pred = model.predict(X_transformed)[0]
        try:
            proba = model.predict_proba(X_transformed)[0]
            bot_probability = proba[1] if len(proba) > 1 else proba[0]
        except:
            bot_probability = float(pred)
        
        predictions[model_name] = {
            'prediction': 'Bot' if pred == 1 else 'Genuine',
            'is_bot': bool(pred),
            'bot_probability': float(bot_probability)
        }
    
    return predictions


# Example usage for single prediction
def demo_single_prediction():
    """
    Demonstrate single user prediction.
    """
    print("\n" + "=" * 70)
    print("DEMO: Single User Prediction")
    print("=" * 70)
    
    # Example suspicious user (bot-like characteristics)
    suspicious_user = {
        'statuses_count': 50000,
        'followers_count': 10,
        'friends_count': 5000,
        'favourites_count': 5,
        'listed_count': 0,
        'default_profile': 1,
        'geo_enabled': 0,
        'profile_use_background_image': 0,
        'verified': 0,
        'created_at': pd.Timestamp('2024-01-01')
    }
    
    print("\nSuspicious User Profile:")
    for key, value in suspicious_user.items():
        if key != 'created_at':
            print(f"  {key}: {value}")
    
    predictions = predict_single_user(suspicious_user)
    
    print("\nPredictions:")
    for model_name, result in predictions.items():
        print(f"  {model_name}: {result['prediction']} (Bot probability: {result['bot_probability']:.2%})")
    
    # Example genuine user
    genuine_user = {
        'statuses_count': 500,
        'followers_count': 300,
        'friends_count': 250,
        'favourites_count': 1000,
        'listed_count': 5,
        'default_profile': 0,
        'geo_enabled': 1,
        'profile_use_background_image': 1,
        'verified': 0,
        'created_at': pd.Timestamp('2015-06-15')
    }
    
    print("\n" + "-" * 40)
    print("\nGenuine User Profile:")
    for key, value in genuine_user.items():
        if key != 'created_at':
            print(f"  {key}: {value}")
    
    predictions = predict_single_user(genuine_user)
    
    print("\nPredictions:")
    for model_name, result in predictions.items():
        print(f"  {model_name}: {result['prediction']} (Bot probability: {result['bot_probability']:.2%})")


if __name__ == "__main__":
    # Run main comparison
    results, comparison = main()
    
    # Run demo single prediction
    demo_single_prediction()
