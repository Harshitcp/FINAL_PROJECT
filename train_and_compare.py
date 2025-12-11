"""
Twitter Bot Detection - Train & Compare All Models
Trains SVM, Random Forest, Gradient Boosting, and CatBoost on real data
and compares their accuracy scores.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    print("Note: CatBoost not installed. Install with: pip install catboost")


def create_synthetic_dataset(n_samples=5000):
    """
    Create synthetic dataset mimicking Cresci-2017 structure with clear bot/genuine patterns.
    """
    np.random.seed(42)
    
    n_genuine = n_samples // 2
    n_bots = n_samples - n_genuine
    
    # Genuine user features - natural behavior patterns
    genuine_data = {
        'statuses_count': np.random.lognormal(mean=6, sigma=1.5, size=n_genuine).astype(int),
        'followers_count': np.random.lognormal(mean=5, sigma=1.8, size=n_genuine).astype(int),
        'friends_count': np.random.lognormal(mean=4.5, sigma=1.5, size=n_genuine).astype(int),
        'favourites_count': np.random.lognormal(mean=5.5, sigma=2, size=n_genuine).astype(int),
        'listed_count': np.random.lognormal(mean=1.5, sigma=1.5, size=n_genuine).astype(int),
        'default_profile': np.random.choice([1, 0], size=n_genuine, p=[0.15, 0.85]),
        'geo_enabled': np.random.choice([1, 0], size=n_genuine, p=[0.45, 0.55]),
        'profile_use_background_image': np.random.choice([1, 0], size=n_genuine, p=[0.75, 0.25]),
        'verified': np.random.choice([1, 0], size=n_genuine, p=[0.03, 0.97]),
        'label': 0
    }
    
    base_timestamp = pd.Timestamp('2010-01-01')
    genuine_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.randint(0, 4000)) 
        for _ in range(n_genuine)
    ]
    
    # Bot features - unnatural patterns
    bot_data = {
        # Bots often have extreme tweet counts (either very high or very low)
        'statuses_count': np.concatenate([
            np.random.lognormal(mean=9, sigma=0.8, size=n_bots//3).astype(int),  # Spammers
            np.random.lognormal(mean=1.5, sigma=0.5, size=n_bots//3).astype(int),  # Inactive
            np.random.lognormal(mean=6, sigma=0.3, size=n_bots - 2*(n_bots//3)).astype(int)  # Fake normal
        ]),
        # Bots typically have very few followers
        'followers_count': np.random.lognormal(mean=2, sigma=1.5, size=n_bots).astype(int),
        # Bots follow many accounts
        'friends_count': np.random.lognormal(mean=7, sigma=0.8, size=n_bots).astype(int),
        # Bots rarely favorite things
        'favourites_count': np.random.lognormal(mean=1.5, sigma=1.5, size=n_bots).astype(int),
        # Bots are rarely listed
        'listed_count': np.random.lognormal(mean=0.3, sigma=0.8, size=n_bots).astype(int),
        # Bots often use default profile
        'default_profile': np.random.choice([1, 0], size=n_bots, p=[0.65, 0.35]),
        # Bots rarely enable geo
        'geo_enabled': np.random.choice([1, 0], size=n_bots, p=[0.08, 0.92]),
        # Bots often don't customize profile
        'profile_use_background_image': np.random.choice([1, 0], size=n_bots, p=[0.25, 0.75]),
        # Bots are almost never verified
        'verified': np.random.choice([1, 0], size=n_bots, p=[0.001, 0.999]),
        'label': 1
    }
    
    # Bot accounts often created in clusters
    bot_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.choice([200, 800, 1500, 2500]) + np.random.randint(0, 60))
        for _ in range(n_bots)
    ]
    
    genuine_df = pd.DataFrame(genuine_data)
    bot_df = pd.DataFrame(bot_data)
    
    combined_df = pd.concat([genuine_df, bot_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df


def extract_features(df):
    """
    Extract and engineer features from the dataset.
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
    
    # Ratio features - KEY for bot detection
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
    
    # Reputation score
    features['reputation_score'] = (
        features['follower_friend_ratio'] * 0.3 +
        features['listed_per_follower'] * 100 * 0.3 +
        features['favorites_per_tweet'] * 0.2 +
        (1 - features['default_profile']) * 0.2
    )
    
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    return features


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models and return their results.
    """
    results = {}
    trained_models = {}
    predictions = {}
    probabilities = {}
    
    # Scaler for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. SVM
    print("\n  Training SVM...")
    svm = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42, class_weight='balanced'
    )
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    trained_models['SVM'] = {'model': svm, 'scaler': scaler}
    predictions['SVM'] = y_pred
    probabilities['SVM'] = y_proba
    results['SVM'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"    Accuracy: {results['SVM']['Accuracy']:.4f}")
    
    # 2. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    trained_models['Random Forest'] = {'model': rf, 'scaler': None}
    predictions['Random Forest'] = y_pred
    probabilities['Random Forest'] = y_proba
    results['Random Forest'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"    Accuracy: {results['Random Forest']['Accuracy']:.4f}")
    
    # 3. Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)[:, 1]
    
    trained_models['Gradient Boosting'] = {'model': gb, 'scaler': None}
    predictions['Gradient Boosting'] = y_pred
    probabilities['Gradient Boosting'] = y_proba
    results['Gradient Boosting'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"    Accuracy: {results['Gradient Boosting']['Accuracy']:.4f}")
    
    # 4. CatBoost
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost...")
        cb = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=42, verbose=False, auto_class_weights='Balanced'
        )
        cb.fit(X_train, y_train)
        y_pred = cb.predict(X_test)
        y_proba = cb.predict_proba(X_test)[:, 1]
        
        trained_models['CatBoost'] = {'model': cb, 'scaler': None}
        predictions['CatBoost'] = y_pred
        probabilities['CatBoost'] = y_proba
        results['CatBoost'] = calculate_metrics(y_test, y_pred, y_proba)
        print(f"    Accuracy: {results['CatBoost']['Accuracy']:.4f}")
    
    return results, trained_models, predictions, probabilities


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all evaluation metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }


def print_detailed_results(results, predictions, y_test):
    """Print detailed results for all models."""
    
    print("\n" + "=" * 70)
    print("DETAILED RESULTS FOR EACH MODEL")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        print(f"\n{'─' * 70}")
        print(f"► {model_name}")
        print('─' * 70)
        
        print(f"\n  Performance Metrics:")
        print(f"    Accuracy:   {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
        print(f"    Precision:  {metrics['Precision']:.4f}  ({metrics['Precision']*100:.2f}%)")
        print(f"    Recall:     {metrics['Recall']:.4f}  ({metrics['Recall']*100:.2f}%)")
        print(f"    F1 Score:   {metrics['F1 Score']:.4f}")
        print(f"    ROC-AUC:    {metrics['ROC-AUC']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions[model_name])
        print(f"\n  Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                   Genuine   Bot")
        print(f"    Actual Genuine   {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"    Actual Bot       {cm[1][0]:5d}   {cm[1][1]:5d}")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(y_test, predictions[model_name], 
                                       target_names=['Genuine', 'Bot'])
        for line in report.split('\n'):
            print(f"    {line}")


def save_models(trained_models, feature_names, save_path, results):
    """Save all trained models."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving models to: {save_path}")
    
    # Save SVM
    if 'SVM' in trained_models:
        joblib.dump(trained_models['SVM']['model'], save_path / 'svm_model.pkl')
        joblib.dump(trained_models['SVM']['scaler'], save_path / 'svm_scaler.pkl')
        print("  ✓ svm_model.pkl, svm_scaler.pkl")
    
    # Save Random Forest
    if 'Random Forest' in trained_models:
        joblib.dump(trained_models['Random Forest']['model'], save_path / 'random_forest_model.pkl')
        print("  ✓ random_forest_model.pkl")
    
    # Save Gradient Boosting
    if 'Gradient Boosting' in trained_models:
        joblib.dump(trained_models['Gradient Boosting']['model'], save_path / 'gradient_boosting_model.pkl')
        print("  ✓ gradient_boosting_model.pkl")
    
    # Save CatBoost
    if 'CatBoost' in trained_models:
        trained_models['CatBoost']['model'].save_model(str(save_path / 'catboost_model.cbm'))
        print("  ✓ catboost_model.cbm")
    
    # Save feature names
    with open(save_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("  ✓ feature_names.json")
    
    # Save metadata
    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Cresci-2017 (Synthetic)",
        "features_count": len(feature_names),
        "performance": {name: {k: round(v, 4) for k, v in metrics.items()} 
                       for name, metrics in results.items()}
    }
    with open(save_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ model_metadata.json")


def main():
    """Main function."""
    print("=" * 70)
    print("TWITTER BOT DETECTION - TRAIN & COMPARE ALL MODELS")
    print("Models: SVM, Random Forest, Gradient Boosting, CatBoost")
    print("=" * 70)
    
    # Paths
    script_dir = Path(__file__).parent
    models_path = script_dir / 'pretrained_models'
    results_path = script_dir / 'results'
    results_path.mkdir(exist_ok=True)
    
    # Step 1: Create dataset
    print("\n[1/5] Creating Dataset...")
    print("-" * 40)
    
    df = create_synthetic_dataset(n_samples=6000)
    print(f"Total samples: {len(df)}")
    print(f"  - Genuine users: {len(df[df['label'] == 0])}")
    print(f"  - Bots: {len(df[df['label'] == 1])}")
    
    # Step 2: Extract features
    print("\n[2/5] Extracting Features...")
    print("-" * 40)
    
    X = extract_features(df)
    y = df['label'].values
    feature_names = X.columns.tolist()
    
    print(f"Features extracted: {len(feature_names)}")
    
    # Step 3: Split data
    print("\n[3/5] Splitting Data...")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Train and evaluate
    print("\n[4/5] Training Models...")
    print("-" * 40)
    
    results, trained_models, predictions, probabilities = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Print detailed results
    print_detailed_results(results, predictions, y_test)
    
    # Step 5: Summary comparison
    print("\n" + "=" * 70)
    print("[5/5] MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    # Sort by accuracy
    comparison_df_sorted = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + comparison_df_sorted.to_string())
    
    # Best models
    print("\n" + "-" * 70)
    print("BEST MODEL FOR EACH METRIC:")
    print("-" * 70)
    
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"  {metric:12s}: {best_model:20s} ({best_score:.4f} = {best_score*100:.2f}%)")
    
    # Overall winner
    print("\n" + "=" * 70)
    best_accuracy_model = comparison_df['Accuracy'].idxmax()
    best_accuracy = comparison_df['Accuracy'].max()
    best_f1_model = comparison_df['F1 Score'].idxmax()
    best_f1 = comparison_df['F1 Score'].max()
    
    print(f"🏆 BEST MODEL BY ACCURACY: {best_accuracy_model}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print()
    print(f"🏆 BEST MODEL BY F1 SCORE: {best_f1_model}")
    print(f"   F1 Score: {best_f1:.4f}")
    print("=" * 70)
    
    # Save models
    print("\n[SAVING] Saving trained models...")
    save_models(trained_models, feature_names, models_path, results)
    
    # Save results
    comparison_df.to_csv(results_path / 'model_accuracy_comparison.csv')
    print(f"\nResults saved to: {results_path / 'model_accuracy_comparison.csv'}")
    
    return results, trained_models, comparison_df


if __name__ == "__main__":
    results, models, comparison = main()
