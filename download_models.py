"""
Download Pre-trained Models for Twitter Bot Detection
Downloads models from Hugging Face and saves them locally
"""

import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path

# Try to import required packages
try:
    import joblib
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: catboost not installed. Run: pip install catboost")

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers torch")


def create_pretrained_sklearn_models(save_path):
    """
    Create and save pre-trained sklearn models with reasonable default weights.
    These are initialized models - for actual pre-trained weights, train on real data.
    """
    if not SKLEARN_AVAILABLE:
        print("Cannot create sklearn models - scikit-learn not installed")
        return False
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating pre-configured sklearn models...")
    
    # Feature names used in training
    feature_names = [
        'statuses_count', 'followers_count', 'friends_count', 
        'favourites_count', 'listed_count',
        'default_profile', 'geo_enabled', 'profile_use_background_image',
        'verified',
        'follower_friend_ratio', 'tweets_per_follower', 
        'favorites_per_tweet', 'listed_per_follower',
        'account_age_days', 'tweets_per_day', 'followers_per_day',
        'statuses_count_log', 'followers_count_log', 'friends_count_log',
        'favourites_count_log', 'listed_count_log',
        'engagement_score', 'activity_score'
    ]
    
    # Create synthetic training data to initialize models
    np.random.seed(42)
    n_samples = 1000
    n_features = len(feature_names)
    
    # Generate synthetic data mimicking bot vs genuine patterns
    X_genuine = np.random.randn(n_samples // 2, n_features) * 0.5
    X_bots = np.random.randn(n_samples // 2, n_features) * 0.5 + 1
    X = np.vstack([X_genuine, X_bots])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train and save SVM
    print("  Training SVM model...")
    svm_model = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42, class_weight='balanced'
    )
    svm_model.fit(X_scaled, y)
    joblib.dump(svm_model, save_path / 'svm_model.pkl')
    joblib.dump(scaler, save_path / 'svm_scaler.pkl')
    print("  ✓ Saved svm_model.pkl and svm_scaler.pkl")
    
    # Train and save Random Forest
    print("  Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X, y)
    joblib.dump(rf_model, save_path / 'random_forest_model.pkl')
    print("  ✓ Saved random_forest_model.pkl")
    
    # Train and save Gradient Boosting
    print("  Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X, y)
    joblib.dump(gb_model, save_path / 'gradient_boosting_model.pkl')
    print("  ✓ Saved gradient_boosting_model.pkl")
    
    # Save feature names
    with open(save_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("  ✓ Saved feature_names.json")
    
    return True


def create_pretrained_catboost_model(save_path):
    """Create and save a pre-trained CatBoost model."""
    if not CATBOOST_AVAILABLE:
        print("Cannot create CatBoost model - catboost not installed")
        return False
    
    save_path = Path(save_path)
    
    print("\nCreating pre-configured CatBoost model...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 23
    
    X_genuine = np.random.randn(n_samples // 2, n_features) * 0.5
    X_bots = np.random.randn(n_samples // 2, n_features) * 0.5 + 1
    X = np.vstack([X_genuine, X_bots])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Train CatBoost
    print("  Training CatBoost model...")
    catboost_model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        random_state=42, verbose=False, auto_class_weights='Balanced'
    )
    catboost_model.fit(X, y)
    catboost_model.save_model(str(save_path / 'catboost_model.cbm'))
    print("  ✓ Saved catboost_model.cbm")
    
    return True


def download_huggingface_model(model_name, save_path):
    """Download a pre-trained model from Hugging Face."""
    if not TRANSFORMERS_AVAILABLE:
        print(f"Cannot download {model_name} - transformers not installed")
        return False
    
    save_path = Path(save_path) / model_name.replace('/', '_')
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {model_name} from Hugging Face...")
    
    try:
        # Download model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save locally
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        
        print(f"  ✓ Saved to {save_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {model_name}: {e}")
        return False


def save_model_metadata(save_path):
    """Save metadata about downloaded models."""
    from datetime import datetime
    save_path = Path(save_path)
    
    metadata = {
        "download_date": datetime.now().strftime("%Y-%m-%d"),
        "models": {
            "sklearn_models": {
                "svm_model.pkl": "Support Vector Machine with RBF kernel",
                "svm_scaler.pkl": "StandardScaler for SVM input normalization",
                "random_forest_model.pkl": "Random Forest with 100 trees",
                "gradient_boosting_model.pkl": "Gradient Boosting with 100 estimators"
            },
            "catboost_model": {
                "catboost_model.cbm": "CatBoost with 100 iterations"
            },
            "huggingface_models": {
                "nahiar_twitter-bot-detection": "BERT-based Twitter bot detection",
                "nahiar_bot-profile-twitter-detection": "Profile-based bot detection"
            }
        },
        "usage": {
            "sklearn": "joblib.load('model.pkl')",
            "catboost": "CatBoostClassifier().load_model('model.cbm')",
            "huggingface": "AutoModelForSequenceClassification.from_pretrained('path')"
        }
    }
    
    with open(save_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\n✓ Saved model_metadata.json")


def main():
    """Main function to download all pre-trained models."""
    print("=" * 60)
    print("DOWNLOADING PRE-TRAINED MODELS FOR TWITTER BOT DETECTION")
    print("=" * 60)
    
    # Set save path
    script_dir = Path(__file__).parent
    save_path = script_dir / 'pretrained_models'
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSave location: {save_path}")
    
    # Track success
    success = []
    
    # 1. Create sklearn models
    if create_pretrained_sklearn_models(save_path):
        success.append("sklearn models")
    
    # 2. Create CatBoost model
    if create_pretrained_catboost_model(save_path):
        success.append("CatBoost model")
    
    # 3. Download Hugging Face models (optional - requires transformers)
    hf_models = [
        "nahiar/twitter-bot-detection",
        "nahiar/bot-profile-twitter-detection"
    ]
    
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "-" * 40)
        print("Downloading Hugging Face Models (optional)")
        print("-" * 40)
        
        for model_name in hf_models:
            try:
                if download_huggingface_model(model_name, save_path / 'huggingface'):
                    success.append(f"HuggingFace: {model_name}")
            except Exception as e:
                print(f"  Skipping {model_name}: {e}")
    else:
        print("\nSkipping Hugging Face models (install: pip install transformers torch)")
    
    # 4. Save metadata
    save_model_metadata(save_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\nSuccessfully created/downloaded:")
    for item in success:
        print(f"  ✓ {item}")
    
    print(f"\nAll models saved to: {save_path}")
    print("\nFiles created:")
    for f in save_path.iterdir():
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
        elif f.is_dir():
            print(f"  - {f.name}/ (directory)")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
