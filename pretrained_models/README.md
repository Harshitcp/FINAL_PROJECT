# Pre-trained Models for Twitter Bot Detection

This folder contains pre-trained machine learning models for Twitter bot detection.

## Available Pre-trained Models

### Download Links

| Model | Source | Link |
|-------|--------|------|
| Twitter Bot Detection (BERT) | Hugging Face | https://huggingface.co/nahiar/twitter-bot-detection |
| Bot Profile Detection | Hugging Face | https://huggingface.co/nahiar/bot-profile-twitter-detection |
| Twitter Bot Detection Model | Hugging Face | https://huggingface.co/newreyy/Twitter-Bot-Detection-Model |

### GitHub Repositories with Pre-trained Models

1. **MachineLearning-Detecting-Twitter-Bots**
   - URL: https://github.com/jubins/MachineLearning-Detecting-Twitter-Bots
   - Models: Custom ML classifiers

2. **SpotBot**
   - URL: https://github.com/khnbilal/SpotBot
   - Models: ML algorithms for bot detection

3. **detecting-twitter-bots**
   - URL: https://github.com/mohammed-imad-umar/detecting-twitter-bots
   - Models: ML-based classifiers

---

## Model Files Structure

After training or downloading, save models as:

```
pretrained_models/
├── svm_model.pkl              # SVM classifier
├── svm_scaler.pkl             # StandardScaler for SVM
├── random_forest_model.pkl    # Random Forest classifier
├── gradient_boosting_model.pkl # Gradient Boosting classifier
├── catboost_model.cbm         # CatBoost model
└── feature_names.json         # Feature names used in training
```

---

## How to Download from Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Example: Load a pre-trained model
model_name = "nahiar/twitter-bot-detection"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## How to Save Your Trained Models

```python
import joblib
import json

# Save sklearn models
joblib.dump(svm_model, 'pretrained_models/svm_model.pkl')
joblib.dump(scaler, 'pretrained_models/svm_scaler.pkl')
joblib.dump(rf_model, 'pretrained_models/random_forest_model.pkl')
joblib.dump(gb_model, 'pretrained_models/gradient_boosting_model.pkl')

# Save CatBoost model
catboost_model.save_model('pretrained_models/catboost_model.cbm')

# Save feature names
with open('pretrained_models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)
```

---

## How to Load Models for Inference

```python
import joblib
from catboost import CatBoostClassifier

# Load sklearn models
svm_model = joblib.load('pretrained_models/svm_model.pkl')
scaler = joblib.load('pretrained_models/svm_scaler.pkl')
rf_model = joblib.load('pretrained_models/random_forest_model.pkl')
gb_model = joblib.load('pretrained_models/gradient_boosting_model.pkl')

# Load CatBoost model
catboost_model = CatBoostClassifier()
catboost_model.load_model('pretrained_models/catboost_model.cbm')
```
