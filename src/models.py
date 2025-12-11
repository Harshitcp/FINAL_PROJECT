"""
Machine Learning Models for Twitter Bot Detection
Implements SVM, Random Forest, Gradient Boosting, and CatBoost classifiers
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from catboost import CatBoostClassifier
import joblib
from typing import Dict, Any, Tuple


class BotDetectionModels:
    """
    Class containing all machine learning models for bot detection.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classification models."""
        
        # Support Vector Machine (SVM)
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        # CatBoost
        self.models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=False,
            auto_class_weights='Balanced'
        )
    
    def get_model(self, model_name: str):
        """
        Get a specific model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            The requested model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Get all models.
        
        Returns:
            Dictionary of model names to model objects
        """
        return self.models
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   scale_features: bool = True) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            scale_features: Whether to scale features (recommended for SVM)
            
        Returns:
            Trained model
        """
        model = self.get_model(model_name)
        
        if scale_features and model_name == 'SVM':
            X_train = self.scaler.fit_transform(X_train)
        
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model_name: str, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            X: Features to predict
            scale_features: Whether to scale features
            
        Returns:
            Predicted labels
        """
        model = self.get_model(model_name)
        
        if scale_features and model_name == 'SVM':
            X = self.scaler.transform(X)
        
        return model.predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features to predict
            scale_features: Whether to scale features
            
        Returns:
            Prediction probabilities
        """
        model = self.get_model(model_name)
        
        if scale_features and model_name == 'SVM':
            X = self.scaler.transform(X)
        
        return model.predict_proba(X)
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        model = self.get_model(model_name)
        joblib.dump(model, filepath)
        
        if model_name == 'SVM':
            joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)
        
        if model_name == 'SVM':
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            try:
                self.scaler = joblib.load(scaler_path)
            except FileNotFoundError:
                print(f"Warning: Scaler not found at {scaler_path}")


def get_hyperparameter_grids() -> Dict[str, Dict]:
    """
    Get hyperparameter grids for grid search.
    
    Returns:
        Dictionary of model names to hyperparameter grids
    """
    return {
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        },
        'CatBoost': {
            'iterations': [50, 100, 150],
            'depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    }


def tune_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
               param_grid: Dict = None, cv: int = 5, random_state: int = 42) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using grid search.
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        param_grid: Hyperparameter grid (uses default if None)
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Tuple of (best model, best parameters)
    """
    models = BotDetectionModels(random_state=random_state)
    base_model = models.get_model(model_name)
    
    if param_grid is None:
        param_grid = get_hyperparameter_grids().get(model_name, {})
    
    # Scale features for SVM
    if model_name == 'SVM':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    
    # CatBoost doesn't work well with GridSearchCV, use manual approach
    if model_name == 'CatBoost':
        best_score = 0
        best_params = {}
        
        for iterations in param_grid.get('iterations', [100]):
            for depth in param_grid.get('depth', [6]):
                for lr in param_grid.get('learning_rate', [0.1]):
                    model = CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=lr,
                        random_state=random_state,
                        verbose=False
                    )
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                    mean_score = scores.mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'iterations': iterations, 'depth': depth, 'learning_rate': lr}
        
        best_model = CatBoostClassifier(**best_params, random_state=random_state, verbose=False)
        best_model.fit(X_train, y_train)
        return best_model, best_params
    
    # Standard GridSearchCV for other models
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    # Test model initialization
    models = BotDetectionModels()
    print("Available models:", list(models.get_all_models().keys()))
    
    # Test with random data
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    
    for name in models.get_all_models().keys():
        print(f"\nTraining {name}...")
        models.train_model(name, X, y)
        predictions = models.predict(name, X)
        accuracy = (predictions == y).mean()
        print(f"{name} training accuracy: {accuracy:.4f}")
