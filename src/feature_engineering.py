"""
Feature Engineering Module for Twitter Bot Detection
Extracts and creates features from user profile data
"""

import pandas as pd
import numpy as np
from datetime import datetime


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer features from the Twitter user data.
    
    Args:
        df: Preprocessed DataFrame with user data
        
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame()
    
    # Basic count features
    count_features = ['statuses_count', 'followers_count', 'friends_count', 
                     'favourites_count', 'listed_count']
    
    for feature in count_features:
        if feature in df.columns:
            features[feature] = df[feature].fillna(0).astype(float)
    
    # Boolean features
    bool_features = ['default_profile', 'geo_enabled', 'profile_use_background_image',
                    'verified', 'protected']
    
    for feature in bool_features:
        if feature in df.columns:
            features[feature] = df[feature].fillna(0).astype(int)
    
    # Text-based features
    if 'description' in df.columns:
        features['has_description'] = (df['description'].fillna('').str.len() > 0).astype(int)
        features['description_length'] = df['description'].fillna('').str.len()
    
    if 'url' in df.columns:
        features['has_url'] = (df['url'].fillna('').str.len() > 0).astype(int)
    
    if 'name' in df.columns:
        features['name_length'] = df['name'].fillna('').str.len()
        features['name_has_digits'] = df['name'].fillna('').str.contains(r'\d', regex=True).astype(int)
    
    if 'screen_name' in df.columns:
        features['screen_name_length'] = df['screen_name'].fillna('').str.len()
        features['screen_name_has_digits'] = df['screen_name'].fillna('').str.contains(r'\d', regex=True).astype(int)
    
    # Ratio features (important for bot detection)
    epsilon = 1e-6  # Small value to avoid division by zero
    
    # Follower to following ratio (bots often have low ratio)
    if 'followers_count' in features.columns and 'friends_count' in features.columns:
        features['follower_friend_ratio'] = (
            features['followers_count'] / (features['friends_count'] + epsilon)
        )
        # Clip extreme values
        features['follower_friend_ratio'] = features['follower_friend_ratio'].clip(0, 100)
    
    # Tweets per follower
    if 'statuses_count' in features.columns and 'followers_count' in features.columns:
        features['tweets_per_follower'] = (
            features['statuses_count'] / (features['followers_count'] + epsilon)
        )
        features['tweets_per_follower'] = features['tweets_per_follower'].clip(0, 1000)
    
    # Favorites per tweet
    if 'favourites_count' in features.columns and 'statuses_count' in features.columns:
        features['favorites_per_tweet'] = (
            features['favourites_count'] / (features['statuses_count'] + epsilon)
        )
        features['favorites_per_tweet'] = features['favorites_per_tweet'].clip(0, 100)
    
    # Listed count per follower (reputation indicator)
    if 'listed_count' in features.columns and 'followers_count' in features.columns:
        features['listed_per_follower'] = (
            features['listed_count'] / (features['followers_count'] + epsilon)
        )
        features['listed_per_follower'] = features['listed_per_follower'].clip(0, 10)
    
    # Account age features
    if 'created_at' in df.columns:
        try:
            created_at = pd.to_datetime(df['created_at'], errors='coerce')
            reference_date = pd.Timestamp.now()
            
            # Account age in days
            features['account_age_days'] = (reference_date - created_at).dt.days
            features['account_age_days'] = features['account_age_days'].fillna(0).clip(0)
            
            # Tweets per day
            if 'statuses_count' in features.columns:
                features['tweets_per_day'] = (
                    features['statuses_count'] / (features['account_age_days'] + epsilon)
                )
                features['tweets_per_day'] = features['tweets_per_day'].clip(0, 1000)
            
            # Followers gained per day
            if 'followers_count' in features.columns:
                features['followers_per_day'] = (
                    features['followers_count'] / (features['account_age_days'] + epsilon)
                )
                features['followers_per_day'] = features['followers_per_day'].clip(0, 1000)
                
        except Exception as e:
            print(f"Warning: Could not process created_at: {e}")
    
    # Log transformations for count features (helps with skewed distributions)
    for feature in count_features:
        if feature in features.columns:
            features[f'{feature}_log'] = np.log1p(features[feature])
    
    # Engagement score (combination of multiple factors)
    if all(col in features.columns for col in ['followers_count', 'listed_count', 'favourites_count']):
        features['engagement_score'] = (
            np.log1p(features['followers_count']) + 
            np.log1p(features['listed_count']) * 2 + 
            np.log1p(features['favourites_count'])
        )
    
    # Activity score
    if all(col in features.columns for col in ['statuses_count', 'favourites_count']):
        features['activity_score'] = (
            np.log1p(features['statuses_count']) + 
            np.log1p(features['favourites_count'])
        )
    
    # Handle any remaining NaN values
    features = features.fillna(0)
    
    # Replace infinite values
    features = features.replace([np.inf, -np.inf], 0)
    
    return features


def get_feature_names() -> list:
    """
    Get the list of feature names used in the model.
    
    Returns:
        List of feature names
    """
    return [
        # Basic counts
        'statuses_count', 'followers_count', 'friends_count', 
        'favourites_count', 'listed_count',
        # Boolean features
        'default_profile', 'geo_enabled', 'profile_use_background_image',
        'verified', 'protected',
        # Text-based
        'has_description', 'description_length', 'has_url',
        # Ratios
        'follower_friend_ratio', 'tweets_per_follower', 
        'favorites_per_tweet', 'listed_per_follower',
        # Time-based
        'account_age_days', 'tweets_per_day', 'followers_per_day',
        # Log-transformed
        'statuses_count_log', 'followers_count_log', 'friends_count_log',
        'favourites_count_log', 'listed_count_log',
        # Composite scores
        'engagement_score', 'activity_score'
    ]


if __name__ == "__main__":
    # Test feature extraction
    from data_loader import create_synthetic_dataset, preprocess_data
    
    df = create_synthetic_dataset(100)
    df = preprocess_data(df)
    features = extract_features(df)
    
    print("Extracted features shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nFeature statistics:")
    print(features.describe())
