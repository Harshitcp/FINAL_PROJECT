"""
Data Loader Module for Cresci-2017 Dataset
Handles loading and preprocessing of Twitter bot detection data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_cresci_2017_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the Cresci-2017 dataset from the specified path.
    
    The dataset structure typically includes:
    - genuine_accounts/users.csv
    - social_spambots_1/users.csv
    - social_spambots_2/users.csv
    - social_spambots_3/users.csv
    - traditional_spambots_1/users.csv
    - fake_followers/users.csv
    
    Args:
        data_path: Path to the dataset folder
        
    Returns:
        Combined DataFrame with all accounts and labels
    """
    data_path = Path(data_path)
    
    # Define dataset categories and their labels
    categories = {
        'genuine_accounts': 0,  # Human (not bot)
        'social_spambots_1': 1,  # Bot
        'social_spambots_2': 1,  # Bot
        'social_spambots_3': 1,  # Bot
        'traditional_spambots_1': 1,  # Bot
        'fake_followers': 1,  # Bot
    }
    
    all_data = []
    
    for category, label in categories.items():
        category_path = data_path / category / 'users.csv'
        
        if category_path.exists():
            try:
                df = pd.read_csv(category_path, encoding='utf-8', low_memory=False)
                df['label'] = label
                df['category'] = category
                all_data.append(df)
                print(f"Loaded {len(df)} records from {category}")
            except Exception as e:
                print(f"Error loading {category}: {e}")
        else:
            print(f"Warning: {category_path} not found")
    
    if not all_data:
        print("No data files found. Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Bots: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Genuine: {len(combined_df[combined_df['label'] == 0])}")
    
    return combined_df


def create_synthetic_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create a synthetic dataset mimicking Cresci-2017 structure for demonstration.
    This allows the code to run even without the actual dataset.
    
    Args:
        n_samples: Total number of samples to generate
        
    Returns:
        Synthetic DataFrame with bot-like and genuine-like profiles
    """
    np.random.seed(42)
    
    n_genuine = n_samples // 2
    n_bots = n_samples - n_genuine
    
    # Generate genuine user features
    genuine_data = {
        'id': range(1, n_genuine + 1),
        'statuses_count': np.random.lognormal(mean=6, sigma=1.5, size=n_genuine).astype(int),
        'followers_count': np.random.lognormal(mean=4, sigma=2, size=n_genuine).astype(int),
        'friends_count': np.random.lognormal(mean=4, sigma=1.5, size=n_genuine).astype(int),
        'favourites_count': np.random.lognormal(mean=5, sigma=2, size=n_genuine).astype(int),
        'listed_count': np.random.lognormal(mean=1, sigma=1.5, size=n_genuine).astype(int),
        'default_profile': np.random.choice([True, False], size=n_genuine, p=[0.2, 0.8]),
        'geo_enabled': np.random.choice([True, False], size=n_genuine, p=[0.4, 0.6]),
        'profile_use_background_image': np.random.choice([True, False], size=n_genuine, p=[0.7, 0.3]),
        'verified': np.random.choice([True, False], size=n_genuine, p=[0.05, 0.95]),
        'protected': np.random.choice([True, False], size=n_genuine, p=[0.1, 0.9]),
        'description': ['User description' if np.random.random() > 0.1 else '' for _ in range(n_genuine)],
        'url': ['http://example.com' if np.random.random() > 0.3 else '' for _ in range(n_genuine)],
        'label': 0,
        'category': 'genuine_accounts'
    }
    
    # Generate account creation timestamps (genuine accounts: spread over years)
    base_timestamp = pd.Timestamp('2010-01-01')
    genuine_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.randint(0, 3000)) 
        for _ in range(n_genuine)
    ]
    
    # Generate bot features (different distribution patterns)
    bot_data = {
        'id': range(n_genuine + 1, n_samples + 1),
        # Bots often have very high or very low tweet counts
        'statuses_count': np.concatenate([
            np.random.lognormal(mean=8, sigma=1, size=n_bots//2).astype(int),
            np.random.lognormal(mean=2, sigma=1, size=n_bots - n_bots//2).astype(int)
        ]),
        # Bots often have unusual follower patterns
        'followers_count': np.random.lognormal(mean=2, sigma=2, size=n_bots).astype(int),
        'friends_count': np.random.lognormal(mean=6, sigma=1, size=n_bots).astype(int),  # Follow many
        'favourites_count': np.random.lognormal(mean=2, sigma=2, size=n_bots).astype(int),  # Low favorites
        'listed_count': np.random.lognormal(mean=0.5, sigma=1, size=n_bots).astype(int),
        'default_profile': np.random.choice([True, False], size=n_bots, p=[0.6, 0.4]),  # More default profiles
        'geo_enabled': np.random.choice([True, False], size=n_bots, p=[0.1, 0.9]),  # Less geo
        'profile_use_background_image': np.random.choice([True, False], size=n_bots, p=[0.3, 0.7]),
        'verified': np.random.choice([True, False], size=n_bots, p=[0.001, 0.999]),  # Almost never verified
        'protected': np.random.choice([True, False], size=n_bots, p=[0.02, 0.98]),
        'description': ['' if np.random.random() > 0.4 else 'Bot description' for _ in range(n_bots)],
        'url': ['' if np.random.random() > 0.2 else 'http://spam.com' for _ in range(n_bots)],
        'label': 1,
        'category': 'bot_accounts'
    }
    
    # Bot accounts often created in bursts
    bot_data['created_at'] = [
        base_timestamp + pd.Timedelta(days=np.random.choice([100, 500, 1000, 1500]) + np.random.randint(0, 30))
        for _ in range(n_bots)
    ]
    
    # Create DataFrames
    genuine_df = pd.DataFrame(genuine_data)
    bot_df = pd.DataFrame(bot_data)
    
    # Combine and shuffle
    combined_df = pd.concat([genuine_df, bot_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created synthetic dataset with {len(combined_df)} samples")
    print(f"Bots: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Genuine: {len(combined_df[combined_df['label'] == 0])}")
    
    return combined_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataset.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Convert boolean columns to int
    bool_columns = ['default_profile', 'geo_enabled', 'profile_use_background_image', 
                   'verified', 'protected']
    
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Handle missing values
    numeric_columns = ['statuses_count', 'followers_count', 'friends_count', 
                      'favourites_count', 'listed_count']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Convert created_at to datetime if it's a string
    if 'created_at' in df.columns:
        if df['created_at'].dtype == object:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    return df


if __name__ == "__main__":
    # Test data loading
    df = create_synthetic_dataset(1000)
    print("\nDataset columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
