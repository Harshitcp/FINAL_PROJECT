# Twitter Bot Detection using Cresci-2017 Dataset

## Project Overview
This project implements Twitter bot detection using machine learning classifiers on the Cresci-2017 dataset. The dataset contains Twitter user accounts labeled as genuine users or bots (social spambots, traditional spambots, and fake followers).

## Models Implemented
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting**
- **CatBoost**

## Dataset
The Cresci-2017 dataset is a widely used benchmark for Twitter bot detection research. It includes:
- Genuine accounts (human users)
- Social spambots (type 1, 2, 3)
- Traditional spambots
- Fake followers

### Dataset Source
Download the dataset from: https://botometer.osome.iu.edu/bot-repository/

After downloading, place the CSV files in the `data/` folder.

## Features Used
The model extracts the following features from user profiles:
- `statuses_count`: Number of tweets
- `followers_count`: Number of followers
- `friends_count`: Number of accounts followed
- `favourites_count`: Number of likes
- `listed_count`: Number of lists the user is in
- `default_profile`: Whether using default profile
- `geo_enabled`: Whether geo-tagging is enabled
- `profile_use_background_image`: Whether using background image
- `verified`: Whether account is verified
- Additional derived features (ratios, account age, etc.)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download the Cresci-2017 dataset
2. Place the data files in the `data/` folder
3. Run the main script:

```bash
python main.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook bot_detection.ipynb
```

## Project Structure
```
twitter_bot_detection/
├── data/                    # Dataset folder
├── models/                  # Saved models
├── results/                 # Output results and plots
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── feature_engineering.py  # Feature extraction
│   ├── models.py           # Model definitions
│   └── evaluation.py       # Model evaluation
├── main.py                 # Main script
├── bot_detection.ipynb     # Jupyter notebook
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Results
The script will output:
- Classification accuracy for each model
- Precision, Recall, F1-Score
- ROC-AUC scores
- Confusion matrices
- Comparison plots

## License
This project is for educational and research purposes only.
