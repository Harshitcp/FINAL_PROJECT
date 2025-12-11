# Data Directory

## Cresci-2017 Dataset

Download the Cresci-2017 dataset from the Bot Repository:
https://botometer.osome.iu.edu/bot-repository/

### Expected Structure

After downloading, extract the files and organize them as follows:

```
data/
├── genuine_accounts/
│   └── users.csv
├── social_spambots_1/
│   └── users.csv
├── social_spambots_2/
│   └── users.csv
├── social_spambots_3/
│   └── users.csv
├── traditional_spambots_1/
│   └── users.csv
└── fake_followers/
    └── users.csv
```

### Expected CSV Columns

The `users.csv` files should contain Twitter user profile data with columns like:
- `id`: User ID
- `name`: Display name
- `screen_name`: Username
- `statuses_count`: Number of tweets
- `followers_count`: Number of followers
- `friends_count`: Number of accounts followed
- `favourites_count`: Number of likes
- `listed_count`: Number of lists
- `created_at`: Account creation date
- `default_profile`: Boolean
- `geo_enabled`: Boolean
- `profile_use_background_image`: Boolean
- `verified`: Boolean
- `protected`: Boolean
- `description`: Bio text
- `url`: Profile URL

### Note

If the actual dataset is not available, the code will automatically generate a synthetic dataset for demonstration purposes.
